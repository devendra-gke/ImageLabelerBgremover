"""Lazy, shared ONNX model sessions.

Loading a segmentation model takes several seconds and around 1 GB of RAM,
so every model is loaded once and then reused for all requests. Models are
warmed up in a background thread when the server starts (see apps.py), so
the first image does not pay the load cost.

Speed notes
-----------
* GPU is used automatically when available: CUDA (NVIDIA, needs
  onnxruntime-gpu), DirectML (any Windows GPU, needs onnxruntime-directml),
  OpenVINO (Intel CPU/iGPU, needs onnxruntime-openvino). Override with
  ``ONNX_PROVIDERS=DmlExecutionProvider,CPUExecutionProvider``.
* One inference runs at a time per model (``_infer_locks``): two parallel
  requests then overlap their CPU clean-up work with the other's model run
  instead of fighting over the same cores and doubling peak RAM.
"""
from __future__ import annotations

import os
import threading

import numpy as np
from PIL import Image

DEFAULT_SEGMENTATION_MODEL = "bria-rmbg"
# Small, fast model used only to *locate* the jewellery. The accurate model
# then runs once, on the crop around it.
DEFAULT_DETECTION_MODEL = "u2netp"

KNOWN_MODELS = (
    "bria-rmbg",
    "birefnet-general",
    "birefnet-general-lite",
    "birefnet-massive",
    "isnet-general-use",
    "u2net",
    "u2netp",
)

_sessions: dict[str, object] = {}
_load_lock = threading.Lock()
_infer_locks: dict[str, threading.Lock] = {}


def resolve_model_name(bg_settings: dict | None = None) -> str:
    """Env var wins (handy on low-RAM machines), then the preset, then default."""
    env_model = os.getenv("BG_REMOVAL_MODEL", "").strip()
    if env_model:
        return env_model
    model = str((bg_settings or {}).get("model") or "").strip()
    return model or DEFAULT_SEGMENTATION_MODEL


def resolve_detection_model(bg_settings: dict | None = None) -> str:
    env_model = os.getenv("DETECTION_MODEL", "").strip()
    if env_model:
        return env_model
    detection = (bg_settings or {}).get("detection") or {}
    return str(detection.get("model") or DEFAULT_DETECTION_MODEL).strip()


def _providers() -> list[str]:
    import onnxruntime as ort

    override = os.getenv("ONNX_PROVIDERS", "").strip()
    if override:
        return [p.strip() for p in override.split(",") if p.strip()]
    available = ort.get_available_providers()
    for gpu in ("CUDAExecutionProvider", "DmlExecutionProvider", "ROCMExecutionProvider", "OpenVINOExecutionProvider"):
        if gpu in available:
            return [gpu, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def active_providers() -> list[str]:
    try:
        return _providers()
    except Exception:
        return ["CPUExecutionProvider"]


def get_session(model_name: str):
    with _load_lock:
        session = _sessions.get(model_name)
        if session is None:
            import onnxruntime as ort
            from rembg import new_session

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            if os.getenv("ONNX_LOW_MEMORY", "0") == "1":
                # Smaller peak RAM (no pre-allocated arena), slightly slower.
                opts.enable_cpu_mem_arena = False
                opts.enable_mem_pattern = False
            threads = int(os.getenv("ONNX_THREADS", "0") or 0)
            if threads > 0:
                opts.intra_op_num_threads = threads
            session = new_session(model_name, sess_opts=opts, providers=_providers())
            _sessions[model_name] = session
            _infer_locks[model_name] = threading.Lock()
        return session


def predict_alpha(image: Image.Image, model_name: str, max_side: int | None = None) -> np.ndarray:
    """Return the model's soft foreground matte (uint8, same size as image).

    ``max_side`` downsizes the input first; the mask is returned at that
    reduced size (callers that only need to *locate* the product use this to
    avoid resizing huge masks).
    """
    from rembg import remove as rembg_remove

    image = image.convert("RGB")
    if max_side and max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.BILINEAR,
        )
    session = get_session(model_name)
    with _infer_locks[model_name]:
        mask = rembg_remove(image, session=session, only_mask=True, post_process_mask=False)
    if not isinstance(mask, Image.Image):
        from io import BytesIO

        mask = Image.open(BytesIO(mask))
    mask = mask.convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.BILINEAR)
    return np.asarray(mask, dtype=np.uint8).copy()


def warm_up(model_names: list[str]) -> None:
    """Load models and run one tiny inference so the first request is fast."""
    dummy = Image.new("RGB", (64, 64), (240, 240, 240))
    for name in model_names:
        try:
            predict_alpha(dummy, name)
        except Exception as exc:  # never block server start-up
            print(f"[warm-up] {name}: {exc}")
