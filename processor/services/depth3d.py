"""Turn a jewellery cutout into data for the interactive 3D viewer.

The viewer builds a relief mesh from a height map. Three height sources:

* ``shape``  - instant, offline. The silhouette is "inflated": height grows
  with distance from the edge (rounded profile), so a ring band becomes a
  rounded tube and a pendant a soft dome. Works well for jewellery because
  every hollow and edge goes smoothly back to zero.
* ``ai``     - Depth Anything V2 (small, ~100 MB ONNX, downloaded once from
  GitHub on first use). Picks up real shape from shading: raised settings,
  stone domes, layered necklaces.
* ``hybrid`` - default when the AI model is available: AI depth shaped by the
  silhouette inflation, so edges still close cleanly.

Two maps are returned:

* ``depth``  - base height, 16-bit packed into PNG red (high byte) and green
  (low byte) channels so the browser can read it without 8-bit banding.
* ``detail`` - fine surface relief from local brightness (engraving, facets,
  milgrain), 8-bit centred on 128. The viewer adds it with its own slider.
"""
from __future__ import annotations

import os
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DEPTH_MODEL_URL = os.getenv(
    "DEPTH_MODEL_URL",
    "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits.onnx",
)
DEPTH_MODEL_PATH = Path(
    os.getenv(
        "DEPTH_MODEL_PATH",
        str(Path.home() / ".image_labeler" / "models" / "depth_anything_v2_vits.onnx"),
    )
)
DEPTH_INPUT = 518

_session = None
_lock = threading.Lock()


# ----------------------------------------------------------------------
# AI depth
# ----------------------------------------------------------------------

def ai_depth_available(download: bool = False) -> bool:
    if DEPTH_MODEL_PATH.is_file():
        return True
    if not download:
        return False
    try:
        _download_model()
    except Exception:
        return False
    return DEPTH_MODEL_PATH.is_file()


def _download_model():
    DEPTH_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEPTH_MODEL_PATH.with_suffix(".part")
    urllib.request.urlretrieve(DEPTH_MODEL_URL, tmp)
    tmp.replace(DEPTH_MODEL_PATH)


def _get_session():
    global _session
    with _lock:
        if _session is None:
            import onnxruntime as ort

            if not DEPTH_MODEL_PATH.is_file():
                _download_model()
            _session = ort.InferenceSession(str(DEPTH_MODEL_PATH), providers=["CPUExecutionProvider"])
        return _session


def predict_ai_depth(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Relative depth in [0, 1] (1 = nearest), same size as the input."""
    session = _get_session()
    h, w = alpha.shape
    # Put the product on neutral grey: a white backdrop confuses the model's
    # sense of "near" vs "far" for bright metal.
    a = (alpha.astype(np.float32) / 255.0)[..., None]
    composed = rgb.astype(np.float32) * a + 128.0 * (1 - a)
    x = cv2.resize(composed, (DEPTH_INPUT, DEPTH_INPUT), interpolation=cv2.INTER_AREA) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    x = ((x - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
    name = session.get_inputs()[0].name
    out = session.run(None, {name: x})[0][0]
    depth = cv2.resize(out.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)

    inside = alpha > 127
    if inside.sum() < 10:
        return np.zeros_like(depth)
    lo, hi = np.percentile(depth[inside], [2, 98])
    return np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)


# ----------------------------------------------------------------------
# Shape inflation + detail
# ----------------------------------------------------------------------

def inflate(alpha: np.ndarray, roundness: float = 1.0) -> np.ndarray:
    """Rounded height from the silhouette, in [0, 1]."""
    solid = (alpha > 127).astype(np.uint8)
    if not solid.any():
        return np.zeros(alpha.shape, np.float32)
    dist = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
    # Normalise per part so a thin chain and a wide pendant both get relief:
    # mix global and local (blurred max) normalisation.
    global_max = float(dist.max())
    local_max = cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    local_max = cv2.GaussianBlur(local_max, (0, 0), 6)
    norm = 0.5 * dist / max(global_max, 1e-6) + 0.5 * dist / np.maximum(local_max, 1e-3)
    norm = np.clip(norm, 0, 1)
    # Rounded "pillow" profile 1 - (1 - d)^2: tube-like in the middle but with
    # a finite slope at the outline, so the mesh has no vertical walls (those
    # show up as streaks of stretched edge pixels when the model rotates).
    # roundness > 1 bulges more, < 1 flattens.
    height = np.clip(1.0 - (1.0 - norm) ** 2, 0, 1) ** (1.0 / max(roundness, 0.2))
    height = cv2.GaussianBlur(height.astype(np.float32), (0, 0), 1.2)
    height[solid == 0] = 0
    return height


def detail_map(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Fine relief in [-1, 1] from local brightness (bright = raised)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    sigma = max(2.0, max(gray.shape) / 160.0)
    high = gray - cv2.GaussianBlur(gray, (0, 0), sigma)
    inside = alpha > 127
    if inside.sum() < 10:
        return np.zeros_like(high)
    scale = np.percentile(np.abs(high[inside]), 98) + 1e-6
    detail = np.clip(high / scale, -1, 1)
    detail[~inside] = 0
    return detail


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

def build_relief(rgba: Image.Image, mode: str = "auto", max_side: int = 1600, roundness: float = 1.0) -> dict:
    """Return the cutout and height maps (PIL images) plus metadata."""
    rgba = rgba.convert("RGBA")
    from .jewellery_detection import crop_to_product

    rgba = crop_to_product(rgba)
    # small margin so the silhouette never touches the texture edge
    pad = max(4, round(max(rgba.size) * 0.02))
    padded = Image.new("RGBA", (rgba.width + 2 * pad, rgba.height + 2 * pad), (0, 0, 0, 0))
    padded.alpha_composite(rgba, (pad, pad))
    rgba = padded
    if max(rgba.size) > max_side:
        scale = max_side / max(rgba.size)
        rgba = rgba.resize((round(rgba.width * scale), round(rgba.height * scale)), Image.Resampling.LANCZOS)

    arr = np.asarray(rgba)
    rgb, alpha = arr[..., :3], arr[..., 3]

    warnings = []
    mode = (mode or "auto").lower()
    if mode == "auto":
        mode = "hybrid" if ai_depth_available() else "shape"

    shape_h = inflate(alpha, roundness)
    height = shape_h
    if mode in ("ai", "hybrid"):
        try:
            ai = predict_ai_depth(rgb, alpha)
            ai = cv2.GaussianBlur(ai, (0, 0), 1.5)
            if mode == "ai":
                # still force edges to zero so the mesh closes at the outline
                height = ai * np.clip(shape_h * 3.0, 0, 1)
            else:
                height = 0.55 * shape_h + 0.45 * ai * np.clip(shape_h * 2.5, 0, 1)
        except Exception as exc:  # network / model problems -> offline fallback
            warnings.append(f"AI depth unavailable ({exc}); used shape depth.")
            mode = "shape"
    height = np.clip(height, 0, 1)
    height[alpha <= 127] = 0

    h16 = np.round(height * 65535).astype(np.uint16)
    packed = np.zeros((*h16.shape, 3), np.uint8)
    packed[..., 0] = (h16 >> 8).astype(np.uint8)
    packed[..., 1] = (h16 & 255).astype(np.uint8)

    detail = detail_map(rgb, alpha)
    detail8 = np.round(detail * 127 + 128).astype(np.uint8)

    return {
        "cutout": rgba,
        "depth": Image.fromarray(packed, "RGB"),
        "detail": Image.fromarray(detail8, "L"),
        "width": rgba.width,
        "height": rgba.height,
        "mode": mode,
        "warnings": warnings,
    }
