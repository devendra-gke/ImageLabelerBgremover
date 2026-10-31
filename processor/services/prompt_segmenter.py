"""Prompt-based background removal.

Pipeline: Grounding DINO (text prompt -> boxes) + SAM (boxes -> masks).
The returned RGBA image keeps the source RGB and uses the SAM mask as alpha.
Models are loaded lazily on first use and cached.
"""
from __future__ import annotations

import os
import threading

import numpy as np
import torch
from PIL import Image, ImageFilter

DETECTOR_MODEL_ID = os.getenv(
    "PROMPT_DETECTOR_MODEL",
    "IDEA-Research/grounding-dino-tiny",
)
SAM_MODEL_ID = os.getenv(
    "PROMPT_SAM_MODEL",
    "facebook/sam-vit-base",
)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_lock = threading.Lock()
_models: dict = {}


def _load_models():
    with _lock:
        if _models:
            return _models

        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            SamModel,
            SamProcessor,
        )

        _models["det_processor"] = AutoProcessor.from_pretrained(
            DETECTOR_MODEL_ID
        )
        _models["det_model"] = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                DETECTOR_MODEL_ID
            ).to(_DEVICE).eval()
        )
        _models["sam_processor"] = SamProcessor.from_pretrained(SAM_MODEL_ID)
        _models["sam_model"] = (
            SamModel.from_pretrained(SAM_MODEL_ID).to(_DEVICE).eval()
        )
        return _models


def _split_prompts(prompt: str) -> list[str]:
    parts = [p.strip().lower() for p in prompt.replace(";", ",").split(",")]
    return [p for p in parts if p]


def segment_by_prompt(
    image: Image.Image,
    prompt: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    max_objects: int = 5,
) -> Image.Image:
    """Return an RGBA image where only objects matching `prompt` are opaque.

    `prompt` may list several objects separated by commas,
    e.g. "necklace, earrings".
    """
    labels = _split_prompts(prompt)
    if not labels:
        raise ValueError("Prompt is empty.")

    source = image.convert("RGB")
    width, height = source.size
    models = _load_models()

    # 1. Text prompt -> bounding boxes.
    det_processor = models["det_processor"]
    det_inputs = det_processor(
        images=source,
        text=[labels],
        return_tensors="pt",
    ).to(_DEVICE)

    with torch.no_grad():
        det_outputs = models["det_model"](**det_inputs)

    detections = det_processor.post_process_grounded_object_detection(
        det_outputs,
        det_inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(height, width)],
    )[0]

    boxes = detections["boxes"].cpu()
    scores = detections["scores"].cpu()

    if len(boxes) == 0:
        raise RuntimeError(f"No object matching '{prompt}' was found.")

    order = torch.argsort(scores, descending=True)[:max_objects]
    boxes = boxes[order].tolist()

    # 2. Boxes -> masks.
    sam_processor = models["sam_processor"]
    sam_inputs = sam_processor(
        source,
        input_boxes=[boxes],
        return_tensors="pt",
    ).to(_DEVICE)

    with torch.no_grad():
        sam_outputs = models["sam_model"](**sam_inputs, multimask_output=False)

    masks = sam_processor.image_processor.post_process_masks(
        sam_outputs.pred_masks.cpu(),
        sam_inputs["original_sizes"].cpu(),
        sam_inputs["reshaped_input_sizes"].cpu(),
    )[0]  # (num_boxes, 1, H, W) boolean (logit > 0)

    # Binary interior keeps the product fully opaque (soft probabilities made
    # it semi-transparent and dark over a coloured background). Only the edge
    # is feathered.
    solid = masks[:, 0].numpy().any(axis=0)
    alpha_img = Image.fromarray((solid * 255).astype(np.uint8), mode="L")
    alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=0.8))
    alpha_u8 = np.asarray(alpha_img, dtype=np.uint8)

    result = source.convert("RGBA")
    result.putalpha(Image.fromarray(alpha_u8, mode="L"))
    return result
