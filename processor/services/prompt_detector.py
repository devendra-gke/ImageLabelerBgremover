"""Optional text-prompt jewellery detector (Grounding DINO).

Only used when a preset sets ``"detection": {"detector": "prompt"}``. Needs
``pip install torch transformers`` (about 2 GB); without them the pipeline
falls back to saliency detection and adds a warning to the result.

Useful for photos where the jewellery is worn (hands, necks), sits on a
display bust, or shares the frame with props/boxes: only regions matching the
prompt (default "jewelry") are kept.
"""
from __future__ import annotations

import os
import threading

from PIL import Image

PROMPT_MODEL = os.getenv("PROMPT_DETECTOR_MODEL", "IDEA-Research/grounding-dino-tiny")

_detector = None
_lock = threading.Lock()


def _load():
    global _detector
    with _lock:
        if _detector is None:
            from transformers import pipeline  # noqa: WPS433 (optional dependency)

            _detector = pipeline("zero-shot-object-detection", model=PROMPT_MODEL)
        return _detector


def _labels(prompt: str) -> list[str]:
    labels = [p.strip() for p in str(prompt).replace(";", ",").split(",") if p.strip()]
    return labels or ["jewelry"]


def detect_boxes(image: Image.Image, prompt: str = "jewelry", threshold: float = 0.25, max_objects: int = 8):
    """Return [[x0, y0, x1, y1], ...] in pixels, best first (may be empty)."""
    detector = _load()
    predictions = detector(image.convert("RGB"), candidate_labels=_labels(prompt), threshold=threshold)
    boxes = []
    for pred in sorted(predictions, key=lambda p: -float(p.get("score", 0))):
        box = pred.get("box") or {}
        try:
            boxes.append([int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])])
        except (KeyError, TypeError, ValueError):
            continue
        if len(boxes) >= max_objects:
            break
    return boxes
