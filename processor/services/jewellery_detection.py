"""Detect the jewellery in a photo and cut it out accurately - fast.

Pipeline
--------
1. **Locate** - a small, fast model (u2netp, ~0.3 s) runs on a reduced copy
   of the photo (<= 1280 px). Its matte, plus pixels that clearly differ from
   the fitted backdrop, is split into parts; dust is dropped and the rest
   gives the product boxes. Optionally a text-prompt detector (Grounding
   DINO) restricts the result to "jewelry" boxes.
2. **Cut out** - the accurate model (BRIA / BiRefNet) runs *once*, on the
   original-resolution crop around the jewellery, so the product fills the
   model's input: thin chains, prongs and stone outlines are segmented at
   several times the resolution of a whole-photo pass.
3. **Clean** - backdrop inside hollows, soft shadows, haze and dust are
   removed; faded chain links are restored (hole_cleanup.py). This runs only
   on the crop, with the backdrop model fitted on the small copy.
4. **Validate** - warnings for cut-off products, blur and empty detections.

Everything outside the crop is transparent by construction, so large photos
cost little more than small ones.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from . import hole_cleanup
from .model_sessions import predict_alpha, resolve_detection_model, resolve_model_name

ANALYSIS_SIDE = 1280

DETECTION_DEFAULTS = {
    "enabled": True,
    "detector": "saliency",            # "saliency" or "prompt"
    "prompt": "jewelry",
    "prompt_box_threshold": 0.25,
    "detect_alpha": 64,
    "detect_color_tolerance": 30.0,
    "min_part_ratio": 0.00003,         # of image area
    "refine": True,                    # accurate model on the crop (always fast path)
    "refine_margin": 0.12,             # padding around the box, fraction of its long side
}


@dataclass
class CutoutResult:
    alpha_roi: np.ndarray | None = None             # matte inside ``roi``
    background_small: np.ndarray | None = None      # local backdrop colour of the roi, 1/4 size
    roi: tuple = (0, 0, 0, 0)                       # x0, y0, x1, y1 in the photo
    size: tuple = (0, 0)                            # photo (w, h)
    boxes: list = field(default_factory=list)       # [x0, y0, x1, y1] px, per item
    bbox: list | None = None                        # union box, px
    coverage: float = 0.0
    refined: bool = False
    model: str = ""
    detector: str = "saliency"
    warnings: list = field(default_factory=list)
    cleanup: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    @property
    def alpha(self) -> np.ndarray:
        """Full-size matte (built on demand)."""
        w, h = self.size
        full = np.zeros((h, w), np.uint8)
        if self.alpha_roi is not None:
            x0, y0, x1, y1 = self.roi
            full[y0:y1, x0:x1] = self.alpha_roi
        return full

    def as_dict(self, width: int, height: int) -> dict:
        def norm(box):
            x0, y0, x1, y1 = box
            return [round(x0 / width, 5), round(y0 / height, 5), round(x1 / width, 5), round(y1 / height, 5)]

        return {
            "bbox": norm(self.bbox) if self.bbox else None,
            "boxes": [norm(b) for b in self.boxes],
            "coverage": round(self.coverage, 5),
            "refined": self.refined,
            "model": self.model,
            "detector": self.detector,
            "parts": len(self.boxes),
            "cleanup": self.cleanup,
            "timings": self.timings,
        }


def _settings(bg_settings: dict | None) -> dict:
    cfg = dict(DETECTION_DEFAULTS)
    cfg.update((bg_settings or {}).get("detection", {}) or {})
    return cfg


# ----------------------------------------------------------------------
# Detection (runs on the reduced analysis copy)
# ----------------------------------------------------------------------

def find_product_parts(source_rgb: np.ndarray, coarse_alpha: np.ndarray, cfg: dict,
                       background: np.ndarray | None = None):
    """Return (boxes, union_box) for the jewellery in the photo.

    Islands are grouped by proximity (chain links, stones, clasp -> one item).
    A group is kept only if the model saw part of it, so dust and backdrop
    blemishes far from the product are ignored.
    """
    h, w = coarse_alpha.shape
    diag = float(np.hypot(h, w))
    model_mask = coarse_alpha >= int(cfg["detect_alpha"])
    if not model_mask.any():
        return [], None

    if background is None:
        background = hole_cleanup.background_model(source_rgb.astype(np.float32), coarse_alpha, 0.03 * diag)
    diff = source_rgb.astype(np.float32) - background
    color_dist = np.sqrt((diff * diff).sum(axis=2))
    distinct = (color_dist > float(cfg["detect_color_tolerance"])).astype(np.uint8)
    distinct = cv2.morphologyEx(distinct, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    combined = (model_mask | distinct.astype(bool)).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    if count <= 1:
        return [], None
    cluster_of, cluster_area = hole_cleanup.cluster_components(combined, labels, count, 0.02 * diag)

    # clusters the model actually saw
    seen = np.zeros(count, bool)
    seen[np.unique(labels[model_mask])] = True
    seen_clusters = {cluster_of[i] for i in range(1, count) if seen[i]}
    if not seen_clusters:
        return [], None
    largest = max(cluster_area[c] for c in seen_clusters)
    min_area = max(9.0, float(cfg["min_part_ratio"]) * h * w, 0.01 * largest)

    groups: dict[int, list] = {}
    for idx in range(1, count):
        cluster = cluster_of[idx]
        if cluster not in seen_clusters or cluster_area[cluster] < min_area:
            continue
        x, y, bw, bh = (int(v) for v in stats[idx, :4])
        box = groups.get(cluster)
        if box is None:
            groups[cluster] = [x, y, x + bw, y + bh, cluster_area[cluster]]
        else:
            box[0], box[1] = min(box[0], x), min(box[1], y)
            box[2], box[3] = max(box[2], x + bw), max(box[3], y + bh)
    if not groups:
        return [], None
    boxes = [b[:4] for b in sorted(groups.values(), key=lambda b: -b[4])]
    return boxes, _union(boxes)


def boxes_from_alpha(alpha: np.ndarray, offset=(0, 0), max_side: int = 1024):
    """Item boxes from a finished matte (for display / validation)."""
    h, w = alpha.shape
    scale = min(1.0, max_side / float(max(h, w)))
    small = alpha if scale >= 1 else cv2.resize(alpha, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    solid = (small > 127).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    if count <= 1:
        return [], None
    sh, sw = solid.shape
    cluster_of, cluster_area = hole_cleanup.cluster_components(solid, labels, count, 0.02 * np.hypot(sh, sw))
    largest = max(cluster_area.values())
    groups: dict[int, list] = {}
    for idx in range(1, count):
        c = cluster_of[idx]
        if cluster_area[c] < 0.01 * largest:
            continue
        x, y, bw, bh = (int(v) for v in stats[idx, :4])
        b = groups.setdefault(c, [x, y, x + bw, y + bh, cluster_area[c]])
        b[0], b[1], b[2], b[3] = min(b[0], x), min(b[1], y), max(b[2], x + bw), max(b[3], y + bh)
    ox, oy = offset
    boxes = []
    for b in sorted(groups.values(), key=lambda b: -b[4]):
        boxes.append([
            ox + int(b[0] / scale), oy + int(b[1] / scale),
            ox + int(np.ceil(b[2] / scale)), oy + int(np.ceil(b[3] / scale)),
        ])
    return (boxes, _union(boxes)) if boxes else ([], None)


def _union(boxes):
    return [
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    ]


def prompt_boxes(image: Image.Image, cfg: dict):
    """Optional text-prompt detector. Returns boxes or None if unavailable."""
    try:
        from .prompt_detector import detect_boxes
    except Exception:
        return None
    try:
        return detect_boxes(
            image,
            prompt=str(cfg.get("prompt") or "jewelry"),
            threshold=float(cfg.get("prompt_box_threshold", 0.25)),
        )
    except Exception:
        return None


def _expand_box(box, margin, w, h):
    x0, y0, x1, y1 = box
    pad = int(round(max(x1 - x0, y1 - y0) * margin)) + 4
    return [max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)]


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

def cutout_jewellery(image: Image.Image, bg_settings: dict | None = None) -> CutoutResult:
    """Detect the jewellery in ``image`` and return a refined, cleaned matte."""
    bg_settings = bg_settings or {}
    cfg = _settings(bg_settings)
    model = resolve_model_name(bg_settings)
    detect_model = resolve_detection_model(bg_settings)
    source = image.convert("RGB")
    W, H = source.size
    timings = {}
    result = CutoutResult(size=(W, H), model=model)

    # -------- locate (reduced copy) -------------------------------------
    t0 = time.perf_counter()
    scale = min(1.0, ANALYSIS_SIDE / float(max(W, H)))
    small = source if scale >= 1 else source.resize(
        (max(1, round(W * scale)), max(1, round(H * scale))), Image.Resampling.BILINEAR, reducing_gap=2.0)
    small_rgb = np.asarray(small, dtype=np.uint8)
    sh, sw = small_rgb.shape[:2]
    coarse = predict_alpha(small, detect_model)
    timings["locate_model_s"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    small_diag = float(np.hypot(sh, sw))
    clean_cfg = hole_cleanup._cfg(bg_settings)
    coeffs = hole_cleanup.fit_background(
        small_rgb.astype(np.float32), coarse, float(clean_cfg["background_far_px"]) * small_diag)
    bg_small = hole_cleanup.eval_background(coeffs, sh, sw)

    detector_used = "saliency"
    boxes, union = [], None
    if cfg["enabled"]:
        boxes_s, union_s = find_product_parts(small_rgb, coarse, cfg, background=bg_small)
        inv = 1.0 / scale
        boxes = [[int(b[0] * inv), int(b[1] * inv), int(np.ceil(b[2] * inv)), int(np.ceil(b[3] * inv))] for b in boxes_s]
        union = _union(boxes) if boxes else None
        if str(cfg.get("detector", "saliency")).lower() == "prompt":
            prompt_found = prompt_boxes(source, cfg)
            if prompt_found is None:
                result.warnings.append(
                    "Prompt detector unavailable (install torch + transformers); used saliency detection.")
            elif prompt_found:
                detector_used = "prompt"
                boxes = [[int(v) for v in b] for b in prompt_found]
                union = _union(boxes)
    else:
        union = [0, 0, W, H]
    timings["detect_s"] = round(time.perf_counter() - t0, 2)
    result.detector = detector_used

    if union is None:
        result.warnings.append("No jewellery detected in the photo.")
        result.timings = timings
        return result

    # -------- accurate cut-out on the crop -------------------------------
    t0 = time.perf_counter()
    roi = _expand_box(union, float(cfg["refine_margin"]), W, H) if cfg["refine"] else [0, 0, W, H]
    x0, y0, x1, y1 = roi
    crop = source.crop(tuple(roi))
    alpha = predict_alpha(crop, model)
    result.refined = (x1 - x0) * (y1 - y0) < 0.9 * W * H
    timings["cutout_model_s"] = round(time.perf_counter() - t0, 2)

    if detector_used == "prompt":
        keep = np.zeros_like(alpha, dtype=bool)
        for b in boxes:
            bx0, by0, bx1, by1 = _expand_box(b, 0.04, W, H)
            keep[max(0, by0 - y0):max(0, by1 - y0), max(0, bx0 - x0):max(0, bx1 - x0)] = True
        alpha = np.where(keep, alpha, 0).astype(np.uint8)

    # -------- clean (crop only) -------------------------------------------
    t0 = time.perf_counter()
    crop_rgb = np.asarray(crop, dtype=np.uint8)
    region = (x0 / max(W - 1, 1), y0 / max(H - 1, 1), (x1 - 1) / max(W - 1, 1), (y1 - 1) / max(H - 1, 1))
    bg_roi = hole_cleanup.eval_background(coeffs, y1 - y0, x1 - x0, region)
    alpha, stats, ctx = hole_cleanup.clean_matte(
        crop_rgb, alpha, bg_settings, background=bg_roi, reference_size=(H, W))
    timings["cleanup_s"] = round(time.perf_counter() - t0, 2)

    result.alpha_roi = alpha
    result.roi = (x0, y0, x1, y1)
    bh, bw = ctx.bg.shape[:2]
    result.background_small = cv2.resize(
        ctx.bg, (max(1, bw // 4), max(1, bh // 4)), interpolation=cv2.INTER_AREA)
    result.cleanup = stats
    result.timings = timings
    final_boxes, final_union = boxes_from_alpha(alpha, offset=(x0, y0))
    result.boxes = final_boxes or boxes
    result.bbox = final_union or union
    result.coverage = float((alpha > 127).sum()) / float(W * H)
    result.warnings.extend(validate(alpha, result, ctx, bg_settings.get("_validation", {})))
    return result


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

def validate(alpha_roi, result: CutoutResult, ctx, validation: dict | None = None) -> list[str]:
    v = {
        "warn_if_product_is_cut_off": True,
        "warn_if_product_is_blurry": True,
        "warn_if_multiple_products_detected": False,
        "require_product_visible": True,
        "blur_threshold": 25.0,
    }
    v.update(validation or {})
    warnings = []
    W, H = result.size
    x0, y0, x1, y1 = result.roi
    solid = alpha_roi > 127

    if v["require_product_visible"] and solid.sum() < max(50, 0.0002 * H * W):
        warnings.append("Product is very small or not visible.")

    if v["warn_if_product_is_cut_off"]:
        edge = max(1, min(H, W) // 400)
        touching = 0
        if y0 == 0:
            touching += solid[:edge, :].sum()
        if y1 == H:
            touching += solid[-edge:, :].sum()
        if x0 == 0:
            touching += solid[:, :edge].sum()
        if x1 == W:
            touching += solid[:, -edge:].sum()
        if touching > max(3, 0.002 * (H + W)):
            warnings.append("Product touches the photo edge - it may be cut off.")

    if v["warn_if_product_is_blurry"] and solid.any():
        gray = ctx.gray
        mask = solid
        scale = 1024.0 / max(gray.shape)
        if scale < 1:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask.astype(np.uint8), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if mask.sum() > 200:
            lap = cv2.Laplacian(gray, cv2.CV_32F)
            sharpness = float(lap[mask].var())
            result.cleanup["sharpness"] = round(sharpness, 1)
            if sharpness < float(v["blur_threshold"]):
                warnings.append("Product looks blurry or out of focus.")

    if v["warn_if_multiple_products_detected"] and len(result.boxes) > 1:
        warnings.append(f"{len(result.boxes)} separate items detected in one photo.")
    return warnings


# ----------------------------------------------------------------------
# Framing helpers
# ----------------------------------------------------------------------

def alpha_bbox(rgba: Image.Image, threshold: int = 8):
    """Bounding box of pixels with alpha > threshold (fast, in PIL's C code)."""
    alpha = rgba.getchannel("A")
    return alpha.point(lambda v: 255 if v > threshold else 0).getbbox()


def crop_to_product(rgba: Image.Image, threshold: int = 8) -> Image.Image:
    box = alpha_bbox(rgba, threshold)
    return rgba.crop(box) if box else rgba


def draw_detection_overlay(image: Image.Image, boxes_px: list, max_side: int = 900) -> Image.Image:
    """Small preview of the original with detection boxes, for the UI."""
    preview = image.convert("RGB")
    scale = min(1.0, max_side / float(max(preview.size)))
    if scale < 1:
        preview = preview.resize((round(preview.width * scale), round(preview.height * scale)), Image.Resampling.LANCZOS)
    arr = np.asarray(preview).copy()
    thickness = max(2, round(max(arr.shape[:2]) / 300))
    for i, (x0, y0, x1, y1) in enumerate(boxes_px):
        color = (209, 70, 127) if i == 0 else (59, 95, 224)
        cv2.rectangle(
            arr,
            (int(x0 * scale), int(y0 * scale)),
            (int(x1 * scale), int(y1 * scale)),
            color,
            thickness,
        )
    return Image.fromarray(arr)
