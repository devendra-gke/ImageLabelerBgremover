"""Remove leftover background from enclosed gaps in a cutout.

Segmentation models often keep the background visible inside circular
hollows (bangles, rings, chain links) and in tiny gaps between parts.
This pass finds smooth regions that match the *local* background colour and
clears their alpha, while leaving textured regions (diamond facets,
sparkle, specular highlights) untouched.

The work is done on a reduced copy of the image for speed and robustness,
then the result is scaled back up to the original size.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

DEFAULTS = {
    "hole_work_size": 2400,          # longest side used for analysis
    # Euclidean RGB distance to the (vignette-aware) background model.
    "hole_core_tolerance": 14.0,     # a gap must contain pixels this close
    "hole_grow_tolerance": 34.0,     # gap may grow to this (soft shadow)
    "hole_flat_gradient": 18.0,      # max local gradient for "smooth"
    "hole_min_area_px": 4,           # smallest gap removed (at work size)
    "hole_max_area_ratio": 0.7,      # largest gap (fraction of the image)
    "hole_ring_px": 3,               # neighbourhood checked around a gap
    "hole_ring_product_ratio": 0.55,  # share of non-background ring (big gaps)
    "hole_small_ring_ratio": 0.9,    # same, for small gaps (sparkle guard)
    "hole_small_area_ratio": 0.02,   # "small" = (ratio * short side)^2 px
    "hole_ring_product_distance": 40.0,
    "hole_fringe_iterations": 2,     # pixels of halo trimmed around a gap
    "hole_fringe_tolerance": 55.0,
    "hole_fringe_max_saturation": 0.22,
}


def _background_model(src: np.ndarray, alpha: np.ndarray, far_px: int) -> np.ndarray:
    """Per-pixel background colour, fitted on pixels far from the product.

    A quadratic surface per channel follows vignetting / light fall-off, so a
    hollow in the middle of the picture is compared with the backdrop that
    would be there, not with the colour at the image border.
    """
    h, w = alpha.shape
    outside = (alpha == 0).astype(np.uint8)
    far = cv2.distanceTransform(outside, cv2.DIST_L2, 3) > far_px

    ys, xs = np.nonzero(far)
    if len(ys) < 500:
        b = max(5, min(h, w) // 100)
        edge = np.zeros((h, w), dtype=bool)
        edge[:b, :] = edge[-b:, :] = edge[:, :b] = edge[:, -b:] = True
        pixels = src[edge & (alpha < 32)]
        if len(pixels) < 50:
            pixels = src[edge]
        return np.broadcast_to(np.median(pixels, axis=0), (h, w, 3))

    step = max(1, len(ys) // 30000)
    ys, xs = ys[::step], xs[::step]

    def features(x, y):
        x = x / max(w - 1, 1) * 2.0 - 1.0
        y = y / max(h - 1, 1) * 2.0 - 1.0
        return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)

    design = features(xs.astype(np.float32), ys.astype(np.float32))
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    grid = features(gx, gy)

    model = np.empty((h, w, 3), dtype=np.float32)
    for c in range(3):
        values = src[ys, xs, c]
        coeffs, *_ = np.linalg.lstsq(design, values, rcond=None)
        lo, hi = np.percentile(values, [1, 99])
        model[..., c] = np.clip(grid @ coeffs, lo, hi)
    return model


def _find_gaps(src: np.ndarray, alpha: np.ndarray, cfg: dict) -> np.ndarray:
    h, w = alpha.shape
    far_px = max(6, int(min(h, w) * 0.01))
    bg = _background_model(src, alpha, far_px)
    dist = np.sqrt(((src - bg) ** 2).sum(axis=2))

    gray = cv2.cvtColor(src.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy) / 4.0

    visible = alpha > 0
    flat = grad <= float(cfg["hole_flat_gradient"])
    core_tol = float(cfg["hole_core_tolerance"])
    grow_tol = float(cfg["hole_grow_tolerance"])
    core = visible & flat & (dist <= core_tol)
    loose = visible & flat & (dist <= grow_tol)

    remove = np.zeros((h, w), dtype=np.uint8)
    if not core.any():
        return remove

    count, labels = cv2.connectedComponents(loose.astype(np.uint8), connectivity=8)
    has_core = np.zeros(count, dtype=bool)
    has_core[np.unique(labels[core])] = True
    has_core[0] = False
    candidate = has_core[labels]

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8
    )

    min_area = int(cfg["hole_min_area_px"])
    max_area = float(cfg["hole_max_area_ratio"]) * h * w
    small_area = (float(cfg["hole_small_area_ratio"]) * min(h, w)) ** 2
    ring_px = max(1, int(cfg["hole_ring_px"]))
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ring_px * 2 + 1, ring_px * 2 + 1)
    )
    product_distance = float(cfg["hole_ring_product_distance"])
    big_ratio = float(cfg["hole_ring_product_ratio"])
    small_ratio = float(cfg["hole_small_ring_ratio"])

    for label in range(1, count):
        x, y, bw, bh, area = stats[label]
        if area < min_area or area > max_area:
            continue

        touches_border = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        is_small = area < small_area

        pad = ring_px + 2
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + bw + pad, w), min(y + bh + pad, h)
        comp = (labels[y0:y1, x0:x1] == label).astype(np.uint8)
        comp_bool = comp.astype(bool)
        local_dist = dist[y0:y1, x0:x1]

        if is_small:
            # Sparkle on a diamond is also smooth and bright. Small gaps must
            # match the backdrop closely...
            if np.median(local_dist[comp_bool]) > core_tol * 1.5:
                continue

        if not touches_border:
            ring = cv2.dilate(comp, ring_kernel) & (1 - comp)
            ring_pixels = local_dist[ring.astype(bool)]
            if ring_pixels.size == 0:
                continue
            # ...and be bordered almost entirely by the product (metal), not
            # by pale facets that look like the backdrop.
            needed = small_ratio if is_small else big_ratio
            if (ring_pixels > product_distance).mean() < needed:
                continue

        remove[labels == label] = 1

    if not remove.any():
        return remove

    # Trim the pale halo that blends a gap into the metal: grow into
    # low-saturation pixels that are still fairly close to the backdrop.
    hsv = cv2.cvtColor(src.astype(np.uint8), cv2.COLOR_RGB2HSV)
    neutral = (hsv[..., 1] / 255.0) <= float(cfg["hole_fringe_max_saturation"])
    allowed = neutral & (dist <= float(cfg["hole_fringe_tolerance"])) & visible
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(int(cfg["hole_fringe_iterations"])):
        grown = cv2.dilate(remove, kernel).astype(bool) & allowed
        remove = (remove.astype(bool) | grown).astype(np.uint8)

    return remove


def remove_enclosed_background(
    source: Image.Image,
    cutout: Image.Image,
    settings: dict | None = None,
) -> Image.Image:
    """Return `cutout` with background-coloured gaps made transparent."""
    cfg = {**DEFAULTS, **(settings or {})}

    rgba = np.asarray(cutout.convert("RGBA"), dtype=np.uint8).copy()
    h, w = rgba.shape[:2]

    work = float(cfg["hole_work_size"])
    scale = min(1.0, work / max(h, w))
    small_size = (max(1, round(w * scale)), max(1, round(h * scale)))

    src_img = source.convert("RGB")
    if src_img.size != (w, h):
        src_img = src_img.resize((w, h), Image.Resampling.LANCZOS)

    if scale < 1.0:
        src_small = np.asarray(
            src_img.resize(small_size, Image.Resampling.BOX), dtype=np.float32
        )
        alpha_small = cv2.resize(rgba[..., 3], small_size, interpolation=cv2.INTER_AREA)
    else:
        src_small = np.asarray(src_img, dtype=np.float32)
        alpha_small = rgba[..., 3]

    remove_small = _find_gaps(src_small, alpha_small, cfg)
    if not remove_small.any():
        return Image.fromarray(rgba, mode="RGBA")

    if scale < 1.0:
        remove = cv2.resize(
            remove_small.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
        ) >= 0.5
    else:
        remove = remove_small.astype(bool)

    rgba[remove, 3] = 0
    return Image.fromarray(rgba, mode="RGBA")
