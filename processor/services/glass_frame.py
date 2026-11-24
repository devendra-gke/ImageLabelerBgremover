"""Keep a magnifier / zoom-lens frame that is printed on a catalogue photo.

Some catalogue photos carry an inset "glass": a stadium-shaped rim (white
band with dark edge lines and tick marks) around an enlarged view of the
piece. Background removal treats the white band as backdrop and cuts the
rim to pieces. This module finds the frame from its dark lines and puts the
whole rim back, with the inside left to the normal cut-out (so the chosen
background colour shows through the lens).

Detection is purely geometric and gated, so photos without a lens frame are
not touched:
  * dark, neutral (grey/black) pixels are joined into components,
  * the frame is one large, wide component whose dark pixels all lie close
    to its own convex outline,
  * the rim is the band between that outline and the outline pulled in by
    ``rim_ratio`` of the frame height.
"""
from __future__ import annotations

import cv2
import numpy as np

DEFAULTS = {
    "preserve_glass_frame": True,
    "glass_dark_lum": 140,        # dark line threshold (0-255) on the proxy
    "glass_max_chroma": 38,       # neutral grey, not gold/ruby
    "glass_min_width": 0.25,      # frame width / image width
    "glass_min_height": 0.07,     # frame height / image height
    "glass_min_aspect": 1.4,
    "glass_max_aspect": 9.0,
    "glass_edge_share": 0.80,     # dark pixels within the outer edge zone
    "glass_outline_coverage": 0.9,  # share of the outline traced by dark pixels
    "glass_rim_ratio": 0.085,     # rim width / frame height
    "glass_proxy_side": 1750,
}


def _largest_frame(rgb: np.ndarray, cfg: dict):
    """Return (hull_mask, frame_h) on the proxy image or None."""
    h, w = rgb.shape[:2]
    rgb_i = rgb.astype(np.int16)
    lum = rgb_i.mean(axis=2)
    chroma = rgb_i.max(axis=2) - rgb_i.min(axis=2)
    dark = ((lum < cfg["glass_dark_lum"]) & (chroma < cfg["glass_max_chroma"])).astype(np.uint8)
    if dark.sum() < 500:
        return None

    k = max(3, int(round(max(h, w) / 190))) | 1          # ~9 px at 1750
    joined = cv2.dilate(dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)

    best = None
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < cfg["glass_min_width"] * w or bh < cfg["glass_min_height"] * h:
            continue
        aspect = bw / max(bh, 1)
        if not (cfg["glass_min_aspect"] <= aspect <= cfg["glass_max_aspect"]):
            continue
        comp_dark = (labels == i) & (dark > 0)
        if comp_dark.sum() < 4 * max(bw, bh) / 3:
            continue
        ys, xs = np.nonzero(comp_dark)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        hull = cv2.convexHull(pts)
        hull_mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(hull_mask, hull, 1)

        # Every dark pixel of a lens frame sits on its outline: near the
        # hull border, not scattered over the inside of the piece.
        edge_zone = hull_mask.astype(bool) & ~(
            cv2.erode(hull_mask, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (int(bh * 0.4) | 1, int(bh * 0.4) | 1))) > 0)
        share = float(edge_zone[comp_dark].mean())
        if share < cfg["glass_edge_share"]:
            continue
        # A lens frame is a closed loop: its whole outline is traced by dark
        # pixels. A bead chain or a shadow only covers part of its hull.
        contour = hull_mask - cv2.erode(hull_mask, np.ones((3, 3), np.uint8))
        reach = int(max(5, bh * 0.07)) * 2 + 1
        near_dark = cv2.dilate(dark, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (reach, reach)))
        coverage = float(near_dark[contour > 0].mean())
        if coverage < cfg["glass_outline_coverage"]:
            continue
        score = int(hull_mask.sum())
        if best is None or score > best[0]:
            best = (score, hull_mask, bh)
    if best is None:
        return None
    return best[1], best[2]


def glass_frame_mask(source_rgb: np.ndarray, settings: dict | None = None):
    """Soft 0..1 mask (full resolution) of the lens rim, or None."""
    cfg = {**DEFAULTS, **(settings or {})}
    if not cfg["preserve_glass_frame"]:
        return None
    H, W = source_rgb.shape[:2]
    scale = min(1.0, cfg["glass_proxy_side"] / max(H, W))
    if scale < 1.0:
        proxy = cv2.resize(source_rgb, (round(W * scale), round(H * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        proxy = source_rgb
    found = _largest_frame(proxy, cfg)
    if found is None:
        return None
    hull, frame_h = found

    rim_w = max(3, int(round(frame_h * cfg["glass_rim_ratio"]))) * 2 + 1
    inner = cv2.erode(hull, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rim_w, rim_w)))
    rim = (hull > 0) & ~(inner > 0)
    mask = rim.astype(np.float32)
    if scale < 1.0:
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
    sigma = max(0.7, 0.8 / scale * 0.5)
    mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    return np.clip(mask * 1.15, 0.0, 1.0)


def restore_glass_frame(rgba: np.ndarray, source_rgb: np.ndarray,
                        settings: dict | None = None) -> int:
    """Union the lens rim into an RGBA cut-out in place; returns pixels added."""
    mask = glass_frame_mask(source_rgb, settings)
    if mask is None:
        return 0
    old = rgba[..., 3].astype(np.float32) / 255.0
    add = mask > old
    if not add.any():
        return 0
    rgba[..., :3][add] = source_rgb[add]
    rgba[..., 3] = np.maximum(old, mask).__mul__(255).round().astype(np.uint8)
    return int(add.sum())
