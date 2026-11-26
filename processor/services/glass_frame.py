"""Keep a magnifier / zoom-lens frame that is printed on a catalogue photo.

Some catalogue photos carry an inset "glass": a stadium-shaped rim (white
band with dark edge lines and tick marks) around an enlarged view of the
piece. Background removal treats the white band as backdrop and cuts the
rim to pieces. This module finds the frame from its dark lines and puts the
whole rim back, with the inside left to the normal cut-out (so the chosen
background colour shows through the lens).

Two kinds of frame are recognised:

* round / oval lenses drawn with thin light-grey lines and tick marks
  (``_ring_frame``): the lines are fitted with an ellipse and the whole band
  between the inner and outer line is restored - including the parts the
  product or the clean-up broke up;
* stadium-shaped frames drawn with dark lines (``_largest_frame``).

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
    "glass_min_hull_fill": 0.6,   # hull area / bbox area (a sloped bead chain is thin)
    # round / oval lens drawn with light-grey lines
    "glass_ring_enabled": True,
    "glass_ring_min_lum": 110,    # line pixels: brighter than black beads ...
    "glass_ring_max_lum": 242,    # ... darker than the white backdrop ...
    "glass_ring_max_chroma": 14,  # ... and neutral (not gold)
    "glass_ring_min_size": 0.15,  # ellipse minor axis / shorter image side
    "glass_ring_fit_share": 0.85, # component pixels lying on the fitted ring band
    "glass_ring_coverage": 0.75,  # share of the circumference traced by lines
    "glass_ring_max_width": 0.12, # band width / radius (a filled disc is not a rim)
    "glass_ring_pad_px": 1.5,     # anti-aliased line edge kept on both sides
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
        # A frame encloses an area; a diagonal bead chain only has a thin hull
        # (it used to be taken for a frame and got a white band painted round it).
        if hull_mask.sum() < cfg["glass_min_hull_fill"] * bw * bh:
            continue

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


def _ellipse_rho(shape, ellipse):
    """Normalised elliptical radius of every pixel (1.0 = on the ellipse)."""
    (cx, cy), (d1, d2), ang = ellipse
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    t = np.deg2rad(ang)
    dx, dy = xs - cx, ys - cy
    u = dx * np.cos(t) + dy * np.sin(t)
    v = -dx * np.sin(t) + dy * np.cos(t)
    return np.sqrt((u / (d1 / 2.0)) ** 2 + (v / (d2 / 2.0)) ** 2), np.arctan2(v, u)


def _ring_frame(rgb: np.ndarray, cfg: dict):
    """Find a round/oval lens drawn with light-grey lines.

    Returns a boolean rim mask on the proxy, or None.
    """
    if not cfg.get("glass_ring_enabled", True):
        return None
    h, w = rgb.shape[:2]
    rgb_i = rgb.astype(np.int16)
    lum = rgb_i.mean(axis=2)
    chroma = rgb_i.max(axis=2) - rgb_i.min(axis=2)
    line = ((lum > cfg["glass_ring_min_lum"]) & (lum < cfg["glass_ring_max_lum"])
            & (chroma < cfg["glass_ring_max_chroma"])).astype(np.uint8)
    if line.sum() < 500:
        return None
    joined = cv2.dilate(line, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    min_size = cfg["glass_ring_min_size"] * min(h, w)
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
    for i in order[:8]:
        x, y, bw, bh, area = stats[i]
        if min(bw, bh) < min_size:
            continue
        comp = (labels == i) & (line > 0)
        ys, xs = np.nonzero(comp)
        if xs.size < 5 * max(bw, bh):
            continue
        ellipse = cv2.fitEllipse(np.stack([xs, ys], axis=1).astype(np.float32))
        (cx, cy), (d1, d2), _ = ellipse
        if min(d1, d2) < min_size:
            continue
        rho, theta = _ellipse_rho((h, w), ellipse)
        r = rho[comp]
        lo, hi = np.percentile(r, [2, 98])
        if hi - lo > cfg["glass_ring_max_width"] or float(((r >= lo) & (r <= hi)).mean()) < cfg["glass_ring_fit_share"]:
            continue
        # take in every line pixel on the band (pieces split off by the
        # chain or the pendant), then re-measure the band
        pad = cfg["glass_ring_pad_px"] / (min(d1, d2) / 2.0)
        band_lines = (line > 0) & (rho >= lo - pad) & (rho <= hi + pad)
        r = rho[band_lines]
        lo, hi = np.percentile(r, [0.5, 99.5])
        # the whole circumference must be drawn (a closed lens, not an arc of chain)
        bins = ((theta[band_lines] + np.pi) / (2 * np.pi) * 180).astype(int).clip(0, 179)
        coverage = np.unique(bins).size / 180.0
        if coverage < cfg["glass_ring_coverage"]:
            continue
        return (rho >= lo - pad) & (rho <= hi + pad)
    return None


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
    rim = _ring_frame(proxy, cfg)
    if rim is None:
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