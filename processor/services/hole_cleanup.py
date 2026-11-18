"""Matte clean-up for jewellery cutouts.

Segmentation models are good at finding *the product* but routinely make
four mistakes on jewellery photos:

1. They keep the studio backdrop visible inside hollows (rings, bangles,
   chain links, open pendants).
2. They leave a faint haze / soft drop shadow around the product.
3. They fade thin parts (chains, prongs, wires) into half-transparency.
4. They keep dust specks on the backdrop as tiny islands.

Every pass here compares each pixel with a *model of the backdrop* (a smooth
quadratic colour surface fitted on pixels far away from the product). That
follows vignetting and light fall-off, so a hollow in the middle of the
picture is compared with the backdrop that would be there, not with the colour
at the image border.

Textured regions (diamond facets, sparkle, engraved gold) are never cleared:
every removal requires the region to be *flat* as well as backdrop-coloured.

All passes are controlled by keys in the preset's ``background_removal``
section; the defaults below are used when a key is missing.
"""
from __future__ import annotations

import cv2
import numpy as np

DEFAULTS = {
    "remove_inner_holes": True,
    "remove_background_shadows": True,
    "recover_thin_details": True,
    "remove_specks": True,
    # backdrop model
    "background_far_px": 0.03,          # fraction of image diagonal
    # border-connected backdrop
    "border_background_tolerance": 32.0,
    "border_background_gradient": 18.0,
    "border_protect_alpha": 240,
    # enclosed holes
    "hole_core_tolerance": 20.0,
    "hole_grow_tolerance": 42.0,
    "hole_flat_gradient": 22.0,
    "hole_fringe_iterations": 1,
    "hole_min_area_ratio": 0.000004,     # of image area (tiny gaps between settings)
    "hole_filled_min_area_ratio": 0.002,  # model kept it solid: needs to be big
    "hole_bright_tolerance": 6.0,
    "hole_small_match_tolerance": 24.0,   # small model-filled gaps must match the backdrop this closely
    "backdrop_texture_max": 14.0,
    "gap_border_min_saturation": 0.12,    # small filled gaps are cleared only when bordered by coloured metal         # local contrast: backdrop is smooth, faceted stones are not
    # stones / settings
    "solidify_stones": True,
    "stone_max_area_ratio": 0.002,        # an enclosed region bigger than this is never a stone
    "stone_edge_px_ratio": 0.0012,        # keep the soft outer edge this wide (fraction of diagonal)
    "stone_size_ratio": 0.004,            # typical stone diameter (fraction of diagonal) for notch filling
    # contour hierarchy
    "contour_hole_min_area_px": 16,
    "contour_hole_alpha_threshold": 8,
    # shadows
    "shadow_min_ratio": 0.30,
    "shadow_max_ratio": 0.975,
    "shadow_chroma_tolerance": 0.035,
    "shadow_texture_max": 7.0,
    "shadow_max_alpha": 235,
    "shadow_max_mean_alpha": 150,
    # thin detail recovery
    "recover_color_tolerance": 48.0,
    "recover_distance_ratio": 0.006,     # of image diagonal
    # specks
    "speck_min_area_ratio": 0.00004,
    "speck_keep_distance_ratio": 0.02,
}


def _cfg(settings: dict | None) -> dict:
    cfg = dict(DEFAULTS)
    for key, value in (settings or {}).items():
        if key in DEFAULTS and value is not None:
            cfg[key] = value
    return cfg


# ----------------------------------------------------------------------
# Backdrop model
# ----------------------------------------------------------------------

def _poly_terms(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(xs), xs, ys, xs * xs, xs * ys, ys * ys], axis=-1)


def fit_background(src: np.ndarray, alpha: np.ndarray, far_px: float) -> np.ndarray:
    """Fit the backdrop surface. Returns 6x3 polynomial coefficients over
    normalised (0..1) image coordinates, so it can be evaluated at any
    resolution or on any sub-region."""
    h, w = alpha.shape
    product = (alpha > 16).astype(np.uint8)
    if product.any():
        dist = cv2.distanceTransform(1 - product, cv2.DIST_L2, 3)
        far = dist > far_px
    else:
        far = np.ones_like(product, dtype=bool)

    if far.sum() < 400:
        # Product fills the frame: fall back to a thin border band.
        band = max(3, min(h, w) // 100)
        far = np.zeros_like(far)
        far[:band, :] = far[-band:, :] = True
        far[:, :band] = far[:, -band:] = True

    ys, xs = np.nonzero(far)
    rng = np.random.default_rng(0)
    if len(xs) > 40000:
        pick = rng.choice(len(xs), 40000, replace=False)
        xs, ys = xs[pick], ys[pick]

    xn = xs.astype(np.float32) / max(w - 1, 1)
    yn = ys.astype(np.float32) / max(h - 1, 1)
    terms = _poly_terms(xn, yn)
    samples = src[ys, xs].astype(np.float32)

    coeffs = np.linalg.lstsq(terms, samples, rcond=None)[0]
    # One robust re-fit: drop samples that are clearly not backdrop
    # (a stray prop, a hand, a label) before the final fit.
    resid = np.abs(samples - terms @ coeffs).max(axis=1)
    keep = resid <= max(6.0, 3.0 * float(np.median(resid)) + 1.0)
    if keep.sum() > 50:
        coeffs = np.linalg.lstsq(terms[keep], samples[keep], rcond=None)[0]
    return coeffs


def eval_background(coeffs: np.ndarray, out_h: int, out_w: int, region=(0.0, 0.0, 1.0, 1.0)) -> np.ndarray:
    """Evaluate the backdrop on an out_h x out_w grid covering ``region``
    (x0, y0, x1, y1 in normalised image coordinates)."""
    x0, y0, x1, y1 = region
    gx = np.linspace(x0, x1, out_w, dtype=np.float32)[None, :]
    gy = np.linspace(y0, y1, out_h, dtype=np.float32)[:, None]
    model = np.empty((out_h, out_w, 3), np.float32)
    for c in range(3):
        a0, a1, a2, a3, a4, a5 = (np.float32(v) for v in coeffs[:, c])
        model[..., c] = a0 + a1 * gx + a2 * gy + a3 * gx * gx + a4 * gx * gy + a5 * gy * gy
    return np.clip(model, 0, 255, out=model)


def background_model(src: np.ndarray, alpha: np.ndarray, far_px: float) -> np.ndarray:
    """Per-pixel backdrop colour for the whole image (fit + evaluate)."""
    h, w = alpha.shape
    return eval_background(fit_background(src, alpha, far_px), h, w)


def _gradient(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy) / 4.0


def _local_std(gray: np.ndarray, k: int = 5) -> np.ndarray:
    mean = cv2.blur(gray, (k, k))
    sq = cv2.blur(gray * gray, (k, k))
    return np.sqrt(np.maximum(sq - mean * mean, 0))


class MatteContext:
    """Shared per-image measurements used by every pass."""

    def __init__(self, source_rgb: np.ndarray, alpha: np.ndarray, cfg: dict,
                 background: np.ndarray | None = None, reference_size: tuple | None = None):
        self.src = source_rgb.astype(np.float32)
        h, w = alpha.shape
        # Size-relative thresholds refer to the whole photo even when only a
        # crop around the product is processed.
        ref_h, ref_w = reference_size or (h, w)
        self.diag = float(np.hypot(ref_h, ref_w))
        self.area = float(ref_h * ref_w)
        self.cfg = cfg
        if background is not None:
            self.bg = background.astype(np.float32, copy=False)
        else:
            far_px = float(cfg["background_far_px"]) * float(np.hypot(h, w))
            self.bg = background_model(self.src, alpha, far_px)
        diff = cv2.subtract(self.src, self.bg)
        # per-pixel colour distance; cv2.transform sums channels multi-threaded
        self.color_dist = cv2.sqrt(cv2.transform(cv2.multiply(diff, diff), np.ones((1, 3), np.float32)))
        self.gray = cv2.cvtColor(self.src, cv2.COLOR_RGB2GRAY)
        self.grad = _gradient(self.gray)
        # signed brightness vs backdrop (white facets / highlights are > 0)
        self.lum_diff = cv2.transform(cv2.subtract(self.src, self.bg), np.full((1, 3), 1 / 3.0, np.float32))
        self._backdrop_like = None
        self._chroma_diff = None

    @property
    def backdrop_like(self) -> np.ndarray:
        """Pixels that look like plain backdrop: backdrop colour, flat, and
        not brighter than the backdrop (so diamond facets never qualify)."""
        if self._backdrop_like is None:
            cfg = self.cfg
            self._backdrop_like = (
                (self.color_dist < float(cfg["hole_grow_tolerance"]))
                & (self.texture < float(cfg["backdrop_texture_max"]))
                & (self.lum_diff <= float(cfg["hole_bright_tolerance"]))
            )
        return self._backdrop_like

    @property
    def chroma_diff(self) -> np.ndarray:
        """Hue difference vs the backdrop (a shadow is the same hue, only darker)."""
        if self._chroma_diff is None:
            ones = np.ones((1, 3), np.float32)
            s_sum = cv2.add(cv2.transform(self.src, ones), 1.0)
            b_sum = cv2.add(cv2.transform(self.bg, ones), 1.0)
            s_n = cv2.divide(self.src, cv2.merge([s_sum] * 3))
            b_n = cv2.divide(self.bg, cv2.merge([b_sum] * 3))
            diff = cv2.absdiff(s_n, b_n)
            c0, c1, c2 = cv2.split(diff)
            self._chroma_diff = cv2.max(cv2.max(c0, c1), c2)
        return self._chroma_diff

    @property
    def texture(self) -> np.ndarray:
        """Local contrast (grey-level std) over a window that scales with the
        photo, so a 6000 px original and its 2000 px copy measure the same
        facets. Diamonds reflect the backdrop colour, but they are never
        smooth - this is what tells a stone from a gap between settings."""
        if getattr(self, "_texture", None) is None:
            k = max(5, int(round(self.diag * 0.0016)) | 1)
            self._texture = _local_std(self.gray, k)
        return self._texture


# ----------------------------------------------------------------------
# Passes
# ----------------------------------------------------------------------

def remove_border_background(ctx: MatteContext, alpha: np.ndarray) -> int:
    """Clear flat backdrop regions that stay connected to the image border."""
    cfg = ctx.cfg
    bg_like = (
        (ctx.color_dist < float(cfg["border_background_tolerance"]))
        & (ctx.grad < float(cfg["border_background_gradient"]))
    ).astype(np.uint8)
    count, labels = cv2.connectedComponents(bg_like, connectivity=4)
    if count <= 1:
        return 0
    border_labels = np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
    ]))
    border_labels = border_labels[border_labels > 0]
    region = np.isin(labels, border_labels)
    clear = region & (alpha > 0) & (alpha < int(cfg["border_protect_alpha"]))
    alpha[clear] = 0
    return int(clear.sum())


def remove_enclosed_background(ctx: MatteContext, alpha: np.ndarray) -> int:
    """Make backdrop-coloured hollows inside the product transparent."""
    cfg = ctx.cfg
    if not cfg["remove_inner_holes"]:
        return 0
    core = (
        (ctx.color_dist < float(cfg["hole_core_tolerance"]) * 1.25)
        & (ctx.grad < float(cfg["hole_flat_gradient"]))
        & (ctx.texture < float(cfg["backdrop_texture_max"]))
        & (alpha > 0)
    ).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=4)
    if count <= 1:
        return 0

    min_area = max(4, int(float(cfg["hole_min_area_ratio"]) * ctx.area))
    filled_min_area = max(min_area, int(float(cfg["hole_filled_min_area_ratio"]) * ctx.area))
    strict_tol = float(cfg["hole_core_tolerance"]) * 0.6

    areas = stats[:, cv2.CC_STAT_AREA]
    candidate_ids = np.nonzero(areas >= min_area)[0]
    candidate_ids = candidate_ids[candidate_ids > 0]
    if candidate_ids.size == 0:
        return 0

    # Mean alpha / colour distance per component in one vectorised pass.
    flat_labels = labels.ravel()
    alpha_sum = np.bincount(flat_labels, weights=alpha.ravel().astype(np.float64), minlength=count)
    dist_sum = np.bincount(flat_labels, weights=ctx.color_dist.ravel().astype(np.float64), minlength=count)
    mean_alpha = alpha_sum / np.maximum(areas, 1)
    mean_dist = dist_sum / np.maximum(areas, 1)
    # Signed brightness vs the backdrop: real hollows show the backdrop at
    # backdrop brightness (or a little darker, from the hollow's own shadow);
    # white diamond facets and polished highlights are brighter than it.
    lum_sum = np.bincount(flat_labels, weights=ctx.lum_diff.ravel().astype(np.float64), minlength=count)
    mean_lum = lum_sum / np.maximum(areas, 1)
    tex_sum = np.bincount(flat_labels, weights=ctx.texture.ravel().astype(np.float64), minlength=count)
    mean_tex = tex_sum / np.maximum(areas, 1)
    tex_max = float(cfg["backdrop_texture_max"])
    bright_tol = float(cfg.get("hole_bright_tolerance", 6.0))

    chosen = []
    for idx in candidate_ids:
        brighter_than_backdrop = mean_lum[idx] > bright_tol
        if brighter_than_backdrop and mean_alpha[idx] > 24:
            continue  # facet / highlight, not a hollow
        if mean_tex[idx] >= tex_max and mean_alpha[idx] > 24:
            continue  # textured: part of a stone reflecting the backdrop
        model_says_hole = mean_alpha[idx] < 96
        # The model kept it solid. Only override for a big, very flat region
        # that matches the backdrop closely; a diamond table is small and
        # brighter than the backdrop, so it does not qualify.
        clearly_backdrop = areas[idx] >= filled_min_area and mean_dist[idx] < strict_tol
        # Small gaps between petals / bezels that the model filled in or left
        # hazy: cleared only when they match the backdrop very closely.
        small_gap = (
            mean_dist[idx] < float(cfg["hole_small_match_tolerance"])
            and _bordered_by_coloured_metal(ctx, labels, stats, idx, alpha, cfg)
        )
        if model_says_hole or clearly_backdrop or small_gap:
            chosen.append(idx)
    if not chosen:
        return 0

    hole = np.isin(labels, chosen).astype(np.uint8)
    # Grow into the soft rim of the hollow (anti-aliased backdrop pixels).
    # The core keeps half a texture-window away from edges; grow back by that
    # much (plus the configured fringe), limited to backdrop-coloured pixels
    # that are not brighter than the backdrop.
    grow_ok = (
        (ctx.color_dist < float(cfg["hole_grow_tolerance"]))
        & (ctx.lum_diff <= float(cfg["hole_bright_tolerance"]))
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    window_half = max(5, int(round(ctx.diag * 0.0016)) | 1) // 2
    for _ in range(int(cfg["hole_fringe_iterations"]) + window_half):
        grown = cv2.dilate(hole, kernel)
        hole = np.where(grow_ok, grown, hole).astype(np.uint8)

    hole_mask = hole.astype(bool)
    before = int((alpha[hole_mask] > 0).sum())
    alpha[hole_mask] = 0
    return before


def _bordered_by_coloured_metal(ctx, labels, stats, idx, alpha, cfg) -> bool:
    """True when the region's rim is mostly coloured metal (yellow/rose gold).

    A gap between settings is framed by metal; a flat diamond table is framed
    by neutral facets. White-gold pieces return False, so their gaps are left
    to the model (a safe failure: a little haze rather than a hole in a stone).
    """
    x, y, w, h = stats[idx, :4]
    # reach past the strip the texture window keeps between core and edge
    reach = max(5, int(round(ctx.diag * 0.0016)) | 1) + 4
    pad = reach + 1
    y0, y1 = max(0, y - pad), min(labels.shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(labels.shape[1], x + w + pad)
    comp = (labels[y0:y1, x0:x1] == idx).astype(np.uint8)
    ring = cv2.dilate(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))).astype(bool)
    ring &= ~comp.astype(bool)
    ring &= alpha[y0:y1, x0:x1] >= 128
    ring &= ~ctx.backdrop_like[y0:y1, x0:x1]
    if ring.sum() < 4:
        return False
    rgb = ctx.src[y0:y1, x0:x1][ring]
    mx, mn = rgb.max(axis=1), rgb.min(axis=1)
    saturation = (mx - mn) / np.maximum(mx, 1.0)
    return float(np.median(saturation)) >= float(cfg["gap_border_min_saturation"])


def remove_hierarchy_holes(ctx: MatteContext, alpha: np.ndarray) -> int:
    """Clear faint residual alpha inside closed hollows of the matte.

    ``RETR_CCOMP`` gives outer contours and the holes directly inside them.
    Any faint alpha left inside a hole (but not on a nested product island)
    is backdrop residue and is set to zero.
    """
    cfg = ctx.cfg
    thr = int(cfg["contour_hole_alpha_threshold"])
    min_area = float(cfg["contour_hole_min_area_px"])
    binary = (alpha > thr).astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    holes = np.zeros_like(binary)
    for i, info in enumerate(hierarchy[0]):
        parent = info[3]
        if parent >= 0 and cv2.contourArea(contours[i]) >= min_area:
            cv2.drawContours(holes, contours, i, 1, thickness=-1)
    clear = holes.astype(bool) & (alpha > 0) & (alpha <= thr)
    alpha[clear] = 0
    # Global floor: alpha this faint is never a visible part of the product.
    floor = (alpha > 0) & (alpha <= max(2, thr // 2))
    alpha[floor] = 0
    return int(clear.sum() + floor.sum())


def shadow_mask(ctx: MatteContext, alpha: np.ndarray) -> np.ndarray:
    """Soft cast shadows: darker than the backdrop, same hue, untextured.

    Evaluated only where it matters - partly transparent pixels (the shadow
    pass) and pixels next to the product (so detail recovery never pulls a
    shadow back in) - which keeps it fast on large crops.
    """
    cfg = ctx.cfg
    out = np.zeros(alpha.shape, bool)
    body = (alpha >= 128).astype(np.uint8)
    reach = max(1, int(round(float(cfg["recover_distance_ratio"]) * ctx.diag)) + 1)
    near_body = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))).astype(bool)
    cand = (alpha < int(cfg["shadow_max_alpha"])) & ((alpha > 0) | near_body)
    ys, xs = np.nonzero(cand)
    if ys.size == 0:
        return out
    eps = 1.0
    src = ctx.src[ys, xs]
    bg = ctx.bg[ys, xs]
    lum_src = src.sum(axis=1) + eps
    lum_bg = bg.sum(axis=1) + eps
    ratio = lum_src / lum_bg
    chroma = np.abs(src / lum_src[:, None] - bg / lum_bg[:, None]).max(axis=1)
    texture = _local_std(ctx.gray, 5)[ys, xs]
    hit = (
        (ratio > float(cfg["shadow_min_ratio"]))
        & (ratio < float(cfg["shadow_max_ratio"]))
        & (chroma < float(cfg["shadow_chroma_tolerance"]))
        & (texture < float(cfg["shadow_texture_max"]))
    )
    out[ys[hit], xs[hit]] = True
    return out


def remove_shadows(ctx: MatteContext, alpha: np.ndarray, shadows: np.ndarray) -> int:
    """Clear shadow regions the model was unsure about.

    Grey diamond facets and white-gold surfaces also look like "darker,
    neutral, smooth" pixels, so a region is only treated as shadow when the
    model gave it a low average alpha *and* it borders the backdrop.
    """
    if not ctx.cfg["remove_background_shadows"]:
        return 0
    candidate = (shadows & (alpha > 0)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    if count <= 1:
        return 0
    areas = stats[:, cv2.CC_STAT_AREA]
    flat = labels.ravel()
    mean_alpha = np.bincount(flat, weights=alpha.ravel().astype(np.float64), minlength=count) / np.maximum(areas, 1)

    backdrop = (alpha < 30).astype(np.uint8)
    touching = cv2.dilate(backdrop, np.ones((3, 3), np.uint8)).astype(bool) & candidate.astype(bool)
    touches_backdrop = np.bincount(flat[touching.ravel()], minlength=count) > 0

    max_mean = float(ctx.cfg.get("shadow_max_mean_alpha", 150))
    chosen = [i for i in range(1, count) if mean_alpha[i] < max_mean and touches_backdrop[i]]
    if not chosen:
        return 0
    clear = np.isin(labels, chosen)
    alpha[clear] = 0
    return int(clear.sum())


def recover_thin_details(ctx: MatteContext, alpha: np.ndarray, shadows: np.ndarray) -> int:
    """Re-opacify chain links / prongs the model faded.

    Only pixels that are clearly *not* backdrop-coloured, not shadow, and
    close to the confident product body are raised. White-on-white parts are
    left to the sparkle-recovery logic in image_processor.
    """
    cfg = ctx.cfg
    if not cfg["recover_thin_details"]:
        return 0
    body = (alpha >= 128).astype(np.uint8)
    if not body.any():
        return 0
    max_dist = float(cfg["recover_distance_ratio"]) * ctx.diag
    dist = cv2.distanceTransform(1 - body, cv2.DIST_L2, 3)
    tol = float(cfg["recover_color_tolerance"])
    strength = np.clip((ctx.color_dist - tol * 0.5) / tol, 0, 1)
    target = (strength * 255).astype(np.uint8)
    eligible = (dist <= max_dist) & (~shadows) & (alpha < 200) & (target > alpha) & (ctx.color_dist > tol * 0.75)
    alpha[eligible] = np.maximum(alpha[eligible], target[eligible])
    return int(eligible.sum())


def remove_specks(ctx: MatteContext, alpha: np.ndarray) -> int:
    """Drop tiny islands far from the product (dust, sensor spots)."""
    cfg = ctx.cfg
    if not cfg["remove_specks"]:
        return 0
    solid = (alpha > 40).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    if count <= 2:
        return 0
    min_area = max(6, int(float(cfg["speck_min_area_ratio"]) * ctx.area))
    keep_dist = float(cfg["speck_keep_distance_ratio"]) * ctx.diag

    # Group islands into clusters (single linkage within keep_dist): every
    # link of a chain is its own island, but the chain is one cluster.
    cluster_of, cluster_area = cluster_components(solid, labels, count, keep_dist)
    areas = stats[:, cv2.CC_STAT_AREA]
    largest_cluster = max(cluster_area.values())

    removed = 0
    for idx in range(1, count):
        cluster = cluster_of[idx]
        if cluster_area[cluster] >= max(min_area, 0.02 * largest_cluster):
            continue  # part of the product or a real separate piece
        x, y, w, h = stats[idx, :4]
        comp = labels[y:y + h, x:x + w] == idx
        alpha[y:y + h, x:x + w][comp] = 0
        removed += int(areas[idx])
    return removed


def cluster_components(mask: np.ndarray, labels: np.ndarray, count: int, link_dist: float):
    """Map each connected component to a cluster of nearby components.

    Returns ({component_id: cluster_id}, {cluster_id: total_area}).
    """
    radius = max(1, int(round(link_dist / 2)))
    h, w = mask.shape
    # Dilate on a reduced copy for speed on large photos.
    scale = min(1.0, 1500.0 / max(h, w))
    small = mask
    if scale < 1:
        small = cv2.resize(mask, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        small = (small > 0).astype(np.uint8)
    r = max(1, int(round(radius * scale)))
    grown = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    _, cluster_labels = cv2.connectedComponents(grown, connectivity=8)
    if scale < 1:
        cluster_labels = cv2.resize(cluster_labels.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)

    flat = labels.ravel()
    clusters = cluster_labels.ravel()
    on = flat > 0
    # component -> cluster id (take the cluster under any of its pixels)
    cluster_of = np.zeros(count, dtype=np.int64)
    cluster_of[flat[on]] = clusters[on]
    comp_areas = np.bincount(flat, minlength=count)
    cluster_area: dict[int, int] = {}
    for idx in range(1, count):
        cluster_area[int(cluster_of[idx])] = cluster_area.get(int(cluster_of[idx]), 0) + int(comp_areas[idx])
    return {idx: int(cluster_of[idx]) for idx in range(1, count)}, cluster_area


def solidify_stones(ctx: MatteContext, alpha: np.ndarray) -> int:
    """Make stones and the inside of settings fully opaque.

    Segmentation models often give diamonds partial alpha (they are bright,
    see-through-looking and similar to a light backdrop), or punch small holes
    into them. A semi-transparent diamond loses its shine: on a white page it
    looks washed out, on a dark one grey and dull.

    * enclosed regions of the matte (inside a bezel/setting) that do not look
      like backdrop and are stone-sized become opaque;
    * semi-transparent pixels away from the outer edge that do not look like
      backdrop become opaque.
    The soft anti-aliased outline is left untouched.
    """
    cfg = ctx.cfg
    if not cfg["solidify_stones"]:
        return 0
    h, w = alpha.shape
    backdrop = ctx.backdrop_like
    changed = 0

    # enclosed regions (holes in the solid matte)
    body = (alpha >= 128).astype(np.uint8)
    if body.any():
        # outside = non-product regions connected to the crop border
        count0, lab0 = cv2.connectedComponents(1 - body, connectivity=4)
        edge_labels = np.unique(np.concatenate([lab0[0, :], lab0[-1, :], lab0[:, 0], lab0[:, -1]]))
        edge_labels = edge_labels[edge_labels > 0]
        outside = np.isin(lab0, edge_labels)
        enclosed = (~body.astype(bool)) & (~outside)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(enclosed.astype(np.uint8), connectivity=4)
        if count > 1:
            max_area = float(cfg["stone_max_area_ratio"]) * ctx.area
            flat = labels.ravel()
            areas = stats[:, cv2.CC_STAT_AREA]
            bd = np.bincount(flat, weights=backdrop.ravel().astype(np.float64), minlength=count) / np.maximum(areas, 1)
            fill_ids = [i for i in range(1, count) if areas[i] <= max_area and bd[i] < 0.5]
            if fill_ids:
                faceted = ctx.texture >= float(cfg["backdrop_texture_max"]) * 1.3
                metal = (ctx.color_dist >= float(cfg["hole_grow_tolerance"])) & (
                    (ctx.chroma_diff >= 0.04) | (ctx.lum_diff < -45)
                    | (ctx.lum_diff > float(cfg["hole_bright_tolerance"]) * 2))
                fill = np.isin(labels, fill_ids) & ~backdrop & (faceted | metal) & (alpha < 255)
                alpha[fill] = 255
                changed += int(fill.sum())

    # stone-sized notches: a stone the model partly dropped at the edge of a
    # setting is not enclosed, but it sits inside the closed outline. Only
    # textured (faceted) pixels are filled; smooth backdrop stays clear.
    r = max(2, int(round(float(cfg["stone_size_ratio"]) * ctx.diag / 2)))
    closed = cv2.morphologyEx(
        (alpha >= 128).astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)),
    ).astype(bool)
    tex_max = float(cfg["backdrop_texture_max"])
    # A backdrop-coloured pixel right outside the outline also measures as
    # "textured" (the window straddles the edge). Such a pixel is only filled
    # when it is mostly surrounded by product - true for a pinhole inside a
    # stone, false for the backdrop along the outline (that would be a halo).
    body_frac = cv2.blur(closed.astype(np.float32), (2 * r + 1, 2 * r + 1))
    backdrop_coloured = ctx.color_dist < float(cfg["hole_small_match_tolerance"])
    surrounded = body_frac >= 0.65
    well_inside = body_frac >= 0.85
    # clearly faceted, or moderately textured and well inside the setting;
    # shadowed backdrop between settings has only moderate texture and sits
    # in an open gap, so it is not raised
    # a stone always has a sparkle highlight close by; shadowed backdrop does not
    rr = max(3, int(round(0.0012 * ctx.diag)))
    sparkle_near = cv2.dilate(
        ctx.lum_diff, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))
    ) > float(cfg.get("stone_sparkle_min", 15.0))
    stone_like = sparkle_near & ((ctx.texture >= tex_max * 1.3) | ((ctx.texture >= tex_max) & well_inside))
    notch = closed & (alpha < 255) & stone_like & ~backdrop & (~backdrop_coloured | surrounded)
    alpha[notch] = 255
    changed += int(notch.sum())

    # semi-transparent interior pixels
    edge = max(1, int(round(float(cfg["stone_edge_px_ratio"]) * ctx.diag)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge + 1, 2 * edge + 1))
    interior = cv2.erode((alpha >= 64).astype(np.uint8), kernel).astype(bool)
    # clearly not backdrop: different colour AND different hue (shadow is not metal)
    # (shadows are only mildly darker; dark stone facets / metal are much darker)
    bright = float(cfg["hole_bright_tolerance"]) * 2
    metal_like = (ctx.color_dist >= float(cfg["hole_grow_tolerance"])) & (
        (ctx.chroma_diff >= 0.04) | (ctx.lum_diff < -45) | (ctx.lum_diff > bright))
    raise_px = interior & (alpha < 255) & ~backdrop & (stone_like | metal_like) & (~backdrop_coloured | surrounded)
    alpha[raise_px] = 255
    changed += int(raise_px.sum())
    return changed


def clean_matte(source_rgb: np.ndarray, alpha: np.ndarray, settings: dict | None = None,
                background: np.ndarray | None = None, reference_size: tuple | None = None):
    """Run every clean-up pass. Returns (alpha, stats, context).

    ``background`` (precomputed backdrop colour for this region) and
    ``reference_size`` (h, w of the full photo) let callers run the passes on
    a crop around the product only, which is much faster on large photos.
    """
    cfg = _cfg(settings)
    alpha = alpha.copy()
    ctx = MatteContext(source_rgb, alpha, cfg, background, reference_size)
    shadows = shadow_mask(ctx, alpha)
    stats = {
        "border_background_px": remove_border_background(ctx, alpha),
        "shadow_px": remove_shadows(ctx, alpha, shadows),
        "recovered_detail_px": recover_thin_details(ctx, alpha, shadows),
        "hole_px": remove_enclosed_background(ctx, alpha),
        "contour_hole_px": remove_hierarchy_holes(ctx, alpha),
        "speck_px": remove_specks(ctx, alpha),
    }
    stats["solid_stone_px"] = solidify_stones(ctx, alpha)
    return alpha, stats, ctx
