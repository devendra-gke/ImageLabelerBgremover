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
    "gap_border_min_saturation": 0.12,
    "exact_backdrop_tolerance": 9.0,      # region matches local backdrop this closely ...
    "exact_backdrop_flatness": 5.0,       # ... and is this flat -> always a hollow
    "local_backdrop_radius": 0.02,        # fraction of diagonal used to sample nearby backdrop
    "full_res_cleanup_max_px": 12_000_000,
    # plain-backdrop openings in any frame (white gold / pave included)
    "open_backdrop_hollows": True,
    "protect_stone_reflections": True,
    "reflection_min_dist": 8.0,           # cut region differs from local backdrop by more than this (median) ...
    "reflection_max_lum": -2.5,           # ... and is darker than it -> a stone facet, restored
    "reflection_model_alpha": 140,        # only where the model was confident it is product
    "hollow_seed_tolerance": 11.0,        # seed pixels match the local backdrop this closely
    "hollow_grow_tolerance": 24.0,
    "hollow_max_mean_dist": 12.0,  # bigger crops: decisions on a reduced copy (much faster)
    # edge refinement (guided filter at native resolution)
    "refine_edges": True,
    "edge_radius_ratio": 0.0005,          # guided-filter radius, fraction of diagonal (~4 px at 6720 px)
    "edge_eps": 0.001,
    "edge_contrast": 2.0,                 # >1 tightens the soft transition (crisper edge)
    # small pockets inside coloured-metal chains (link holes)
    "open_metal_pockets": True,
    "pocket_max_area_ratio": 0.00003,
    "pocket_dark_guard": 40.0,            # a pocket with pixels this much darker than backdrop is a stone    # small filled gaps are cleared only when bordered by coloured metal         # local contrast: backdrop is smooth, faceted stones are not
    # stones / settings
    "solidify_stones": True,
    "stone_max_area_ratio": 0.002,        # an enclosed region bigger than this is never a stone
    "stone_edge_px_ratio": 0.0012,        # keep the soft outer edge this wide (fraction of diagonal)
    "stone_size_ratio": 0.004,            # typical stone diameter (fraction of diagonal) for notch filling
    "flat_region_min_stone_areas": 0.35,  # open backdrop patch: min area (in stone areas)
    "open_backdrop_max_fine": 3.0,        # a patch this flat and ...
    "open_backdrop_max_dist": 26.0,        # ... this close to the exterior backdrop is not a stone reflection
    "clear_wedges": True,                 # openings between spokes/wires that are pure backdrop colour
    "wedge_dist": 24.0,
    "wedge_chroma": 0.03,
    "wedge_open": 5,
    "wedge_min_px": 40,
    "wedge_max_px": 9000,
    "wedge_core_dist": 9.0,
    "wedge_max_std": 4.0,
    "wedge_sparkle": 22.0,
    "wedge_metal_frac": 0.12,
    "wedge_bare_min_px": 100,
    "wedge_bare_dist": 6.5,
    "wedge_bare_std": 3.4,
    "finish_edges": True,                 # final: smooth contour, defringe, clear tiny holes
    "smooth_edges": True,
    "smooth_sigma_ratio": 0.00018,
    "smooth_gain": 2.4,
    "finish_band_px": 2,
    "defringe": True,
    "defringe_d0": 9.0,
    "defringe_d1": 30.0,
    "defringe_chroma": 0.04,
    "defringe_max_brighter": 4.0,
    "defringe_max_alpha": 0.97,
    "tiny_hole_model_max": 0.88,
    "tiny_holes": True,
    "tiny_hole_dist": 26.0,
    "tiny_hole_chroma": 0.035,
    "tiny_hole_max_px": 400,
    "tiny_hole_metal_frac": 0.85,
    "clear_flat_backdrop": True,          # final: solid pale backdrop blobs left opaque become clear
    "flat_backdrop_dist": 60.0,
    "flat_backdrop_chroma": 0.04,
    "flat_backdrop_open": 7,
    "flat_backdrop_min_area": 120,
    "flat_backdrop_ref_px": 3000,
    "clear_backdrop_haze": True,          # final: semi-transparent backdrop-coloured pixels become clear
    "haze_color_tolerance": 14.0,
    "haze_chroma": 0.03,
    "haze_max_brighter": 5.0,
    "remove_thin_gaps": True,             # clear narrow backdrop strips between metal strands
    "thin_gap_tolerance": 38.0,           # max colour distance from the backdrop (strips are tinted by the metal)
    "thin_gap_chroma": 0.03,              # max hue difference
    "thin_gap_max_darker": 30.0,          # max brightness below the backdrop
    "thin_gap_max_brighter": 22.0,        # max brightness above the backdrop (spill from polished wire)
    "thin_gap_half_width_ratio": 0.0011,  # widest strip, half-width as fraction of diagonal
    "thin_gap_elongation": 3.0,           # strips are long, stones are compact
    "band_color_tolerance": 16.0,         # flat backdrop patch: max colour distance from the local backdrop
    "band_flat_texture": 3.5,             # ... and max 5x5 local contrast
    "band_chroma_tolerance": 0.025,       # ... and max hue difference
    "band_min_area_ratio": 0.12,          # smallest patch (fraction of one stone's area) that counts as backdrop
    "band_reach_ratio": 0.0012,           # rim around such a patch that is also kept clear (fraction of diagonal)
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
    # white (clipped) backdrop guard - see _ClippedBackdropGuard
    "protect_on_white_backdrop": True,
    "clipped_backdrop_lum": 245.0,        # backdrop this bright = blown-out white studio backdrop
    "white_guard_model_alpha": 220,       # model was this sure it is product -> clean-up may not cut it
    "white_guard_exact_dist": 6.0,        # ... unless the cut region is pure backdrop colour
    "white_guard_exact_texture": 2.5,     # ... and perfectly flat (a real see-through gap)
    "white_guard_facet_dark": 35.0,       # neutral pixels this much darker than backdrop = diamond facet lines
    "white_guard_facet_texture": 8.0,
    "white_guard_stone_ratio": 0.004,     # stone-zone growth, fraction of diagonal
    # light grey / off-white studio backdrops: diamonds are the same grey as
    # the backdrop, so stones get holes punched in them the same way
    "protect_on_light_backdrop": True,
    "light_backdrop_lum": 170.0,
    "light_backdrop_max_chroma": 0.12,
    "light_guard_exact_dist": 10.0,       # real see-through gap on a noisy grey backdrop
    "light_guard_exact_texture": 3.5,
    "light_guard_exact_share": 0.5,       # (white backdrops) share of backdrop-like pixels -> real opening
    # light grey backdrops: a real opening is backdrop-coloured almost everywhere;
    # a dull diamond facet only partly. Colour distance does not depend on the
    # photo's resolution (texture windows do), so this works from 2 to 30 MP.
    "light_opening_near_dist": 10.0,
    "light_opening_near_share": 0.65,
    "light_backdrop_like_lum": 30.0,      # backdrop-like: brightness within this of the backdrop ...
    "light_backdrop_like_sat": 0.10,      # ... neutral ...
    "light_backdrop_like_texture": 6.0,   # ... and flat in a 3 px window (diamonds are not)
    "fill_stone_holes": True,             # close holes inside diamonds (model or clean-up made)
    "stone_hole_max_ratio": 0.0004,       # largest hole closed, share of the image
    # white backdrop showing through tiny gaps (between chain links and beads)
    "clear_white_pockets": True,
    "white_pocket_min_lum": 245.0,        # all channels at least this bright
    "white_pocket_max_px": 150,           # always cleared up to this size (at 2 MP; grows with the photo) ...
    "white_pocket_ref_px": 2_000_000,
    "white_pocket_big_ratio": 0.0002,     # ... up to this share of the image when not solid
    "white_pocket_big_max_alpha": 230,    # (mean alpha below this: the model was unsure)
    "white_pocket_max_facet_share": 0.3,  # rim mostly grey facets -> diamond sparkle, keep
    "backdrop_pocket_darker": 15.0,       # light backdrop: pocket pixels at most this darker than the backdrop
    "backdrop_pocket_brighter": 40.0,     # ... and at most this brighter (lit by the metal around)
    "backdrop_pocket_sat": 0.15,          # ... nearly neutral
    "backdrop_pocket_min_ratio": 4e-6,
    # an opening that matches the backdrop almost exactly over a fair area
    # is backdrop even when diamonds surround it (openings in a stone ring)
    "backdrop_pocket_exact_lum": 8.0,
    "backdrop_pocket_exact_sat": 0.05,
    "backdrop_pocket_exact_min_px": 40,
    "backdrop_pocket_exact_flat_share": 0.15,  # share of perfectly flat pixels (dull diamond facets have ~0.08)
    "backdrop_pocket_exact_max_ratio": 0.001,    # smaller pale spots on metal are highlights, not holes
    "backdrop_pocket_texture": 22.0,      # ... and not facet-textured (3 px window: pockets are thin)
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
        self.bg = self._local_backdrop(alpha, self.bg)
        diff = cv2.subtract(self.src, self.bg)
        # per-pixel colour distance; cv2.transform sums channels multi-threaded
        self.color_dist = cv2.sqrt(cv2.transform(cv2.multiply(diff, diff), np.ones((1, 3), np.float32)))
        self.gray = cv2.cvtColor(self.src, cv2.COLOR_RGB2GRAY)
        self.grad = _gradient(self.gray)
        # signed brightness vs backdrop (white facets / highlights are > 0)
        self.lum_diff = cv2.transform(cv2.subtract(self.src, self.bg), np.full((1, 3), 1 / 3.0, np.float32))
        self._backdrop_like = None
        self._chroma_diff = None

    def _local_backdrop(self, alpha: np.ndarray, smooth_bg: np.ndarray) -> np.ndarray:
        """Blend the smooth backdrop surface with the backdrop actually seen
        nearby. Studio lighting varies (reflector spill, light fall-off near
        the product), so a hollow is compared with the backdrop around *it*,
        not with a whole-photo average."""
        h, w = alpha.shape
        clear = (alpha <= 2).astype(np.uint8)
        clear = cv2.erode(clear, np.ones((5, 5), np.uint8))       # keep off the product edge
        if clear.sum() < 50:
            return smooth_bg
        scale = min(1.0, 768.0 / max(h, w))
        sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
        src_s = cv2.resize(self.src, (sw, sh), interpolation=cv2.INTER_AREA)
        m_s = cv2.resize(clear.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
        m_s = (m_s > 0.99).astype(np.float32)
        radius = max(3, int(round(float(self.cfg["local_backdrop_radius"]) * self.diag * scale)))
        k = (2 * radius + 1, 2 * radius + 1)
        weight = cv2.blur(m_s, k)
        acc = cv2.blur(src_s * m_s[..., None], k)
        local = acc / np.maximum(weight, 1e-6)[..., None]
        trust = np.clip(weight / 0.05, 0, 1)[..., None]
        smooth_s = cv2.resize(smooth_bg, (sw, sh), interpolation=cv2.INTER_AREA)
        blended = local * trust + smooth_s * (1 - trust)
        return cv2.resize(blended, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    @property
    def fine_texture(self) -> np.ndarray:
        """Local contrast over a small fixed 5x5 window (for small hollows such
        as chain-link holes, which are narrower than the scaled window)."""
        if getattr(self, "_fine_texture", None) is None:
            self._fine_texture = _local_std(self.gray, 5)
        return self._fine_texture

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
        & (ctx.fine_texture < float(cfg["backdrop_texture_max"]))
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
    fine_sum = np.bincount(flat_labels, weights=ctx.fine_texture.ravel().astype(np.float64), minlength=count)
    mean_fine = fine_sum / np.maximum(areas, 1)
    tex_max = float(cfg["backdrop_texture_max"])
    exact_tol = float(cfg["exact_backdrop_tolerance"])
    exact_flat = float(cfg["exact_backdrop_flatness"])
    bright_tol = float(cfg.get("hole_bright_tolerance", 6.0))

    chosen = []
    for idx in candidate_ids:
        # Flat and indistinguishable from the nearby backdrop: a hollow,
        # whatever frames it (white-gold pave included). Stones and polished
        # metal are never this flat *and* this close to the backdrop colour.
        if mean_dist[idx] < exact_tol and mean_fine[idx] < exact_flat and abs(mean_lum[idx]) < exact_tol:
            chosen.append(idx)
            continue
        brighter_than_backdrop = mean_lum[idx] > bright_tol
        if brighter_than_backdrop and mean_alpha[idx] > 24:
            continue  # facet / highlight, not a hollow
        if min(mean_tex[idx], mean_fine[idx] * 1.5) >= tex_max and mean_alpha[idx] > 24:
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


def _box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.blur(x, (2 * r + 1, 2 * r + 1))


def refine_edges(ctx: MatteContext, alpha: np.ndarray) -> int:
    """Snap the matte to the real edges of the photo (see refine_edges_core)."""
    if not ctx.cfg["refine_edges"]:
        return 0
    return refine_edges_core(ctx.src, alpha, ctx.bg, ctx.cfg, ctx.diag)


def _guided_strip(src, bg, alpha, r, sub, cfg) -> np.ndarray:
    S = src.astype(np.float32)
    B = bg.astype(np.float32, copy=False)
    diff = cv2.subtract(S, B)
    ones = np.ones((1, 3), np.float32)
    d = cv2.sqrt(cv2.transform(cv2.multiply(diff, diff), ones))
    lum = cv2.transform(diff, np.full((1, 3), 1 / 3.0, np.float32))
    del diff
    s_sum = cv2.add(cv2.transform(S, ones), 1.0)
    b_sum = cv2.add(cv2.transform(B, ones), 1.0)
    cd = cv2.absdiff(cv2.divide(S, cv2.merge([s_sum] * 3)), cv2.divide(B, cv2.merge([b_sum] * 3)))
    c0, c1, c2 = cv2.split(cd)
    chroma = cv2.max(cv2.max(c0, c1), c2)
    del cd, c0, c1, c2, S
    # Camera sharpening leaves a thin halo of *brighter*, neutral backdrop
    # next to dark or gold edges (and inside chain links); discount it so the
    # matte does not grow a light outline. Clearly brighter pixels (diamonds,
    # white metal) still count.
    halo = (lum > 0) & (chroma < 0.03)
    d = np.where(halo, np.maximum(d - float(cfg.get("edge_halo_allowance", 35.0)), 0), d)
    I = np.clip(d / 60.0, 0, 1).astype(np.float32)
    p = alpha.astype(np.float32) / 255.0
    eps = float(cfg["edge_eps"])
    hh, ww = I.shape
    if sub > 1 and hh >= 2 * sub and ww >= 2 * sub:
        Is = cv2.resize(I, (ww // sub, hh // sub), interpolation=cv2.INTER_AREA)
        ps = cv2.resize(p, (ww // sub, hh // sub), interpolation=cv2.INTER_AREA)
        rs = max(1, r // sub)
    else:
        Is, ps, rs, sub = I, p, r, 1
    mI, mp = _box(Is, rs), _box(ps, rs)
    cov = _box(Is * ps, rs) - mI * mp
    var = _box(Is * Is, rs) - mI * mI
    ca = cov / (var + eps)
    cb = mp - ca * mI
    ma, mb = _box(ca, rs), _box(cb, rs)
    if sub > 1:
        ma = cv2.resize(ma, (ww, hh), interpolation=cv2.INTER_LINEAR)
        mb = cv2.resize(mb, (ww, hh), interpolation=cv2.INTER_LINEAR)
    q = np.clip(ma * I + mb, 0, 1)
    k_contrast = float(cfg["edge_contrast"])
    if k_contrast > 1.0:
        # tighten the transition around 50 % so the edge reads as crisp
        # (about half the width) while staying anti-aliased
        q = np.clip((q - 0.5) * k_contrast + 0.5, 0, 1)
    return q


def refine_edges_core(src: np.ndarray, alpha: np.ndarray, bg: np.ndarray, cfg: dict, diag: float) -> int:
    """Guided-filter edge refinement (He et al.) at native resolution.

    The model's matte is computed at ~1024 px and stretched to the crop, so
    on large photos its edges are soft and stair-stepped. The matte is
    re-fitted locally as a linear function of a guide image: "how different
    is this pixel from the local backdrop", which separates gold, stones and
    beads from the backdrop far better than brightness alone. Only a band
    around the existing edges changes. On big crops the filter coefficients
    are computed at half resolution (fast guided filter) and applied with the
    full-resolution guide, so edges stay native-sharp.
    """
    r = max(2, int(round(float(cfg["edge_radius_ratio"]) * diag)))
    solid = (alpha >= 128).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    edge = cv2.morphologyEx(solid, cv2.MORPH_GRADIENT, k).astype(bool) | ((alpha > 8) & (alpha < 247))
    if not edge.any():
        return 0
    band = cv2.dilate(edge.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * r + 1, 4 * r + 1))).astype(bool)
    ys, xs = np.nonzero(band)
    y0, y1 = max(0, ys.min() - 2 * r), min(alpha.shape[0], ys.max() + 2 * r + 1)
    x0, x1 = max(0, xs.min() - 2 * r), min(alpha.shape[1], xs.max() + 2 * r + 1)

    hh, ww = y1 - y0, x1 - x0
    sub = 2 if hh * ww > 4_000_000 and r >= 4 else 1
    # Process in horizontal strips (identical result thanks to the overlap,
    # but memory stays flat even for a 30+ MP crop).
    overlap = 4 * r + 4
    overlap += overlap % 2
    strip = 1024
    q = np.empty((hh, ww), np.float32)
    for sy in range(0, hh, strip):
        ey = min(hh, sy + strip)
        py0, py1 = max(0, sy - overlap), min(hh, ey + overlap)
        q[sy:ey] = _guided_strip(
            src[y0 + py0:y0 + py1, x0:x1], bg[y0 + py0:y0 + py1, x0:x1],
            alpha[y0 + py0:y0 + py1, x0:x1], r, sub, cfg)[sy - py0:ey - py0]
    sub_band = band[y0:y1, x0:x1]
    new = (q * 255 + 0.5).astype(np.uint8)
    region = alpha[y0:y1, x0:x1]
    changed = int((region[sub_band] != new[sub_band]).sum())
    region[sub_band] = new[sub_band]
    return changed


def open_metal_pockets(ctx: MatteContext, alpha: np.ndarray) -> int:
    """Open the small backdrop pockets inside chain links.

    Through a cable chain you see the backdrop, often a little *brighter*
    than elsewhere (lens sharpening halo), so the regular hollow rule does not
    apply. A pocket is opened only if it is small, neutral like the backdrop
    (not gold), flat, framed by coloured metal - and contains no dark pixels:
    every diamond, even a tiny melee, has dark facets.
    """
    cfg = ctx.cfg
    if not cfg["open_metal_pockets"]:
        return 0
    cand = (
        (alpha > 0)
        & (ctx.chroma_diff < 0.03)
        & (ctx.lum_diff > -20)
        & (ctx.color_dist < 50)
    ).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=4)
    if count <= 1:
        return 0
    max_area = float(cfg["pocket_max_area_ratio"]) * ctx.area
    # dark AND neutral = diamond facet / pavilion (gold shading is coloured)
    dark = ((ctx.lum_diff < -float(cfg["pocket_dark_guard"])) & (ctx.chroma_diff < 0.04)).astype(np.uint8)
    dark_near = cv2.dilate(dark, np.ones((5, 5), np.uint8)).ravel()
    flat = labels.ravel()
    areas = stats[:, cv2.CC_STAT_AREA]
    has_dark = np.bincount(flat, weights=dark_near.astype(np.float64), minlength=count) > 0
    fine = np.bincount(flat, weights=ctx.fine_texture.ravel().astype(np.float64), minlength=count) / np.maximum(areas, 1)
    chosen = []
    for idx in range(1, count):
        if areas[idx] < 4 or areas[idx] > max_area or has_dark[idx] or fine[idx] > 16:
            continue
        x, y, w, h = stats[idx, :4]
        if x == 0 or y == 0 or x + w == alpha.shape[1] or y + h == alpha.shape[0]:
            continue
        if _bordered_by_coloured_metal(ctx, labels, stats, idx, alpha, cfg):
            chosen.append(idx)
    if not chosen:
        return 0
    pocket = np.isin(labels, chosen)
    alpha[pocket] = 0
    return int(pocket.sum())


def remove_thin_gaps(ctx: MatteContext, alpha: np.ndarray) -> int:
    """Clear narrow strips of backdrop between two strands of coloured metal.

    The gap between the two wires of a hook, or between a wire and the next
    scroll, is only a few pixels wide. It is tinted by the metal beside it, so
    it never matches the backdrop closely enough for the hole passes, and the
    texture window straddles the metal so it never looks flat. Left alone it
    shows up as a pale outline around every wire. A strip qualifies when it is
    long and thin (a stone is a compact blob), neutral, not brighter than the
    backdrop, has no sparkle at its rim, and is framed by coloured metal.
    """
    cfg = ctx.cfg
    if not cfg.get("remove_thin_gaps", True):
        return 0
    tol = float(cfg.get("thin_gap_tolerance", 38.0))
    cand = (
        (alpha >= 128)
        & (ctx.color_dist < tol)
        & (ctx.chroma_diff < float(cfg.get("thin_gap_chroma", 0.03)))
        & (ctx.lum_diff <= float(cfg.get("thin_gap_max_brighter", 22.0)))
        & (ctx.lum_diff >= -float(cfg.get("thin_gap_max_darker", 30.0)))
    ).astype(np.uint8)
    if not cand.any():
        return 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=4)
    if count <= 1:
        return 0
    dist = cv2.distanceTransform(cand, cv2.DIST_L2, 3)
    thick = np.zeros(count, np.float32)
    np.maximum.at(thick, labels.ravel(), dist.ravel())
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    stone_d = float(cfg["stone_size_ratio"]) * ctx.diag
    min_area = max(12.0, 0.04 * 0.785 * stone_d * stone_d)
    max_half = float(cfg.get("thin_gap_half_width_ratio", 0.0011)) * ctx.diag
    elong = areas / np.maximum(np.pi * thick.astype(np.float64) ** 2, 1.0)
    ok = (areas >= min_area) & (thick <= max_half) & (elong >= float(cfg.get("thin_gap_elongation", 3.0)))
    ok[0] = False
    ids = np.nonzero(ok)[0]
    if ids.size == 0:
        return 0
    # no sparkle at the rim: a stone facet has a highlight right beside it
    sparkle = (ctx.lum_diff > 30).astype(np.uint8)
    sparkle_near = cv2.dilate(sparkle, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(np.float64)
    spark_frac = np.bincount(labels.ravel(), weights=sparkle_near.ravel(), minlength=count) / np.maximum(areas, 1)
    chosen = [int(i) for i in ids if spark_frac[i] < 0.05 and _bordered_by_coloured_metal(ctx, labels, stats, int(i), alpha, cfg)]
    if not chosen:
        return 0
    clear = np.isin(labels, chosen).astype(np.uint8)
    # take in the anti-aliased rim of the strip
    rim = cv2.dilate(clear, np.ones((3, 3), np.uint8)).astype(bool)
    rim &= (ctx.color_dist < tol * 1.3) & (ctx.chroma_diff < 0.035) & (ctx.lum_diff <= float(cfg.get("thin_gap_max_brighter", 22.0)))
    clear = clear.astype(bool) | rim
    before = int((alpha[clear] > 0).sum())
    alpha[clear] = 0
    ctx.thin_gap_mask = clear
    return before


def open_backdrop_hollows(ctx: MatteContext, alpha: np.ndarray) -> int:
    """Clear openings that show plain backdrop, whatever frames them.

    Openwork pendants and bead strands have many small windows framed by
    pave or white gold. The model often fills them, and the other passes
    need a coloured-metal frame or a large area. Here an opening qualifies
    when it is neutral, flat and matches the local backdrop closely. Diamonds
    that reflect the backdrop are protected: a stone always has dark facets
    or a sparkle right next to its backdrop-coloured facets, so a candidate
    with such pixels inside or at its rim is only cleared when it is clearly
    larger than a stone.
    """
    cfg = ctx.cfg
    if not cfg["open_backdrop_hollows"]:
        return 0
    neutral_flat = (ctx.chroma_diff < 0.03) & (ctx.lum_diff <= 6.0) & (ctx.lum_diff >= -10.0)
    seed = (
        (ctx.color_dist < float(cfg["hollow_seed_tolerance"]))
        & (_local_std(ctx.gray, 3) < 3.5)
        & neutral_flat
        & (alpha > 0)
    ).astype(np.uint8)
    if not seed.any():
        return 0
    grow_ok = ((ctx.color_dist < float(cfg["hollow_grow_tolerance"]) * 1.25) & neutral_flat).astype(np.uint8)
    region = seed.copy()
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(4):
        region = cv2.dilate(region, k3) & grow_ok
    count, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=4)
    if count <= 1:
        return 0
    flat = labels.ravel()
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    mean_dist = np.bincount(flat, weights=ctx.color_dist.ravel().astype(np.float64), minlength=count) / np.maximum(areas, 1)
    seed_frac = np.bincount(flat, weights=seed.ravel().astype(np.float64), minlength=count) / np.maximum(areas, 1)
    mean_alpha = np.bincount(flat, weights=alpha.ravel().astype(np.float64), minlength=count) / np.maximum(areas, 1)

    stone_d = float(cfg["stone_size_ratio"]) * ctx.diag
    stone_area = 0.785 * stone_d * stone_d
    min_area = max(4.0, 0.02 * stone_area)
    rim = max(2, int(round(stone_d * 0.12)))
    # A diamond shows dark pavilion facets AND a sparkle close together; a
    # black bead or dark metal shading has no sparkle, so it does not count.
    k_rim = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim + 1, 2 * rim + 1))
    k_stone = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(stone_d) + 1, 2 * int(stone_d) + 1))
    dark = ((ctx.lum_diff < -45) & (ctx.chroma_diff < 0.05)).astype(np.uint8)
    sparkle = (ctx.lum_diff > 30).astype(np.uint8)
    stone_zone = cv2.dilate(dark, k_stone) & cv2.dilate(sparkle, k_stone)
    facet = ((dark | sparkle) & stone_zone).astype(np.uint8)
    facet_near = cv2.dilate(facet, k_rim)
    facet_frac = np.bincount(flat, weights=facet_near.ravel().astype(np.float64), minlength=count) / np.maximum(areas, 1)

    chosen = []
    for idx in range(1, count):
        if mean_alpha[idx] < 8 or areas[idx] < min_area:
            continue
        big = areas[idx] >= 1.5 * stone_area
        # Openings bigger than a stone cannot be a reflection in a stone, so a
        # looser match is accepted (soft photos, strips between bead strands).
        max_dist = float(cfg["hollow_max_mean_dist"]) * (1.5 if big else 1.0)
        if mean_dist[idx] > max_dist or seed_frac[idx] < (0.12 if big else 0.35):
            continue
        if not big and facet_frac[idx] > 0.25:
            continue        # backdrop reflected in a stone's facets
        chosen.append(idx)
    if not chosen:
        return 0
    clear = np.isin(labels, chosen)
    # include the anti-aliased rim of each opening
    clear = cv2.dilate(clear.astype(np.uint8), k3).astype(bool) & (ctx.color_dist < float(cfg["hollow_grow_tolerance"]) * 1.4) & (ctx.lum_diff <= 6.0) & (ctx.chroma_diff < 0.035)
    before = int((alpha[clear] > 0).sum())
    alpha[clear] = 0
    return before


def _exterior_colour(src: np.ndarray, before: np.ndarray) -> np.ndarray | None:
    """Median colour of the backdrop around the product (clear in the model's matte)."""
    clear = cv2.erode((before <= 2).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if clear.sum() < 200:
        return None
    ys, xs = np.nonzero(clear)
    step = max(1, len(ys) // 20000)
    return np.median(src[ys[::step], xs[::step]].astype(np.float32), axis=0)


def _open_backdrop_mask(ctx: MatteContext, before: np.ndarray) -> np.ndarray:
    """Pixels of open backdrop seen through the piece: flat, and the same
    colour as the backdrop around the product. A stone's facet reflection is
    off-colour and has facet edges inside it, however broad it is, so it never
    qualifies; the stone-reflection restore must not bring these back."""
    cfg = ctx.cfg
    empty = np.zeros(before.shape, bool)
    ext = _exterior_colour(ctx.src, before)
    if ext is None:
        return empty
    src_float = ctx.src.astype(np.float32)
    ext_float = np.asarray(ext, dtype=np.float32).reshape(1, 1, 3)

    diff = np.abs(src_float - ext_float)
    dist = cv2.sqrt(cv2.transform(cv2.multiply(diff, diff), np.ones((1, 3), np.float32)))
    m = ((dist < float(cfg.get("open_backdrop_max_dist", 26.0))) & (ctx.chroma_diff < 0.02) & (ctx.lum_diff <= float(cfg["hole_bright_tolerance"])) & (ctx.fine_texture < float(cfg.get("open_backdrop_max_fine", 3.0)))).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    if not m.any():
        return empty
    stone_d = float(cfg["stone_size_ratio"]) * ctx.diag
    min_area = float(cfg.get("flat_region_min_stone_areas", 0.35)) * 0.785 * stone_d * stone_d
    count, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    big = np.nonzero(stats[:, cv2.CC_STAT_AREA] >= min_area)[0]
    big = big[big > 0]
    if big.size == 0:
        return empty
    return np.isin(labels, big)


def restore_stone_reflections(ctx: MatteContext, before: np.ndarray, alpha: np.ndarray) -> int:
    """Undo cuts that were really a diamond reflecting the backdrop.

    Large brilliant-cut stones have broad facets that mirror the grey studio
    backdrop, and the clean-up passes can mistake them for openings (holes
    punched into solitaires). Measured on real photos, true openings match
    the nearby backdrop almost exactly (median difference 2-7), while a facet
    reflection is clearly off (10-35) and darker than the backdrop. Any region
    the model was confident about, that a pass cut out, and that looks like
    a reflection, is restored.
    """
    cfg = ctx.cfg
    if not cfg["protect_stone_reflections"]:
        return 0
    open_mask = _open_backdrop_mask(ctx, before)
    ctx.open_backdrop = open_mask
    cut = ((before >= int(cfg["reflection_model_alpha"])) & (alpha <= 127) & ~open_mask).astype(np.uint8)
    if not cut.any():
        return 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cut, connectivity=8)
    restored = 0
    stone_d = float(cfg["stone_size_ratio"]) * ctx.diag
    stone_area = 0.785 * stone_d * stone_d
    min_d = float(cfg["reflection_min_dist"])
    max_lum = float(cfg["reflection_max_lum"])
    for idx in range(1, count):
        x, y, w, h = stats[idx, :4]
        comp = labels[y:y + h, x:x + w] == idx
        d_med = float(np.median(ctx.color_dist[y:y + h, x:x + w][comp]))
        lum_med = float(np.median(ctx.lum_diff[y:y + h, x:x + w][comp]))
        small = stats[idx, cv2.CC_STAT_AREA] < 0.5 * stone_area
        if lum_med < max_lum and (d_med > min_d or (small and d_med > 0.6 * min_d)):
            region = alpha[y:y + h, x:x + w]
            region[comp] = before[y:y + h, x:x + w][comp]
            restored += int(comp.sum())
    return restored


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


def _backdrop_zone(ctx: MatteContext, alpha: np.ndarray) -> np.ndarray:
    """Pixels that sit on, or right beside, a flat patch of plain backdrop.

    A stone is faceted: even a bright flat facet is only a few pixels wide.
    A gap between two wires, or the backdrop around the outline, is a large
    flat area that matches the backdrop colour and hue. The rim of such an
    area measures as "textured" (the measuring window straddles the metal),
    which made earlier passes mistake the rim for a stone and give it back
    as a pale outline. Anything within the window's reach of a sizeable flat
    backdrop patch that is already clear stays clear.
    """
    cfg = ctx.cfg
    bright = float(cfg["hole_bright_tolerance"])
    flat = (
        (ctx.color_dist < float(cfg.get("band_color_tolerance", 16.0)))
        & (ctx.fine_texture < float(cfg.get("band_flat_texture", 3.5)))
        & (ctx.chroma_diff < float(cfg.get("band_chroma_tolerance", 0.025)))
        & (ctx.lum_diff <= bright)
        & (alpha < 128)
    )
    flat = cv2.morphologyEx(flat.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    thin = getattr(ctx, "thin_gap_mask", None)
    thin_zone = cv2.dilate(thin.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) if thin is not None else None
    if not flat.any():
        return thin_zone if thin_zone is not None else np.zeros(alpha.shape, bool)
    stone_r = float(cfg["stone_size_ratio"]) * ctx.diag / 2.0
    min_area = float(cfg.get("band_min_area_ratio", 0.12)) * np.pi * stone_r * stone_r
    count, labels, stats, _ = cv2.connectedComponentsWithStats(flat, connectivity=4)
    big = np.nonzero(stats[:, cv2.CC_STAT_AREA] >= min_area)[0]
    big = big[big > 0]
    if big.size == 0:
        return thin_zone if thin_zone is not None else np.zeros(alpha.shape, bool)
    core = np.isin(labels, big).astype(np.uint8)
    reach = max(2, int(round(float(cfg.get("band_reach_ratio", 0.0012)) * ctx.diag)))
    zone = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))).astype(bool)
    return zone | thin_zone if thin_zone is not None else zone


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
    band = _backdrop_zone(ctx, alpha)
    open_bd = getattr(ctx, "open_backdrop", None)
    if open_bd is not None:
        band = band | cv2.dilate(open_bd.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
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
                fill = np.isin(labels, fill_ids) & ~backdrop & ~band & (faceted | metal) & (alpha < 255)
                alpha[fill] = 255
                changed += int(fill.sum())

    edge = max(1, int(round(float(cfg["stone_edge_px_ratio"]) * ctx.diag)))
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge + 1, 2 * edge + 1))
    interior = cv2.erode((alpha >= 64).astype(np.uint8), edge_kernel).astype(bool)

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
    notch = closed & (alpha < 255) & stone_like & ~backdrop & ~band & (~backdrop_coloured | surrounded)
    # A dropped stone leaves a compact, stone-sized notch. Thin slivers along
    # an outline (closing a curved chain, sharpening halo) are not stones.
    open_area = (closed & (alpha < 128)).astype(np.uint8)
    if open_area.any():
        n_open, lab_open = cv2.connectedComponents(open_area, connectivity=8)
        thickness = cv2.distanceTransform(open_area, cv2.DIST_L2, 3)
        max_thick = np.zeros(n_open, np.float32)
        np.maximum.at(max_thick, lab_open.ravel(), thickness.ravel())
        thick_ids = np.nonzero(max_thick >= 0.35 * r)[0]
        thick_ids = thick_ids[thick_ids > 0]
        compact = np.isin(lab_open, thick_ids) | interior
        notch &= compact
    else:
        notch &= interior
    alpha[notch] = 255
    changed += int(notch.sum())

    # semi-transparent interior pixels (soft outline left untouched)
    # clearly not backdrop: different colour AND different hue (shadow is not metal)
    # (shadows are only mildly darker; dark stone facets / metal are much darker)
    bright = float(cfg["hole_bright_tolerance"]) * 2
    metal_like = (ctx.color_dist >= float(cfg["hole_grow_tolerance"])) & (
        (ctx.chroma_diff >= 0.04) | (ctx.lum_diff < -45) | (ctx.lum_diff > bright))
    raise_px = interior & (alpha < 255) & ~backdrop & ~band & (stone_like | metal_like) & (~backdrop_coloured | surrounded)
    alpha[raise_px] = 255
    changed += int(raise_px.sum())
    return changed


def _colour_measures(src: np.ndarray, bg: np.ndarray):
    """(dist, dlum, chroma, warm) between a photo region and its backdrop.

    dist   - colour distance to the backdrop
    dlum   - brightness difference (positive = brighter than the backdrop)
    chroma - largest per-channel difference of the normalised colours
    warm   - red minus blue share of the photo colour (gold / rose metal)

    Per-channel OpenCV arithmetic: same numbers as the numpy expressions it
    replaces, without the slow reductions over the colour axis.
    """
    c0, c1, c2 = cv2.split(np.ascontiguousarray(src, dtype=np.float32))
    b0, b1, b2 = cv2.split(np.ascontiguousarray(bg, dtype=np.float32))
    d0, d1, d2 = c0 - b0, c1 - b1, c2 - b2
    dist = cv2.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    dlum = (d0 + d1 + d2) / np.float32(3.0)
    ss = c0 + c1 + c2 + 1.0
    bs = b0 + b1 + b2 + 1.0
    n0, n2 = c0 / ss, c2 / ss
    chroma = cv2.max(cv2.max(cv2.absdiff(n0, b0 / bs), cv2.absdiff(c1 / ss, b1 / bs)),
                     cv2.absdiff(n2, b2 / bs))
    return dist, dlum, chroma, n0 - n2


def clear_backdrop_haze(src: np.ndarray, bg: np.ndarray, alpha: np.ndarray, cfg: dict) -> int:
    """Final pass: a semi-transparent pixel that is exactly backdrop colour is
    haze, not product. A real edge pixel is a blend with the metal, so its
    colour is off the backdrop. Works in strips (large photos)."""
    if not cfg.get("clear_backdrop_haze", True):
        return 0
    tol = float(cfg.get("haze_color_tolerance", 20.0))
    bright = float(cfg.get("haze_max_brighter", 8.0))
    cleared = 0
    step = 512
    for y in range(0, alpha.shape[0], step):
        sl = slice(y, min(alpha.shape[0], y + step))
        a = alpha[sl]
        cand = (a > 0) & (a < float(cfg.get("haze_max_alpha", 200)))
        if not cand.any():
            continue
        dist, lum, chroma, _ = _colour_measures(src[sl], bg[sl])
        haze = cand & (dist < tol) & (chroma < float(cfg.get("haze_chroma", 0.03))) & (lum <= bright)
        cleared += int(haze.sum())
        a[haze] = 0
    return cleared


def clear_flat_backdrop(src: np.ndarray, bg: np.ndarray, alpha: np.ndarray, cfg: dict) -> int:
    """Final pass: large, solid, neutral patches whose colour is the backdrop
    (pale grey/white openings framed by metal) are cleared even when the model
    left them opaque. Diamonds are speckled by facets, so a morphological open
    removes them from the candidate mask; only solid blobs survive."""
    if not cfg.get("clear_flat_backdrop", True):
        return 0
    H, W = alpha.shape
    s = max(1, int(round(max(H, W) / float(cfg.get("flat_backdrop_ref_px", 3000)))))
    if s > 1:
        small = lambda x: cv2.resize(x, (W // s, H // s), interpolation=cv2.INTER_AREA)
    else:
        small = lambda x: x
    S = small(src).astype(np.float32)
    B = small(bg).astype(np.float32)
    A = small(alpha)
    dist, _, chroma, warm_n = _colour_measures(S, B)
    m = ((A > 0) & (dist < float(cfg.get("flat_backdrop_dist", 60.0)))
         & (chroma < float(cfg.get("flat_backdrop_chroma", 0.04)))).astype(np.uint8)
    k = int(cfg.get("flat_backdrop_open", 7))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    min_area = float(cfg.get("flat_backdrop_min_area", 120)) * (2.0 / s) ** 2 if s < 2 else float(cfg.get("flat_backdrop_min_area", 120))
    keep_ids = np.flatnonzero(st[:, cv2.CC_STAT_AREA] >= min_area)
    keep_ids = keep_ids[keep_ids > 0]
    if keep_ids.size == 0:
        return 0
    # A real opening is framed by metal; a flat facet inside a diamond is
    # framed by other (neutral) facets. Keep only blobs whose border is mostly
    # warm metal.
    warm = warm_n > float(cfg.get("flat_backdrop_warm", 0.06))
    need = float(cfg.get("flat_backdrop_metal_frac", 0.6))
    ok = []
    k3, k9 = np.ones((3, 3), np.uint8), np.ones((9, 9), np.uint8)
    for i in keep_ids:
        x, y, w, h = st[i, 0], st[i, 1], st[i, 2], st[i, 3]
        y0, y1, x0, x1 = max(0, y - 6), min(m.shape[0], y + h + 6), max(0, x - 6), min(m.shape[1], x + w + 6)
        sel = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
        ring = (cv2.dilate(sel, k9) > 0) & (cv2.dilate(sel, k3) == 0)
        if not (ring.any() and float(warm[y0:y1, x0:x1][ring].mean()) >= need):
            continue
        core = cv2.erode(sel, k3, iterations=2) > 0
        if core.sum() < 10:
            core = sel > 0
        # a true opening is the backdrop colour; a facet that merely looks pale is not
        if float(dist[y0:y1, x0:x1][core].mean()) <= float(cfg.get("flat_backdrop_core_dist", 16.0)):
            ok.append(i)
    if not ok:
        return 0
    keep = np.isin(lab, np.array(ok)).astype(np.uint8)
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    if s > 1:
        keep = cv2.resize(keep, (W, H), interpolation=cv2.INTER_NEAREST)
    sel = (keep > 0) & (alpha > 0)
    cleared = int(sel.sum())
    alpha[sel] = 0
    return cleared


def _finish_strip(src, bg, alpha, cfg, sigma, protect_rows, ref=None):
    """One strip of finish_edges. Returns the new alpha (uint8) for the strip."""
    a = alpha.astype(np.float32) / 255.0
    solid = (a >= 0.5).astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    band_w = int(cfg.get("finish_band_px", 3))
    kb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_w + 1, 2 * band_w + 1))
    contour = cv2.morphologyEx(solid, cv2.MORPH_GRADIENT, k3).astype(bool)
    edge = contour | ((a > 0.01) & (a < 0.99))
    band = cv2.dilate(edge.astype(np.uint8), kb) > 0

    d, lum, chroma, warm_n = _colour_measures(src, bg)
    # Stone clusters / texture: lots of contour in a small window, or sparkle
    # right next to it. Facets and diamond shine are left exactly as they are.
    zr = int(cfg.get("finish_zone_px", 21))
    density = cv2.boxFilter(contour.astype(np.float32), -1, (zr, zr))
    spark = cv2.dilate((lum > float(cfg.get("finish_sparkle", 18.0))).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    stone = (density > float(cfg.get("finish_zone_density", 0.16))) | spark
    stone = cv2.dilate(stone.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    band &= ~stone

    out = a.copy()
    # 1. smooth the contour: blur, then re-sharpen around 50 % so the edge is
    #    anti-aliased and free of stair-steps but not soft. Only the band moves.
    if cfg.get("smooth_edges", True):
        bl = cv2.GaussianBlur(a, (0, 0), sigma)
        q = np.clip((bl - 0.5) * float(cfg.get("smooth_gain", 2.4)) + 0.5, 0.0, 1.0)
        # thin wires and small gaps keep their width: never smooth below the
        # original where the feature is narrow (blurred peak < 0.9)
        thin = (bl < 0.9) & (a > 0.9)
        q = np.where(thin, np.maximum(q, a * 0.92), q)
        out = np.where(band, q, out)

    # 2. remove backdrop-coloured fringe on the edge. A genuine edge pixel is a
    #    blend with the metal/stone, so its colour is clearly off the backdrop.
    #    Pixels that are brighter than the backdrop (diamond shine, sparkle)
    #    or coloured are never touched.
    if cfg.get("defringe", True):
        d0, d1 = float(cfg.get("defringe_d0", 9.0)), float(cfg.get("defringe_d1", 30.0))
        cap = np.clip((d - d0) / max(1e-3, d1 - d0), 0.0, 1.0)
        bgish = (band & (out < float(cfg.get("defringe_max_alpha", 0.97)))
                 & (chroma < float(cfg.get("defringe_chroma", 0.04))) & (lum <= float(cfg.get("defringe_max_brighter", 4.0))))
        out = np.where(bgish, np.minimum(out, np.maximum(cap, 0.0)), out)

    # 3. tiny enclosed holes (gaps between links, bead holes, filigree dots)
    if cfg.get("tiny_holes", True):
        dh = float(cfg.get("tiny_hole_dist", 26.0))
        m = ((out > 0.5) & (d < dh) & (chroma < float(cfg.get("tiny_hole_chroma", 0.035))) & (lum <= 4.0)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n > 1:
            warm = warm_n > float(cfg.get("flat_backdrop_warm", 0.06))
            hmax = float(cfg.get("tiny_hole_max_px", 400))
            H, W = m.shape
            for i in range(1, n):
                area = st[i, cv2.CC_STAT_AREA]
                if area < int(cfg.get("tiny_hole_min_px", 10)) or area > hmax:
                    continue
                x, y, w, h = st[i, 0], st[i, 1], st[i, 2], st[i, 3]
                if y <= protect_rows[0] or y + h >= protect_rows[1]:
                    continue
                y0, y1, x0, x1 = max(0, y - 4), min(H, y + h + 4), max(0, x - 4), min(W, x + w + 4)
                sel = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
                ring = (cv2.dilate(sel, np.ones((7, 7), np.uint8)) > 0) & (cv2.dilate(sel, k3) == 0)
                if not ring.any():
                    continue
                # framed by metal on most of its border, not by facets ...
                if float(warm[y0:y1, x0:x1][ring].mean()) < float(cfg.get("tiny_hole_metal_frac", 0.6)):
                    continue
                inside = sel > 0
                if stone[y0:y1, x0:x1][inside].any():
                    continue
                # ... flat inside (a facet has shading), the same colour as the
                # backdrop, and with no sparkle next to it (diamond shine)
                if float(lum[y0:y1, x0:x1][inside].std()) > float(cfg.get("tiny_hole_max_std", 3.0)):
                    continue
                if float(d[y0:y1, x0:x1][inside].mean()) > float(cfg.get("tiny_hole_core_dist", 14.0)):
                    continue
                # the model itself was unsure here (a flat plate of pale metal
                # is solid in the model matte, a real opening is not)
                if ref is not None and float((ref[y0:y1, x0:x1][inside].astype(np.float32) / 255.0).mean()) > float(cfg.get("tiny_hole_model_max", 0.88)):
                    continue
                near = cv2.dilate(sel, np.ones((11, 11), np.uint8)) > 0
                if float((lum[y0:y1, x0:x1][near] > float(cfg.get("tiny_hole_sparkle", 18.0))).mean()) > 0.02:
                    continue
                out[y0:y1, x0:x1][cv2.dilate(sel, k3) > 0] = 0.0
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


def finish_edges(src: np.ndarray, bg: np.ndarray, alpha: np.ndarray, cfg: dict, diag: float, model_alpha=None) -> int:
    """Last step: smooth the contour, drop backdrop-coloured fringe and clear
    tiny enclosed holes, in strips so 30 MP crops stay light on memory."""
    if not cfg.get("finish_edges", True):
        return 0
    H, W = alpha.shape
    sigma = float(np.clip(float(cfg.get("smooth_sigma_ratio", 0.00018)) * diag, 0.7, 1.6))
    overlap = 48
    strip = 1024
    changed = 0
    for sy in range(0, H, strip):
        ey = min(H, sy + strip)
        py0, py1 = max(0, sy - overlap), min(H, ey + overlap)
        part = alpha[py0:py1]
        if not ((part > 0) & (part < 255)).any() and not part.any():
            continue
        new = _finish_strip(src[py0:py1], bg[py0:py1], part, cfg, sigma,
                            (overlap if py0 > 0 else -1, (py1 - py0) - overlap if py1 < H else (py1 - py0) + 1),
                            None if model_alpha is None else model_alpha[py0:py1])
        a_old = alpha[sy:ey]
        a_new = new[sy - py0:ey - py0]
        changed += int((a_old != a_new).sum())
        alpha[sy:ey] = a_new
    return changed


def clear_backdrop_wedges(src: np.ndarray, bg: np.ndarray, alpha: np.ndarray, cfg: dict, model_alpha=None) -> int:
    """Native-resolution pass for small openings between spokes, wires and
    stone rows that the model left as pale backdrop. A patch is cleared only
    if its colour IS the backdrop (tiny distance, flat, neutral), it has no
    sparkle inside, and it touches metal. Works in strips."""
    if not cfg.get("clear_wedges", True):
        return 0
    H, W = alpha.shape
    ys = np.flatnonzero((alpha > 0).any(axis=1))
    if ys.size == 0:
        return 0
    k3 = np.ones((3, 3), np.uint8)
    kopen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(cfg.get("wedge_open", 5)),) * 2)
    amin, amax = int(cfg.get("wedge_min_px", 90)), int(cfg.get("wedge_max_px", 9000))
    overlap, strip = 96, 1024
    cleared = 0
    for sy in range(int(ys[0]), int(ys[-1]) + 1, strip):
        ey = min(H, sy + strip)
        py0, py1 = max(0, sy - overlap), min(H, ey + overlap)
        a = alpha[py0:py1]
        if not (a > 0).any():
            continue
        d, lum, chroma, warm_n = _colour_measures(src[py0:py1], bg[py0:py1])
        m = ((a > 0) & (d < float(cfg.get("wedge_dist", 24.0))) & (chroma < float(cfg.get("wedge_chroma", 0.03)))
             & (lum <= 6.0)).astype(np.uint8)
        if not m.any():
            continue
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kopen)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            continue
        warm = warm_n > float(cfg.get("flat_backdrop_warm", 0.06))
        h_, w_ = m.shape
        top_ok = overlap if py0 > 0 else -1
        bot_ok = (py1 - py0) - overlap if py1 < H else (py1 - py0) + 1
        for i in range(1, n):
            area = st[i, cv2.CC_STAT_AREA]
            if area < amin or area > amax:
                continue
            x, y, w, h = st[i, 0], st[i, 1], st[i, 2], st[i, 3]
            if y <= top_ok or y + h >= bot_ok:
                continue
            y0, y1, x0, x1 = max(0, y - 5), min(h_, y + h + 5), max(0, x - 5), min(w_, x + w + 5)
            sel = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
            inside = sel > 0
            if float(d[y0:y1, x0:x1][inside].mean()) > float(cfg.get("wedge_core_dist", 9.0)):
                continue
            if float(lum[y0:y1, x0:x1][inside].std()) > float(cfg.get("wedge_max_std", 4.0)):
                continue
            if float(lum[y0:y1, x0:x1][inside].max()) > float(cfg.get("wedge_sparkle", 22.0)):
                continue
            ring = (cv2.dilate(sel, np.ones((9, 9), np.uint8)) > 0) & (cv2.dilate(sel, k3) == 0)
            if not ring.any():
                continue
            wf = float(warm[y0:y1, x0:x1][ring].mean())
            core_d = float(d[y0:y1, x0:x1][inside].mean())
            sd = float(lum[y0:y1, x0:x1][inside].std())
            if wf >= float(cfg.get("wedge_metal_frac", 0.12)) and (area >= 90 or wf >= 0.25):
                pass                                      # touches metal: normal rule above
            elif (area >= int(cfg.get("wedge_bare_min_px", 100)) and core_d <= float(cfg.get("wedge_bare_dist", 6.5))
                  and sd <= float(cfg.get("wedge_bare_std", 3.4))):
                pass                                      # framed by stones only, but exactly backdrop-flat
            else:
                continue
            grow = cv2.dilate(sel, k3) > 0
            sub = alpha[py0 + y0:py0 + y1, x0:x1]
            cleared += int((sub[grow] > 0).sum())
            sub[grow] = 0
    return cleared


def _light_backdrop(bg: np.ndarray, cfg: dict) -> bool:
    b = bg[::4, ::4].astype(np.float32).reshape(-1, 3)
    lum = float(np.median(b.mean(axis=1)))
    mx, mn = np.median(b.max(axis=1)), np.median(b.min(axis=1))
    return lum >= float(cfg.get("light_backdrop_lum", 170.0)) and \
        (mx - mn) / max(mx, 1.0) < float(cfg.get("light_backdrop_max_chroma", 0.12))


class _PixelFeatures:
    """Per-pixel colour measures shared by the white/grey backdrop passes.

    Computed once per image (``_features``) with per-channel OpenCV ops and
    reused by the guard and the pocket pass.

    The thresholds of those passes (texture windows, "flat" shares) were tuned
    on photos of about 4000 px diagonal. On a 30 MP photo the same diamond
    facet is twice as large, looks twice as "flat" in a 3 px window, and gets
    mistaken for backdrop. So for large photos the measures are computed on a
    copy reduced to that analysis scale and then enlarged back: the decisions
    are the same at every resolution (and it is faster).
    """

    ANALYSIS_DIAG = 4000.0

    def __init__(self, src: np.ndarray, bg: np.ndarray, photo_diag: float | None = None):
        h, w = src.shape[:2]
        diag = float(photo_diag or np.hypot(h, w))
        f = max(1, int(round(diag / self.ANALYSIS_DIAG)))
        self.factor = f
        self._full = (w, h)
        s = src if src.dtype == np.float32 else src.astype(np.float32)
        b = bg if bg.dtype == np.float32 else bg.astype(np.float32)
        if f > 1:
            small = (max(1, w // f), max(1, h // f))
            s = cv2.resize(s, small, interpolation=cv2.INTER_AREA)
            b = cv2.resize(b, small, interpolation=cv2.INTER_AREA)
        c0, c1, c2 = cv2.split(np.ascontiguousarray(s))
        b0, b1, b2 = cv2.split(np.ascontiguousarray(b))
        mx = cv2.max(cv2.max(c0, c1), c2)
        mn = cv2.min(cv2.min(c0, c1), c2)
        lum = (c0 + c1 + c2) / np.float32(3.0)
        dlum = lum - (b0 + b1 + b2) / np.float32(3.0)
        d0, d1, d2 = c0 - b0, c1 - b1, c2 - b2
        dist = cv2.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
        sat = (mx - mn) / cv2.max(mx, 1.0)
        self._gray_small = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)
        up = self._up
        self.mx, self.mn, self.lum, self.dlum = up(mx), up(mn), up(lum), up(dlum)
        self.dist, self.sat = up(dist), up(sat)
        self._tex = {}

    def _up(self, img: np.ndarray) -> np.ndarray:
        if self.factor == 1:
            return img
        return cv2.resize(img, self._full, interpolation=cv2.INTER_LINEAR)

    def tex(self, k: int) -> np.ndarray:
        """Local contrast in a k x k window (at the analysis scale)."""
        if k not in self._tex:
            self._tex[k] = self._up(_local_std(self._gray_small, k))
        return self._tex[k]


_FEATURE_CACHE: list = []


def _features(src: np.ndarray, bg: np.ndarray, photo_diag: float | None = None) -> _PixelFeatures:
    """Return the shared features for this exact (src, bg) pair."""
    if _FEATURE_CACHE and _FEATURE_CACHE[0] is src and _FEATURE_CACHE[1] is bg:
        return _FEATURE_CACHE[2]
    feats = _PixelFeatures(src, bg, photo_diag)
    _FEATURE_CACHE[:] = [src, bg, feats]
    return feats


def _release_features() -> None:
    _FEATURE_CACHE.clear()


def _component_means(labels: np.ndarray, ids_mask: np.ndarray, count: int, images) -> list:
    """Mean of each image per component, using only the selected pixels.

    ``ids_mask`` marks the pixels to visit, so the work scales with the
    candidate regions instead of the whole image.
    """
    sel = np.flatnonzero(ids_mask)
    lab = labels.ravel()[sel]
    cnt = np.maximum(np.bincount(lab, minlength=count).astype(np.float64), 1.0)
    return [np.bincount(lab, weights=img.ravel()[sel].astype(np.float64), minlength=count) / cnt
            for img in images]


def _backdrop_mode(bg: np.ndarray, cfg: dict):
    """'clipped' (blown-out white), 'light' (light grey) or None."""
    b = bg[::4, ::4].astype(np.float32)
    bg_lum = float(np.median(b.mean(axis=2)))
    if bg_lum >= float(cfg.get("clipped_backdrop_lum", 245.0)):
        return "clipped", bg_lum
    if _light_backdrop(bg, cfg):
        return "light", bg_lum
    return None, bg_lum


def clear_white_pockets(src: np.ndarray, bg: np.ndarray, alpha: np.ndarray, cfg: dict,
                        photo_diag: float | None = None) -> int:
    """Final pass for white / light-grey backdrops: clear small backdrop pockets.

    Between chain links, beads and jump rings the backdrop shows through gaps
    of a few pixels. They are too small for the hole passes, so they stay
    opaque and show as white/grey dots on a dark page. A pocket is cleared when
    its rim is gold / black bead; a pale spot ringed by grey facets is diamond
    sparkle and is kept - unless it matches the backdrop exactly (an opening
    inside a ring of stones).
    """
    if not cfg.get("clear_white_pockets", True):
        return 0
    mode, bg_lum = _backdrop_mode(bg, cfg)
    if mode is None:
        return 0
    clipped = mode == "clipped"
    f = _features(src, bg, photo_diag)
    mx, mn, lum = f.mx, f.mn, f.lum
    visible = alpha > 0
    if clipped:
        white = (mn >= float(cfg.get("white_pocket_min_lum", 245.0))) & visible
    else:
        # grey backdrop seen through a jump ring / between limbs of a figure
        tex = f.tex(3)
        white = (f.dlum > -float(cfg.get("backdrop_pocket_darker", 15.0))) & \
                (f.dlum < float(cfg.get("backdrop_pocket_brighter", 40.0))) & \
                (f.sat < float(cfg.get("backdrop_pocket_sat", 0.2))) & \
                (tex < float(cfg.get("backdrop_pocket_texture", 9.0))) & visible
    white = white.astype(np.uint8)
    if not white.any():
        return 0
    # transparent area connected to the photo border = outside the jewellery;
    # a pocket touching it is the product's outer edge, not a hole
    clear_bg = (alpha < 64).astype(np.uint8)
    n_o, lab_o, st_o, _ = cv2.connectedComponentsWithStats(clear_bg, connectivity=8)
    keep_o = np.zeros(n_o, bool)
    keep_o[np.unique(np.concatenate([lab_o[0], lab_o[-1], lab_o[:, 0], lab_o[:, -1]]))] = True
    keep_o |= st_o[:, cv2.CC_STAT_AREA] > 0.05 * alpha.size
    keep_o[0] = False
    outside = keep_o[lab_o]
    if not clipped:
        # grey backdrops only: there the pocket colour test also matches the
        # pale rim of the product, which must not join a hole to the outside
        halo = cv2.dilate(outside.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        white[halo] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    if count <= 1:
        return 0
    areas = stats[:, cv2.CC_STAT_AREA]
    # pocket sizes are in pixels: scale them with the photo, or a jump ring
    # hole in a 30 MP photo is "too big" while the same hole at 3 MP is not
    scale = max(1.0, alpha.size / float(cfg.get("white_pocket_ref_px", 2_000_000)))
    max_px = int(float(cfg.get("white_pocket_max_px", 150)) * scale)
    big_px = max(max_px, int(float(cfg.get("white_pocket_big_ratio", 0.0002)) * alpha.size))
    big_alpha = float(cfg.get("white_pocket_big_max_alpha", 230))
    min_px = 0 if clipped else int(float(cfg.get("backdrop_pocket_min_ratio", 4e-6)) * alpha.size)

    exact = np.zeros(count, bool)
    if not clipped:
        exact_max = int(float(cfg.get("backdrop_pocket_exact_max_ratio", 0.001)) * alpha.size)
        maybe = (areas >= int(cfg.get("backdrop_pocket_exact_min_px", 40))) & (areas <= exact_max)
        maybe[0] = False
        if maybe.any():
            m_dl, m_sat, m_flat = _component_means(
                labels, maybe[labels], count,
                (np.abs(f.dlum), f.sat, (f.tex(3) < 3.0).astype(np.float32)))
            exact = maybe & (m_dl < float(cfg.get("backdrop_pocket_exact_lum", 8.0))) & \
                (m_flat >= float(cfg.get("backdrop_pocket_exact_flat_share", 0.15))) & \
                (m_sat < float(cfg.get("backdrop_pocket_exact_sat", 0.05)))

    # size gates for every component at once; only survivors reach the loop
    cand = (areas >= min_px) & ((areas <= big_px) | exact)
    cand[0] = False
    if not cand.any():
        return 0
    big = cand & (areas > max_px) & ~exact
    if big.any():
        (m_alpha,) = _component_means(labels, big[labels], count, (alpha,))
        cand &= ~(big & (m_alpha >= big_alpha))   # solid white highlight on gold stays

    grey_facet = ((mx - mn) < 20) & (lum > 90) & (lum < float(cfg.get("white_pocket_min_lum", 245.0)))
    if not clipped:
        grey_facet &= white == 0
    max_share = float(cfg.get("white_pocket_max_facet_share", 0.3))
    k5 = np.ones((5, 5), np.uint8)
    k9 = np.ones((9, 9), np.uint8)
    clear = np.zeros(alpha.shape, bool)
    for i in np.flatnonzero(cand):
        x, y, w, h, _ = stats[i]
        y0, y1, x0, x1 = max(0, y - 5), y + h + 5, max(0, x - 5), x + w + 5
        comp = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
        ring = (cv2.dilate(comp, k5) > 0) & (comp == 0) & (alpha[y0:y1, x0:x1] >= 128) & (white[y0:y1, x0:x1] == 0)
        if not ring.any():
            continue
        if not clipped:
            rim = (cv2.dilate(comp, k9) > 0) & (comp == 0)
            if outside[y0:y1, x0:x1][rim].any():
                continue                   # open to the outside: product edge
        if not exact[i] and float(grey_facet[y0:y1, x0:x1][ring].mean()) >= max_share:
            continue                       # diamond sparkle
        clear[y0:y1, x0:x1] |= comp.astype(bool)
    if not clear.any():
        return 0
    # soft edge: pale, neutral pixels right beside the pocket are mostly backdrop too
    edge = cv2.dilate(clear.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~clear
    edge &= ((mx - mn) < 25) & ((lum > 225) if clipped else (white > 0) | (lum > bg_lum - 20))
    n = int(np.count_nonzero(alpha[clear]))
    alpha[clear] = 0
    alpha[edge] = (alpha[edge] * 0.35).astype(np.uint8)
    return n


class _ClippedBackdropGuard:
    """Undo clean-up cuts that remove real product on white / light-grey backdrops.

    Most passes tell a diamond facet / polished highlight from a hollow by
    "brighter than the backdrop". On a blown-out white backdrop nothing can be
    brighter, and on light grey the stones are the backdrop's own grey, so
    white stones, bright rims and thin highlights look like "backdrop-coloured
    gaps" and get cut out. This guard runs after every hole/gap pass and puts
    back any cut region that lies in a stone zone (next to neutral, dark facet
    lines) - or, on white, that the model was confident about - unless the
    region looks like plain backdrop (a real opening).

    The per-pixel measures are built lazily, on the first cut, and shared with
    the pocket pass; per-cut statistics only visit the cut pixels.
    """

    def __init__(self, src, bg, model_alpha, cfg, diag):
        self.on = False
        self.restored = 0
        if not cfg.get("protect_on_white_backdrop", True) or model_alpha is None:
            return
        mode, _ = _backdrop_mode(bg, cfg)
        if mode is None or (mode == "light" and not cfg.get("protect_on_light_backdrop", True)):
            return
        self.on = True
        self.clipped = mode == "clipped"
        self.cfg = cfg
        self.model = model_alpha
        self._src, self._bg, self._diag = src, bg, diag
        self._ready = False

    def _prepare(self) -> None:
        if self._ready:
            return
        cfg, f = self.cfg, _features(self._src, self._bg, self._diag)
        self.dist = f.dist
        self.fine = f.tex(5)
        self.backdrop_like = (
            (np.abs(f.dlum) < float(cfg.get("light_backdrop_like_lum", 30.0)))
            & (f.sat < float(cfg.get("light_backdrop_like_sat", 0.10)))
            & (f.tex(3) < float(cfg.get("light_backdrop_like_texture", 6.0)))
        ).astype(np.float32)
        self.near = (f.dist < float(cfg.get("light_opening_near_dist", 10.0))).astype(np.float32)
        facets = (
            (f.sat < 0.10)                                       # diamonds are grey, gold is not
            & (f.dlum < -float(cfg.get("white_guard_facet_dark", 35.0)))
            & (f.lum > float(cfg.get("white_guard_facet_min_lum", 70.0)))  # not black beads
            & (self.fine > float(cfg.get("white_guard_facet_texture", 8.0)))
            & (self.model >= 128)
        ).astype(np.uint8)
        r = max(3, int(round(float(cfg.get("white_guard_stone_ratio", 0.004)) * self._diag)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        self.stone = cv2.morphologyEx(cv2.dilate(facets, k), cv2.MORPH_CLOSE, k).astype(np.float32)
        self._ready = True

    def _exact(self, means: dict) -> np.ndarray:
        cfg = self.cfg
        if self.clipped:
            ed = float(cfg.get("white_guard_exact_dist", 6.0))
            et = float(cfg.get("white_guard_exact_texture", 2.5))
        else:
            ed = float(cfg.get("light_guard_exact_dist", 10.0))
            et = float(cfg.get("light_guard_exact_texture", 3.5))
        return (means["dist"] < ed) & (means["fine"] < et)

    def __call__(self, before: np.ndarray, alpha: np.ndarray) -> int:
        if not self.on:
            return 0
        cut = (cv2.subtract(before, alpha) > 32).astype(np.uint8)   # saturating, no int16 copies
        x, y, w, h = cv2.boundingRect(cut)
        if w == 0 or h == 0:
            return 0
        self._prepare()
        sl = (slice(y, y + h), slice(x, x + w))
        count, labels = cv2.connectedComponents(cut[sl], connectivity=8)
        m_model, m_dist, m_fine, m_stone, m_like = _component_means(
            labels, labels > 0, count,
            (self.model[sl], self.dist[sl], self.fine[sl], self.stone[sl], self.backdrop_like[sl]))
        m_near = (_component_means(labels, labels > 0, count, (self.near[sl],))[0]
                  if not self.clipped else None)
        cfg = self.cfg
        exact = self._exact({"dist": m_dist, "fine": m_fine})
        if self.clipped:
            confident = m_model >= float(cfg.get("white_guard_model_alpha", 220))
            put_back = ~exact & ((m_stone >= 0.5) | confident)
        else:
            # grey backdrop: the "brighter than backdrop" test still works for
            # sparkle, so only stones are protected (not every confident cut)
            exact |= (m_like >= float(cfg.get("light_guard_exact_share", 0.5))) | \
                     (m_near >= float(cfg.get("light_opening_near_share", 0.65)))
            put_back = ~exact & (m_stone >= 0.5)
        put_back[0] = False
        if not put_back.any():
            return 0
        mask = put_back[labels]
        a, b = alpha[sl], before[sl]
        a[mask] = b[mask]
        n = int(np.count_nonzero(mask))
        self.restored += n
        return n

    def fill_stone_holes(self, alpha: np.ndarray) -> int:
        """Close see-through holes inside diamonds, whoever made them.

        A hole is closed when it is enclosed by product, lies in a stone zone
        and is not plain backdrop (a real opening stays open).
        """
        if not self.on or not self.cfg.get("fill_stone_holes", True):
            return 0
        holes = (alpha < 128).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
        if count <= 1:
            return 0
        cap = float(self.cfg.get("stone_hole_max_ratio", 0.0004)) * alpha.size
        cand = stats[:, cv2.CC_STAT_AREA] <= cap
        cand[np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))] = False
        cand[0] = False
        if not cand.any():
            return 0
        self._prepare()
        m_stone, m_dist, m_fine, m_like, m_near = _component_means(
            labels, cand[labels], count,
            (self.stone, self.dist, self.fine, self.backdrop_like, self.near))
        # mostly backdrop-like or backdrop-coloured = a real opening
        exact = self._exact({"dist": m_dist, "fine": m_fine})
        exact |= m_like >= float(self.cfg.get("light_guard_exact_share", 0.5))
        if not self.clipped:
            exact |= m_near >= float(self.cfg.get("light_opening_near_share", 0.65))
        fill = cand & (m_stone >= 0.6) & ~exact
        if not fill.any():
            return 0
        mask = fill[labels]
        alpha[mask] = 255
        n = int(np.count_nonzero(mask))
        self.restored += n
        return n

    def run(self, fn, alpha, *args, **kwargs):
        """Run one clean-up pass with the guard around it."""
        before = alpha.copy() if self.on else None
        result = fn(*args, **kwargs)
        if self.on:
            self(before, alpha)
        return result

def clean_matte(source_rgb: np.ndarray, alpha: np.ndarray, settings: dict | None = None,
                background: np.ndarray | None = None, reference_size: tuple | None = None):
    """Run every clean-up pass. Returns (alpha, stats, context).

    ``background`` (precomputed backdrop colour for this region) and
    ``reference_size`` (h, w of the full photo) let callers run the passes on
    a crop around the product only, which is much faster on large photos.
    """
    cfg = _cfg(settings)
    max_px = float(cfg.get("full_res_cleanup_max_px", 12_000_000))
    h, w = alpha.shape
    if h * w > max_px:
        return _clean_matte_large(source_rgb, alpha, settings, cfg, background, reference_size, max_px)
    alpha = alpha.copy()
    model_alpha = alpha.copy()
    ctx = MatteContext(source_rgb, alpha, cfg, background, reference_size)
    shadows = shadow_mask(ctx, alpha)
    guard = _ClippedBackdropGuard(ctx.src, ctx.bg, model_alpha, cfg, ctx.diag)
    g = guard.run
    stats = {
        "edge_refined_px": refine_edges(ctx, alpha),
        "border_background_px": remove_border_background(ctx, alpha),
        "shadow_px": remove_shadows(ctx, alpha, shadows),
        "recovered_detail_px": recover_thin_details(ctx, alpha, shadows),
        "hole_px": g(remove_enclosed_background, alpha, ctx, alpha),
        "contour_hole_px": g(remove_hierarchy_holes, alpha, ctx, alpha),
        "pocket_px": g(open_metal_pockets, alpha, ctx, alpha),
        "hollow_px": g(open_backdrop_hollows, alpha, ctx, alpha),
        "thin_gap_px": g(remove_thin_gaps, alpha, ctx, alpha),
        "speck_px": g(remove_specks, alpha, ctx, alpha),
    }
    stats["restored_reflection_px"] = restore_stone_reflections(ctx, model_alpha, alpha)
    stats["solid_stone_px"] = solidify_stones(ctx, alpha)
    stats["haze_px"] = g(clear_backdrop_haze, alpha, ctx.src, ctx.bg, alpha, cfg)
    stats["flat_backdrop_px"] = g(clear_flat_backdrop, alpha, ctx.src, ctx.bg, alpha, cfg)
    stats["wedge_px"] = g(clear_backdrop_wedges, alpha, ctx.src, ctx.bg, alpha, cfg)
    stats["finished_edge_px"] = g(finish_edges, alpha, ctx.src, ctx.bg, alpha, cfg, ctx.diag, model_alpha)
    stats["stone_hole_px"] = guard.fill_stone_holes(alpha)
    stats["white_backdrop_guard_px"] = guard.restored
    stats["white_pocket_px"] = clear_white_pockets(ctx.src, ctx.bg, alpha, cfg, ctx.diag)
    _release_features()
    return alpha, stats, ctx


def _clean_matte_large(source_rgb, alpha, settings, cfg, background, reference_size, max_px):
    """Clean-up for very large crops (e.g. a necklace filling a 30 MP photo).

    Decisions (hollows, shadows, specks, pockets, stones) are made on a
    reduced copy - those features are many pixels wide even there - and
    transferred to the full-resolution matte; only edge refinement runs at
    native resolution. Several times faster, same result where it matters.
    """
    h, w = alpha.shape
    scale = min(1.0, (max_px / float(h * w)) ** 0.5)
    sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    small_src = cv2.resize(source_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    small_in = cv2.resize(alpha, (sw, sh), interpolation=cv2.INTER_AREA)
    small_bg = None if background is None else cv2.resize(background, (sw, sh), interpolation=cv2.INTER_AREA)
    ref_h, ref_w = reference_size or (h, w)
    small_ref = (max(1, int(round(ref_h * scale))), max(1, int(round(ref_w * scale))))
    small_cfg = dict(settings or {})
    small_cfg["refine_edges"] = False          # done at native resolution below
    # never take the large path again on the reduced copy (rounding can leave
    # it a few pixels over the limit, which used to recurse until out of memory)
    small_cfg["full_res_cleanup_max_px"] = float("inf")
    small_out, stats, small_ctx = clean_matte(small_src, small_in, small_cfg, small_bg, small_ref)

    # transfer the decisions
    delta = small_out.astype(np.int16) - small_in.astype(np.int16)
    cleared = cv2.resize((delta < -64).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR) > 0
    raised = cv2.resize((delta > 64).astype(np.uint8) * (small_out >= 250), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    out = alpha.copy()
    out[cleared] = 0
    out[raised & ~cleared] = 255

    bg_full = cv2.resize(small_ctx.bg, (w, h), interpolation=cv2.INTER_LINEAR)
    restored_mask = None
    if cfg["protect_stone_reflections"]:
        open_small = getattr(small_ctx, "open_backdrop", None)
        open_full = None
        if open_small is not None and open_small.any():
            open_full = cv2.resize(open_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        restored_mask = _restore_reflections_fullres(
            source_rgb, bg_full, alpha, out, cfg, float(np.hypot(ref_h, ref_w)), open_full)
        stats["restored_reflection_px"] = stats.get("restored_reflection_px", 0) + int(restored_mask.sum())
        cleared &= ~restored_mask
    if cfg["refine_edges"]:
        stats["edge_refined_px"] = refine_edges_core(source_rgb, out, bg_full, cfg, float(np.hypot(ref_h, ref_w)))
    # keep cleared regions clear after edge refinement
    out[cleared & (out < 128)] = 0
    if restored_mask is not None and restored_mask.any():
        # stone facets: edge refinement must not thin them out again
        out[restored_mask] = np.maximum(out[restored_mask], alpha[restored_mask])

    guard = _ClippedBackdropGuard(source_rgb, bg_full, alpha, cfg, float(np.hypot(ref_h, ref_w)))
    g = guard.run
    stats["haze_px"] = g(clear_backdrop_haze, out, source_rgb, bg_full, out, cfg)
    stats["flat_backdrop_px"] = g(clear_flat_backdrop, out, source_rgb, bg_full, out, cfg)
    stats["wedge_px"] = g(clear_backdrop_wedges, out, source_rgb, bg_full, out, cfg)
    stats["finished_edge_px"] = g(finish_edges, out, source_rgb, bg_full, out, cfg, float(np.hypot(ref_h, ref_w)), alpha)
    stats["stone_hole_px"] = stats.get("stone_hole_px", 0) + guard.fill_stone_holes(out)
    stats["white_backdrop_guard_px"] = stats.get("white_backdrop_guard_px", 0) + guard.restored
    stats["white_pocket_px"] = clear_white_pockets(source_rgb, bg_full, out, cfg, float(np.hypot(ref_h, ref_w)))
    _release_features()

    class _FullCtx:
        """What callers use from the context, at full resolution."""
    ctx = _FullCtx()
    ctx.bg = bg_full
    ctx.gray = cv2.cvtColor(source_rgb.astype(np.float32), cv2.COLOR_RGB2GRAY)
    ctx.diag = float(np.hypot(ref_h, ref_w))
    stats["reduced_scale"] = round(scale, 3)
    return out, stats, ctx


def _stone_zone(src, bg, stone_d):
    """Pixels within a stone's diameter of both dark facets and a sparkle."""
    diff = src.astype(np.float32) - bg
    lum = diff.mean(axis=2)
    tot = src.astype(np.float32).sum(axis=2) + 1
    btot = bg.sum(axis=2) + 1
    chroma = np.abs(src / tot[..., None] - bg / btot[..., None]).max(axis=2)
    dark = ((lum < -45) & (chroma < 0.05)).astype(np.uint8)
    sparkle = (lum > 30).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(stone_d) + 1, 2 * int(stone_d) + 1))
    return (cv2.dilate(dark, k) & cv2.dilate(sparkle, k)).astype(bool)


def _restore_reflections_fullres(src, bg, before, out, cfg, diag, open_mask=None) -> int:
    """restore_stone_reflections for the large-photo path, at full
    resolution, computing colour statistics only inside each cut region."""
    restored_mask = np.zeros(out.shape, bool)
    stone_d = float(cfg["stone_size_ratio"]) * diag
    low = int(cfg.get("reflection_low_alpha", 90))
    cut = (before >= low) & (out <= 127)
    if open_mask is not None:
        cut &= ~open_mask
    cut = cut.astype(np.uint8)
    if not cut.any():
        return restored_mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cut, connectivity=8)
    conf = before >= int(cfg["reflection_model_alpha"])
    stone_area = 0.785 * stone_d * stone_d
    min_d = float(cfg["reflection_min_dist"])
    max_lum = float(cfg["reflection_max_lum"])
    restored = 0
    for idx in range(1, count):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area < 6:
            continue
        x, y, w, h = stats[idx, :4]
        comp = labels[y:y + h, x:x + w] == idx
        diff = src[y:y + h, x:x + w][comp].astype(np.float32) - bg[y:y + h, x:x + w][comp]
        d_med = float(np.median(np.sqrt((diff * diff).sum(axis=1))))
        lum_med = float(np.median(diff.mean(axis=1)))
        small = area < 0.5 * stone_area
        if lum_med < max_lum and (d_med > min_d or (small and d_med > 0.6 * min_d)):
            if conf[y:y + h, x:x + w][comp].mean() < 0.5:
                # model was unsure: only inside a stone (dark facets + sparkle)
                pad = int(stone_d) + 2
                Y0, Y1 = max(0, y - pad), min(out.shape[0], y + h + pad)
                X0, X1 = max(0, x - pad), min(out.shape[1], x + w + pad)
                zone = _stone_zone(src[Y0:Y1, X0:X1], bg[Y0:Y1, X0:X1], stone_d)
                if zone[y - Y0:y - Y0 + h, x - X0:x - X0 + w][comp].mean() < 0.6:
                    continue
            region = out[y:y + h, x:x + w]
            # restore to at least a solid level: these are stone facets
            region[comp] = np.maximum(before[y:y + h, x:x + w][comp], 200)
            restored_mask[y:y + h, x:x + w] |= comp
    return restored_mask