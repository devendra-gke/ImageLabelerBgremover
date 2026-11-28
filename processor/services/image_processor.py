from __future__ import annotations

import base64
import json
import os
import re
import ssl
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
import time

import cv2
import os
from pymatting import estimate_foreground_ml

from .jewellery_detection import (
    CutoutResult,
    cutout_jewellery,
    crop_to_product,
    draw_detection_overlay,
)

ssl._create_default_https_context = ssl._create_unverified_context

# ======================================================
# CONFIGURATION
# ======================================================

HF_TOKEN = os.getenv("HF_TOKEN")
HF_SEGMENTATION_MODEL = os.getenv(
    "HF_SEGMENTATION_MODEL",
    "briaai/RMBG-1.4",
)

BACKGROUND_REMOVAL_BACKEND = os.getenv(
    "BACKGROUND_REMOVAL_BACKEND",
    "local",
).lower()

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
WINDOWS_TREBUCHET = r"C:\Windows\Fonts\trebuc.ttf"

TARGET_WIDTH = 1750
TARGET_HEIGHT = 1160

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRESET_FILES = {
    "diamond": "diamond_image_settings.json",
    "diamond_image": "diamond_image_settings.json",
    "jewellery": "jewellery_image_settings.json",
    "jewelry": "jewellery_image_settings.json",
    "necklace": "jewellery_image_settings.json",
    "haram": "jewellery_image_settings.json",
    "jewellery_image": "jewellery_image_settings.json",
}

# The segmentation model (default Bria RMBG 2.0, set "model" in the preset's
# background_removal section or BG_REMOVAL_MODEL env var) is loaded lazily
# once and shared - see services/model_sessions.py.


# ======================================================
# HELPERS
# ======================================================


from io import BytesIO
def remove_background_huggingface(image: Image.Image) -> Image.Image:
    """
    Use a Hugging Face image-segmentation model to generate
    an alpha mask. The original jewellery RGB pixels are retained.

    HF_SEGMENTATION_MODEL must identify a compatible segmentation
    model, not an image-generation model such as Qwen-Image.
    """
    from huggingface_hub import InferenceClient

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not configured.")

    if not HF_SEGMENTATION_MODEL:
        raise RuntimeError(
            "HF_SEGMENTATION_MODEL is not configured."
        )

    client = InferenceClient(
        model=HF_SEGMENTATION_MODEL,
        token=HF_TOKEN,
        timeout=120,
    )

    source = image.convert("RGB")
    predictions = client.image_segmentation(source)

    if not predictions:
        raise RuntimeError(
            "The segmentation model returned no masks."
        )

    candidates = []

    for prediction in predictions:
        if isinstance(prediction, dict):
            label = str(prediction.get("label", "")).lower()
            mask = prediction.get("mask")
            score = float(prediction.get("score", 0.0))
        else:
            label = str(
                getattr(prediction, "label", "")
            ).lower()
            mask = getattr(prediction, "mask", None)
            score = float(
                getattr(prediction, "score", 0.0)
            )

        if mask is None:
            continue

        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(np.asarray(mask))

        mask = mask.convert("L").resize(
            source.size,
            Image.Resampling.LANCZOS,
        )

        # This helper expects a foreground mask.
        # Do not silently invert masks without checking the model's
        # documented label semantics.
        if label in {"background", "bg"}:
            mask = Image.eval(mask, lambda value: 255 - value)
        elif label in {"", "foreground", "object", "product"}:
            pass
        else:
            # Unknown labels may describe other objects or classes.
            continue

        candidates.append((score, mask))

    if not candidates:
        raise RuntimeError(
            "No usable foreground mask was returned. "
            "Check the model's output labels and documentation."
        )

    _, foreground_mask = max(
        candidates,
        key=lambda item: item[0],
    )

    rgba = source.convert("RGBA")
    rgba.putalpha(foreground_mask)
    return rgba



def _preserve_in_strips(crop_rgb, result, protection, foreground_rgb, backdrop,
                        strip_rows: int = 1024, overlap: int = 64):
    """Run preserve_original_product_pixels on horizontal strips.

    Every operation in it reaches at most a few pixels (highlight distance,
    facet edges), so overlapping strips give the same result while peak
    memory stays small even for 30+ MP crops (a whole-crop pass needed ~6 GB).
    """
    w, h = crop_rgb.size
    if w * h <= 8_000_000:
        return preserve_original_product_pixels(
            crop_rgb, result, protection, estimated_rgb=foreground_rgb, backdrop=backdrop)
    out = np.empty((h, w, 4), np.uint8)
    for sy in range(0, h, strip_rows):
        ey = min(h, sy + strip_rows)
        py0, py1 = max(0, sy - overlap), min(h, ey + overlap)
        part = preserve_original_product_pixels(
            crop_rgb.crop((0, py0, w, py1)),
            result.crop((0, py0, w, py1)),
            protection,
            estimated_rgb=None if foreground_rgb is None else foreground_rgb[py0:py1],
            backdrop=None if backdrop is None else backdrop[py0:py1],
        )
        out[sy:ey] = np.asarray(part, dtype=np.uint8)[sy - py0:ey - py0]
    return Image.fromarray(out, mode="RGBA")


def decontaminate_edges(rgba: np.ndarray, background_small: np.ndarray) -> int:
    """Remove the backdrop colour mixed into semi-transparent edge pixels.

    An edge pixel with alpha a shows a*product + (1-a)*backdrop. With the
    local backdrop colour B known, the product colour is (I - (1-a)B) / a.
    Without this, cutouts get a light/grey glow on dark pages.
    """
    a = rgba[..., 3]
    edge = (a > 0) & (a < 250)
    if not edge.any():
        return 0
    ys, xs = np.nonzero(edge)
    h, w = a.shape
    bg = cv2.resize(background_small, (w, h), interpolation=cv2.INTER_LINEAR)[ys, xs]
    af = (a[ys, xs].astype(np.float32) / 255.0)[:, None]
    src = rgba[ys, xs, :3].astype(np.float32)
    fg = (src - (1.0 - af) * bg) / np.maximum(af, 0.15)
    rgba[ys, xs, :3] = np.clip(fg, 0, 255).astype(np.uint8)
    return int(len(ys))


def _distance_to(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance of every pixel to the nearest True pixel in mask.

    Same result as scipy's distance_transform_edt(~mask) but several times
    faster (OpenCV, exact mask). Pixels are infinitely far when mask is empty.
    """
    if not mask.any():
        return np.full(mask.shape, np.inf, np.float32)
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def preserve_original_product_pixels(
    original: Image.Image,
    processed: Image.Image,
    product_protection: dict | None = None,
    estimated_rgb: np.ndarray | None = None,
    backdrop: np.ndarray | None = None,
) -> Image.Image:
    """
    Preserve original jewellery colours and shine.
    Keep source RGB and model alpha, with optional conservative edge cleanup.

    ``backdrop`` (local backdrop colour per pixel) makes sparkle/facet
    recovery relative to the backdrop: on a light studio backdrop (~210+),
    fixed brightness thresholds would otherwise treat the backdrop next to
    the product - and the camera's sharpening halo - as diamond sparkle and
    paint a light outline around the piece.
    """
    product_protection = product_protection or {}

    source = original.convert("RGB")
    result = processed.convert("RGBA")

    if result.size != source.size:
        result = result.resize(
            source.size,
            Image.Resampling.LANCZOS
        )

    src = np.asarray(source, dtype=np.float32)
    rgba = np.asarray(result, dtype=np.float32).copy()

    alpha_u8 = rgba[..., 3].astype(np.uint8)
    initial_alpha_u8 = alpha_u8.copy()
    alpha = alpha_u8 / 255.0

    # Pixels that clearly stand out from the local backdrop. Without a
    # backdrop estimate every pixel is eligible (previous behaviour).
    # brightest channel per pixel, computed once (OpenCV per-channel max is
    # much faster than a numpy reduction over the colour axis)
    s0, s1, s2 = cv2.split(src)
    src_max = cv2.max(cv2.max(s0, s1), s2)
    if backdrop is not None:
        bk = np.ascontiguousarray(backdrop, dtype=np.float32)
        # RGB2GRAY uses the same 0.299/0.587/0.114 weights as before
        luma_src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        luma_bg = cv2.cvtColor(bk, cv2.COLOR_RGB2GRAY)
        brighter_than_backdrop = (luma_src - luma_bg) >= float(
            product_protection.get("sparkle_backdrop_margin", 25))
        k0, k1, k2 = cv2.split(bk)
        d0, d1, d2 = s0 - k0, s1 - k1, s2 - k2
        differs_from_backdrop = cv2.sqrt(d0 * d0 + d1 * d1 + d2 * d2) >= float(
            product_protection.get("facet_backdrop_contrast", 30))
    else:
        brighter_than_backdrop = np.ones(alpha_u8.shape, bool)
        differs_from_backdrop = brighter_than_backdrop

    # Use the PyMatting estimate when supplied. This keeps decontaminated
    # edge colour while the recovery logic below protects fine product detail.
    # For reflective products, PyMatting can soften bright facets and glow.
    # Keep the camera pixels unless a preset explicitly opts into estimated
    # foreground RGB. The alpha mask is still used for the cutout.
    # Diamond colour must come from the original camera pixels. Estimated
    # foreground RGB is useful for contaminated matte edges, but it can
    # change the stone's size/shape appearance and wash out its facets.
    preserve_source_rgb = bool(
        product_protection.get("preserve_source_rgb", True)
    )
    use_estimated_rgb = bool(
        product_protection.get("use_estimated_foreground_rgb", False)
    ) and not preserve_source_rgb
    output_rgb = (
        np.asarray(estimated_rgb, dtype=np.float32).copy()
        if use_estimated_rgb and estimated_rgb is not None
        else src.copy()
    )
    if output_rgb.shape != src.shape:
        output_rgb = src.copy()

    edge_recovery = bool(product_protection.get("edge_recovery", False))

    if edge_recovery:
        height, width = alpha.shape
        border_size = max(5, min(height, width) // 100)

        border_pixels = np.concatenate([
            src[:border_size, :, :].reshape(-1, 3),
            src[-border_size:, :, :].reshape(-1, 3),
            src[:, :border_size, :].reshape(-1, 3),
            src[:, -border_size:, :].reshape(-1, 3),
        ])

        background_rgb = np.median(border_pixels, axis=0)

        # Focus only on uncertain transition pixels.
        edge = (alpha >= 0.35) & (alpha < 0.98)

        if np.any(edge):
            # Protect bright highlight pixels to preserve diamond sparkle.
            highlight_threshold = float(
                product_protection.get("highlight_threshold", 228)
            )
            highlight_mask = (src_max >= highlight_threshold) & brighter_than_backdrop
            eligible = edge & (~highlight_mask)

            if np.any(eligible):
                a = np.maximum(alpha[eligible, None], 0.35)
                corrected = (
                    src[eligible]
                    - (1.0 - a) * background_rgb[None, :]
                ) / a

                corrected = np.clip(corrected, 0, 255)

                # Keep edge cleanup mild so gem colors are not flattened.
                max_edge_cleanup = float(
                    product_protection.get("max_edge_cleanup", 0.18)
                )
                max_edge_cleanup = min(max(max_edge_cleanup, 0.0), 0.35)

                correction_strength = (
                    (1.0 - alpha[eligible]) * max_edge_cleanup
                )[:, None]

                output_rgb[eligible] = (
                    src[eligible] * (1.0 - correction_strength)
                    + corrected * correction_strength
                )

    alpha_floor = int(product_protection.get("alpha_floor", 0) or 0)
    alpha_floor = min(max(alpha_floor, 0), 255)

    if alpha_floor > 0:
        nonzero = alpha_u8 > 0
        alpha_u8[nonzero] = np.maximum(alpha_u8[nonzero], alpha_floor)

    alpha_expand_px = int(product_protection.get("alpha_expand_px", 1))
    if product_protection.get("preserve_diamond_shape", False):
        alpha_expand_px = 0
    alpha_expand_px = min(max(alpha_expand_px, 0), 4)

    if alpha_expand_px > 0:
        kernel = (alpha_expand_px * 2) + 1
        # square max filter == dilation with a square kernel (OpenCV, faster)
        expanded_alpha = cv2.dilate(alpha_u8, np.ones((kernel, kernel), np.uint8))
    else:
        expanded_alpha = alpha_u8.copy()

    # Recover tiny bright facets and specular highlights that can become
    # overly transparent in segmentation masks.
    if product_protection.get("preserve_diamond_color", True):
        highlight_alpha_recovery = bool(
            product_protection.get("highlight_alpha_recovery", True)
        )

        if highlight_alpha_recovery:
            highlight_threshold = int(
                product_protection.get("highlight_threshold", 228)
            )
            sparkle_min_alpha = int(
                product_protection.get("sparkle_min_alpha", 205)
            )
            sparkle_min_alpha = min(max(sparkle_min_alpha, 0), 255)

            # Protect only bright pixels close to the nearest existing
            # product border. This preserves diamond shine while preventing
            # white studio background from being pulled into the cutout.
            sparkle = (src_max >= highlight_threshold) & brighter_than_backdrop
            uncertain = alpha_u8 < 245
            alpha_seed = int(
                product_protection.get("highlight_alpha_seed", 8)
            )
            alpha_seed = min(max(alpha_seed, 1), 255)
            foreground_seed = initial_alpha_u8 >= alpha_seed
            nearest_border_distance = _distance_to(foreground_seed)
            max_highlight_distance = max(
                float(
                    product_protection.get(
                        "max_highlight_distance_px", 6
                    )
                ),
                0.0,
            )
            near_product_border = (
                nearest_border_distance <= max_highlight_distance
            )
            keep_mask = (
                sparkle
                & uncertain
                & near_product_border
                & (expanded_alpha > 0)
            )

            # Only the actual highlight pixel is raised. The source RGB is
            # retained below, so its original diamond colour remains sharp.
            boosted = np.full_like(alpha_u8, sparkle_min_alpha)

            alpha_u8[keep_mask] = np.maximum(
                alpha_u8[keep_mask],
                boosted[keep_mask],
            )

            # Recover zero-alpha sparkle pixels only when close to the
            # original matte, preventing distant background bleed.
            if product_protection.get("recover_zero_alpha_sparkle", True):
                zero_seed_alpha = int(
                    product_protection.get("zero_alpha_seed", 28)
                )
                zero_seed_alpha = min(max(zero_seed_alpha, 1), 255)
                recover_zero = (
                    sparkle
                    & (initial_alpha_u8 == 0)
                    & (expanded_alpha >= zero_seed_alpha)
                    & near_product_border
                )
                alpha_u8[recover_zero] = np.maximum(
                    alpha_u8[recover_zero],
                    boosted[recover_zero],
                )

    # Keep the product body fully opaque. Segmentation mattes often leave
    # the diamond interior slightly translucent (e.g. alpha 180-240), which
    # makes white shine blend into the canvas and look dull. Pixels already
    # confidently inside the product are raised to solid; the soft edge
    # transition below the threshold is left untouched.
    solid_core_threshold = int(
        product_protection.get("solid_core_threshold", 0) or 0
    )
    if solid_core_threshold > 0:
        solid_core_threshold = min(max(solid_core_threshold, 1), 254)
        solid_core_alpha = int(
            product_protection.get("solid_core_alpha", 255) or 255
        )
        solid_core_alpha = min(max(solid_core_alpha, 0), 255)
        core = alpha_u8 >= solid_core_threshold
        alpha_u8[core] = np.maximum(alpha_u8[core], solid_core_alpha)

    # Remove leftover background haze. Segmentation often leaves a soft,
    # low-alpha white blur around the product (the defocused studio
    # backdrop). Any faint pixel that is NOT close to the solid product
    # body is background residue and is cut to zero. Bright sparkle pixels
    # near the body are protected so real diamond shine is never removed.
    haze_cutoff = int(
        product_protection.get("haze_cutoff_alpha", 0) or 0
    )
    if haze_cutoff > 0:
        haze_cutoff = min(max(haze_cutoff, 1), 254)
        body_seed = int(
            product_protection.get("haze_body_seed", 200) or 200
        )
        body_seed = min(max(body_seed, 1), 255)
        max_haze_distance = max(
            float(product_protection.get("haze_max_distance_px", 3) or 3),
            0.0,
        )
        haze_highlight_threshold = int(
            product_protection.get(
                "haze_highlight_threshold",
                product_protection.get("highlight_threshold", 195),
            )
        )

        distance_to_body = _distance_to(initial_alpha_u8 >= body_seed)
        near_body = distance_to_body <= max_haze_distance
        bright_sparkle = (src_max >= haze_highlight_threshold) & brighter_than_backdrop

        haze = (
            (alpha_u8 > 0)
            & (alpha_u8 < haze_cutoff)
            & (~(near_body & bright_sparkle))
        )
        alpha_u8[haze] = 0

    if (
        product_protection.get("preserve_facet_boundaries", True)
        and not product_protection.get("preserve_diamond_shape", False)
    ):
        # Preserve thin prongs and facet edges that segmentation can fade.
        gray = (
            (0.299 * src[..., 0])
            + (0.587 * src[..., 1])
            + (0.114 * src[..., 2])
        ).astype(np.uint8)

        edge_map = np.asarray(
            Image.fromarray(gray, mode="L").filter(ImageFilter.FIND_EDGES),
            dtype=np.uint8,
        )

        facet_edge_threshold = int(
            product_protection.get("facet_edge_threshold", 20)
        )
        facet_brightness_threshold = int(
            product_protection.get("facet_brightness_threshold", 150)
        )
        facet_min_alpha = int(
            product_protection.get("facet_min_alpha", 185)
        )

        facet_edge_threshold = min(max(facet_edge_threshold, 1), 255)
        facet_brightness_threshold = min(
            max(facet_brightness_threshold, 0),
            255,
        )
        facet_min_alpha = min(max(facet_min_alpha, 0), 255)

        uncertain = alpha_u8 < facet_min_alpha
        facet_mask = (
            uncertain
            & (edge_map >= facet_edge_threshold)
            & (gray >= facet_brightness_threshold)
            & (expanded_alpha > 0)
            & differs_from_backdrop
        )

        alpha_u8[facet_mask] = np.maximum(
            alpha_u8[facet_mask],
            np.minimum(expanded_alpha[facet_mask], facet_min_alpha),
        )

    # Remove white matte contamination from the visible edge.  The source
    # RGB is normally retained to protect jewellery colour, but semi-
    # transparent edge pixels can still contain the original white backdrop.
    # Correct only neutral, bright fringe pixels; diamond sparkle and coloured
    # stones are left untouched.
    if product_protection.get("remove_white_edge", False):
        edge_alpha = (alpha_u8 > 0) & (alpha_u8 < 245)
        edge_rgb = output_rgb.astype(np.float32)
        max_channel = np.max(edge_rgb, axis=2)
        min_channel = np.min(edge_rgb, axis=2)
        neutral = (max_channel - min_channel) <= float(
            product_protection.get("white_edge_neutral_tolerance", 18)
        )
        bright = max_channel >= float(
            product_protection.get("white_edge_brightness", 185)
        )
        fringe = edge_alpha & neutral & bright

        if np.any(fringe):
            border_size = max(5, min(src.shape[:2]) // 100)
            border_pixels = np.concatenate([
                src[:border_size, :, :].reshape(-1, 3),
                src[-border_size:, :, :].reshape(-1, 3),
                src[:, :border_size, :].reshape(-1, 3),
                src[:, -border_size:, :].reshape(-1, 3),
            ])
            background_rgb = np.median(border_pixels, axis=0)
            a = np.maximum(alpha_u8[fringe, None] / 255.0, 0.08)
            corrected = (src[fringe] - (1.0 - a) * background_rgb) / a
            output_rgb[fringe] = np.clip(corrected, 0, 255)

    # Preserve the model's alpha mask; do not hard-threshold it.
    output = np.dstack([
        np.clip(output_rgb, 0, 255).astype(np.uint8),
        alpha_u8
    ])

    return Image.fromarray(output, mode="RGBA")


def apply_reference_matte(
    original: Image.Image,
    matte_path: str | os.PathLike,
    use_reference_rgb: bool = False,
    repair_regions: list | None = None,
) -> Image.Image:
    """Apply a reference cutout and optionally recover source details in ROIs."""
    source = original.convert("RGB")
    reference_path = Path(matte_path)

    if not reference_path.is_absolute():
        reference_path = PROJECT_ROOT / reference_path

    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Reference matte not found: {reference_path}"
        )

    with Image.open(reference_path) as reference_image:
        reference = reference_image.convert("RGBA")
        source_ratio = source.width / source.height
        reference_ratio = reference.width / reference.height

        if abs(source_ratio - reference_ratio) > 0.02:
            raise ValueError(
                "Reference matte aspect ratio does not match the source image."
            )

        reference = reference.resize(source.size, Image.Resampling.LANCZOS)

    if use_reference_rgb:
        result = reference.copy()
    else:
        result = source.convert("RGBA")
        result.putalpha(reference.getchannel("A"))

    if repair_regions:
        source_rgb = np.asarray(source, dtype=np.float32)
        output = np.asarray(result, dtype=np.uint8).copy()
        height, width = source_rgb.shape[:2]

        for region in repair_regions:
            if not isinstance(region, (list, tuple)) or len(region) != 4:
                continue

            left, top, right, bottom = (float(value) for value in region)
            x1 = min(max(round(left * width), 0), width)
            y1 = min(max(round(top * height), 0), height)
            x2 = min(max(round(right * width), x1), width)
            y2 = min(max(round(bottom * height), y1), height)
            if x1 >= x2 or y1 >= y2:
                continue

            crop = source_rgb[y1:y2, x1:x2]

            # The source photos use a near-white background. Build a local
            # soft matte and restrict it to the explicitly marked ROIs.
            deviation = 255.0 - np.min(crop, axis=2)
            low = 5.0
            high = 38.0
            recovered_alpha = np.clip(
                (deviation - low) * (255.0 / (high - low)),
                0,
                255,
            ).astype(np.uint8)

            existing_alpha = output[y1:y2, x1:x2, 3]
            restore = recovered_alpha > existing_alpha
            if not np.any(restore):
                continue

            alpha = np.maximum(existing_alpha, recovered_alpha).astype(np.float32)
            alpha_fraction = np.maximum(alpha / 255.0, 0.12)[..., None]

            # Remove the source's white backdrop from recovered edge colors.
            decontaminated = (
                crop - (1.0 - alpha_fraction) * 255.0
            ) / alpha_fraction
            decontaminated = np.clip(decontaminated, 0, 255).astype(np.uint8)

            rgb_roi = output[y1:y2, x1:x2, :3]
            rgb_roi[restore] = decontaminated[restore]
            output[y1:y2, x1:x2, 3] = np.maximum(
                existing_alpha,
                recovered_alpha,
            )

        result = Image.fromarray(output, mode="RGBA")

    return result




def load_preset_settings(preset_name: str | None = None, base_dir: str | os.PathLike | None = None):
    """Load a preset JSON from the project folder and return its dictionary."""
    if preset_name is None:
        preset_name = "jewellery"

    normalized = str(preset_name).strip().lower()
    file_name = PRESET_FILES.get(normalized)

    if file_name is None:
        file_name = str(preset_name)

    base_path = Path(base_dir) if base_dir else PROJECT_ROOT
    config_path = Path(base_path) / file_name

    if not config_path.exists():
        fallback = Path(base_path) / "jewellery_image_settings.json"
        if fallback.exists():
            config_path = fallback

    if not config_path.exists():
        raise FileNotFoundError(f"Preset settings file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def resolve_preset_name(request=None, default_name: str = "jewellery", filename: str | None = None):
    """Resolve the preset name from request payload or the image filename."""
    if request is not None:
        post = getattr(request, "POST", {})
        for key in ("preset", "image_type", "product_type", "preset_name"):
            value = post.get(key)
            if value:
                return str(value).strip().lower()

    if filename:
        lowered = filename.lower()
        if "diamond" in lowered:
            return "diamond"
        if any(term in lowered for term in ("jewel", "necklace", "haram")):
            return "jewellery"

    return default_name


# def preserve_original_product_pixels(original: Image.Image, processed: Image.Image) -> Image.Image:
#     """Use the AI result only as an alpha mask and retain source RGB values.

#     Do not restore bright pixels outside the mask: those can be background
#     pixels and create white/grey halos around diamonds and fine gold wires.
#     """
#     source = original.convert("RGB")
#     result = processed.convert("RGBA")

#     if result.size != source.size:
#         result = result.resize(source.size, Image.Resampling.LANCZOS)

#     source_rgb = np.asarray(source, dtype=np.uint8)
#     rgba = np.asarray(result, dtype=np.uint8).copy()
#     alpha = rgba[..., 3].copy()

#     # Keep the model's original soft alpha edges. Do not hard-threshold them;
#     # tiny diamonds, prongs and thin chain links may occupy only a few pixels.
#     rgba[..., :3] = source_rgb
#     rgba[..., 3] = alpha
#     rgba[alpha == 0, :3] = 0
#     return Image.fromarray(rgba, mode="RGBA")

def base_token(s: str) -> str:
    s = (s or "").strip()
    if "_" in s:
        s = s.split("_", 1)[0]
    # Image names commonly contain variants such as "K-05684 P (3)".
    # Match the product tag only so per-image protection settings apply to
    # every view/copy of the same item.
    match = re.match(r"^([a-z]+-\d+|\d+)", s, flags=re.IGNORECASE)
    return (match.group(1) if match else s).lower()


def get_series(df, col_letter):
    col_letter = col_letter.upper()

    if col_letter in df.columns:
        return df[col_letter]

    idx = ord(col_letter) - ord("A")
    return df.iloc[:, idx]


_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def safe_filename(name, fallback: str = "image", max_len: int = 150) -> str:
    """A file/folder name that Windows, macOS and Linux all accept.

    Uploaded names can carry characters that are invisible in the UI but
    rejected by Windows with "OSError: [Errno 22] Invalid argument": control
    characters (tab, newline), zero-width / non-breaking spaces, < > : " | ? *
    or a trailing dot/space. These are replaced; readable characters
    (letters in any script, digits, spaces, - _ ( ) etc.) are kept.
    """
    import unicodedata

    text = unicodedata.normalize("NFKC", str(name))
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in '<>:"/\\|?*' or cat in ("Cc", "Cf", "Cs", "Co", "Cn"):
            out.append("_")                 # forbidden or invisible character
        elif cat.startswith("Z"):
            out.append(" ")                 # any kind of space -> plain space
        else:
            out.append(ch)
    text = re.sub(r" {2,}", " ", "".join(out)).strip().rstrip(". ")
    if not text:
        text = fallback
    if text.split(".")[0].upper() in _WINDOWS_RESERVED:
        text = "_" + text
    return text[:max_len].rstrip(". ") or fallback


# ----------------------------------------------------------------------
# File properties (author / software) written into every saved image
# ----------------------------------------------------------------------
DEFAULT_IMAGE_AUTHOR = "GK -Devendra -Image BGRemove"

_EXIF_ARTIST = 0x013B      # "Artist" - Authors in most viewers
_EXIF_SOFTWARE = 0x0131
_EXIF_XP_AUTHOR = 0x9C9D   # Windows Explorer > Properties > Details > Authors


def image_metadata(preset: dict | None = None) -> dict:
    """Author / software for saved files.

    Order: preset ``"metadata": {"author": ..., "software": ...}``, then the
    IMAGE_AUTHOR / IMAGE_SOFTWARE environment variables, then the default.
    """
    meta = dict((preset or {}).get("metadata") or {})
    author = str(meta.get("author") or os.getenv("IMAGE_AUTHOR") or DEFAULT_IMAGE_AUTHOR).strip()
    software = str(meta.get("software") or os.getenv("IMAGE_SOFTWARE") or "Image BGRemove").strip()
    return {"author": author, "software": software}


def _exif_with_author(meta: dict, source_exif: bytes | None = None) -> bytes:
    """EXIF block with the author set, keeping the photo's own EXIF data."""
    exif = Image.Exif()
    if source_exif:
        try:
            exif.load(source_exif)
        except Exception:
            exif = Image.Exif()        # unreadable camera EXIF: start clean
    exif[_EXIF_ARTIST] = meta["author"]
    exif[_EXIF_SOFTWARE] = meta["software"]
    # Windows reads XP* tags as UTF-16LE bytes ending in a null character
    exif[_EXIF_XP_AUTHOR] = (meta["author"] + "\x00").encode("utf-16-le")
    return exif.tobytes()


def _xmp_packet(meta: dict) -> str:
    from xml.sax.saxutils import escape
    a, sw = escape(meta["author"]), escape(meta["software"])
    return (
        '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:xmp="http://ns.adobe.com/xap/1.0/">'
        f'<dc:creator><rdf:Seq><rdf:li>{a}</rdf:li></rdf:Seq></dc:creator>'
        f'<xmp:CreatorTool>{sw}</xmp:CreatorTool>'
        '</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>'
    )


def metadata_save_options(save_format: str, meta: dict, source_exif: bytes | None = None) -> dict:
    """Extra ``Image.save`` keyword arguments that write the author/software.

    JPEG/WEBP: EXIF (Artist, Software, Windows XPAuthor) plus XMP.
    PNG: text chunks (Author, Software), an XMP packet and an eXIf block.
    The PNG gets a fresh EXIF (no camera orientation tag), because its
    pixels are saved exactly as they are and must not be rotated by viewers.
    """
    fmt = save_format.upper()
    if fmt in ("JPEG", "JPG", "WEBP"):
        opts = {"exif": _exif_with_author(meta, source_exif)}
        if fmt != "WEBP":
            opts["xmp"] = _xmp_packet(meta).encode("utf-8")   # used by Pillow >= 11
        return opts
    if fmt == "PNG":
        from PIL.PngImagePlugin import PngInfo
        info = PngInfo()
        info.add_text("Author", meta["author"])
        info.add_text("Software", meta["software"])
        info.add_itxt("XML:com.adobe.xmp", _xmp_packet(meta))
        return {"pnginfo": info, "exif": _exif_with_author(meta)}
    return {}


def sanitize(name):
    return safe_filename(name, fallback="Others")


def image_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG", compress_level=1)
    return base64.b64encode(buffer.getvalue()).decode()


def thumbnail(img, max_side=900, crop=False):
    """Downscaled copy for previews (optionally cropped to the product)."""
    if crop and img.mode == "RGBA":
        img = crop_to_product(img)
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.Resampling.BILINEAR, reducing_gap=2.0)
    return img


def image_to_base64_jpeg(img, quality=85):
    buffer = BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode()


def media_url_for(path) -> str | None:
    """URL of a saved file when it lives under MEDIA_ROOT (served in DEBUG)."""
    try:
        from django.conf import settings

        rel = Path(path).resolve().relative_to(Path(settings.MEDIA_ROOT).resolve())
    except Exception:
        return None
    return settings.MEDIA_URL.rstrip("/") + "/" + rel.as_posix()


def hex_to_rgb(hex_color):
    if not hex_color:
        return (255, 255, 255)

    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def resolve_output_directory(directory: str | os.PathLike, default_relative: str) -> Path:
    """Resolve an output directory against the project root and create it."""
    value = str(directory or default_relative).strip()
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path.mkdir(parents=True, exist_ok=True)
    return path


# ======================================================
# EXCEL READER
# ======================================================

def normalize_excel_header(value):
    return str(value or "").strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def get_named_series(df, *candidate_names):
    if df.empty:
        return None

    normalized_map = {
        normalize_excel_header(col): col for col in df.columns
    }

    for candidate in candidate_names:
        key = normalize_excel_header(candidate)
        if key in normalized_map:
            return df[normalized_map[key]]

    return None


_EXCEL_CACHE: dict[str, tuple] = {}


def read_excel_mapping_cached(excel_file):
    """The page sends the same Excel with every image; parse it once."""
    import hashlib

    data = excel_file.read()
    key = hashlib.sha1(data).hexdigest()
    cached = _EXCEL_CACHE.get(key)
    if cached is None:
        cached = read_excel_mapping(BytesIO(data))
        if len(_EXCEL_CACHE) >= 8:
            _EXCEL_CACHE.pop(next(iter(_EXCEL_CACHE)))
        _EXCEL_CACHE[key] = cached
    return cached


# Catalogue sheet layout (Excel column letters, 0-based positions):
#   N -> image file name, E..K -> text printed on the image (in that order),
#   B -> category folder, E -> StyleBio folder (first 3 letters).
COL_CATEGORY = 1
COL_TEXT_FIRST, COL_TEXT_LAST = 4, 9   # E..J
COL_FILE_NAME = 13                      # N


def _cell_text(df, row, col):
    if col >= df.shape[1]:
        return ""
    val = df.iat[row, col]
    if pd.isna(val):
        return ""
    text = str(val).strip()
    return "" if text.lower() == "nan" else text


def _file_key(value):
    """'RGF-1.jpg' / 'rgf-1' -> 'rgf-1' (name without a known image extension)."""
    value = (value or "").strip()
    stem, ext = os.path.splitext(value)
    if ext.lower() in SUPPORTED_EXTS:
        value = stem
    return value.lower()


def read_excel_mapping(excel_file):
    df = pd.read_excel(excel_file, engine="openpyxl", dtype=str)
    df.columns = [str(c).strip() for c in df.columns]

    # Older sheets without a file-name column are still matched by TagNo /
    # StyleBio so they keep working.
    tag_series = get_named_series(df, "TagNo", "Tag No", "Image Tag")
    tag_idx = list(df.columns).index(tag_series.name) if tag_series is not None else None

    mapping = {}
    category_map = {}
    style_map = {}

    for i in range(len(df)):
        lines = [
            t for t in (
                _cell_text(df, i, c) for c in range(COL_TEXT_FIRST, COL_TEXT_LAST + 1)
            ) if t
        ]
        if not lines:
            continue

        category = _cell_text(df, i, COL_CATEGORY)
        style_full = _cell_text(df, i, COL_TEXT_FIRST)

        file_name = _file_key(_cell_text(df, i, COL_FILE_NAME))
        if file_name:
            keys = [file_name]
        else:
            tag = _cell_text(df, i, tag_idx) if tag_idx is not None else ""
            keys = []
            if tag and style_full:
                keys.append(f"{style_full}_{tag}".lower())
            if style_full:
                keys.append(style_full.lower())
            if tag:
                keys.append(base_token(tag))

        for key in keys:
            mapping[key] = lines
            if category:
                category_map[key] = sanitize(category)
            if style_full:
                style_map[key] = sanitize(style_full)[:3]

    return mapping, category_map, style_map


def find_excel_key(name, mapping):
    """The Excel key that belongs to an image file name (without extension)."""
    name = (name or "").strip().lower()
    candidates = [name]
    if "_" in name:
        candidates.append(name.rsplit("_", 1)[0])   # drop the view number
        candidates.append(name.split("_", 1)[0])
    candidates.append(base_token(name))
    for candidate in candidates:
        if candidate in mapping:
            return candidate
    return base_token(name)


# ======================================================
# BACKGROUND REMOVAL
# ======================================================

def estimate_foreground_color(
    source: Image.Image,
    alpha_image: Image.Image,
    settings: dict | None = None,
) -> np.ndarray:
    """Estimate decontaminated foreground RGB while keeping RGBA detail.

    PyMatting is run on a bounded-size copy to avoid making large uploads
    disproportionately slow. The returned RGB image is resized back to the
    source dimensions; the original rembg alpha is never changed. Opaque
    source pixels and bright diamond highlights are always copied back from
    the camera image so PyMatting cannot wash out the shine.
    """
    settings = settings or {}
    source_rgba = source.convert("RGBA")
    source = source_rgba.convert("RGB")
    alpha_image = alpha_image.convert("L")
    original_size = source.size

    max_size = int(settings.get("max_size", 1024) or 0)
    if max_size > 0 and max(original_size) > max_size:
        scale = max_size / max(original_size)
        estimate_size = (
            max(1, round(original_size[0] * scale)),
            max(1, round(original_size[1] * scale)),
        )
        source_for_estimate = source.resize(
            estimate_size,
            Image.Resampling.LANCZOS,
        )
        alpha_for_estimate = alpha_image.resize(
            estimate_size,
            Image.Resampling.LANCZOS,
        )
    else:
        source_for_estimate = source
        alpha_for_estimate = alpha_image

    image_array = np.asarray(source_for_estimate, dtype=np.float32) / 255.0
    alpha_array = np.asarray(alpha_for_estimate, dtype=np.float32) / 255.0

    foreground = estimate_foreground_ml(
        image_array,
        alpha_array,
        regularization=float(settings.get("regularization", 1e-5)),
        n_small_iterations=int(settings.get("small_iterations", 10)),
        n_big_iterations=int(settings.get("big_iterations", 2)),
        small_size=int(settings.get("small_size", 32)),
    )
    foreground = np.clip(foreground, 0.0, 1.0)
    foreground_image = Image.fromarray(
        np.round(foreground * 255.0).astype(np.uint8),
        mode="RGB",
    )

    if foreground_image.size != original_size:
        foreground_image = foreground_image.resize(
            original_size,
            Image.Resampling.BILINEAR,
        )

    foreground_rgb = np.asarray(foreground_image, dtype=np.float32)
    source_rgb = np.asarray(source, dtype=np.float32)
    source_alpha = np.asarray(source_rgba.getchannel("A"), dtype=np.float32) / 255.0
    model_alpha = np.asarray(alpha_image, dtype=np.float32) / 255.0
    alpha = model_alpha * source_alpha

    # Opaque pixels already contain reliable product colour. PyMatting is
    # needed mainly for the semi-transparent pixels contaminated by backdrop.
    reliable = alpha >= float(settings.get("opaque_threshold", 0.995))

    # Diamond shine is usually a near-white, high-frequency highlight. Keep
    # those original RGB values even when their matte is soft or translucent.
    highlight_threshold = float(
        settings.get("highlight_threshold", 220)
    )
    highlight = np.max(source_rgb, axis=2) >= highlight_threshold
    preserve_source = reliable | highlight
    foreground_rgb[preserve_source] = source_rgb[preserve_source]

    return np.clip(foreground_rgb, 0.0, 255.0).astype(np.uint8)

def remove_background(
    image: Image.Image,
    bg_settings: dict | None = None,
    product_protection: dict | None = None,
):
    """Backwards-compatible wrapper returning only the RGBA cutout."""
    return remove_background_detailed(image, bg_settings, product_protection)[0]


def remove_background_detailed(
    image: Image.Image,
    bg_settings: dict | None = None,
    product_protection: dict | None = None,
) -> tuple[Image.Image, CutoutResult]:
    """Detect the jewellery, cut it out and protect product colour/sparkle.

    Returns (rgba_cutout, detection_result). All colour work runs only on the
    crop around the jewellery; the rest of the photo is transparent.
    """
    source_rgba = image.convert("RGBA")
    source = source_rgba.convert("RGB")
    detection = cutout_jewellery(source, bg_settings)
    full_w, full_h = source.size
    output = np.zeros((full_h, full_w, 4), np.uint8)
    if detection.alpha_roi is None:
        return Image.fromarray(output, mode="RGBA"), detection

    t0 = time.perf_counter()
    x0, y0, x1, y1 = detection.roi
    crop_rgba = source_rgba.crop((x0, y0, x1, y1))
    crop_rgb = crop_rgba.convert("RGB")

    # Respect an alpha channel that came with the input instead of replacing
    # it with a new matte. This keeps RGBA transparent-folder detail intact.
    model_alpha = detection.alpha_roi.astype(np.uint16)
    input_alpha = np.asarray(crop_rgba.getchannel("A"), dtype=np.uint16)
    alpha = Image.fromarray(
        ((model_alpha * input_alpha + 127) // 255).astype(np.uint8),
        mode="L",
    )

    protection = product_protection or {}
    protect = any(protection.get(key, False) for key in (
        "preserve_diamond_color",
        "preserve_setting_details",
        "no_detail_smoothing",
    ))
    # The PyMatting colour estimate is only used when the protection step is
    # off, or when the preset explicitly asks for estimated colours. With
    # preserve_source_rgb (the jewellery default) it would be discarded, so
    # skip it - it is one of the slowest steps.
    estimate_used = (not protect) or (
        bool(protection.get("use_estimated_foreground_rgb", False))
        and not bool(protection.get("preserve_source_rgb", True))
    )
    settings = (bg_settings or {}).get("foreground_estimation", {})
    foreground_rgb = np.asarray(crop_rgb, dtype=np.uint8)
    if estimate_used and settings.get("enabled", True):
        try:
            foreground_rgb = estimate_foreground_color(crop_rgba, alpha, settings)
        except Exception:
            # Keep background removal resilient if PyMatting cannot process a
            # particular image; the source RGB remains the safest fallback.
            foreground_rgb = np.asarray(crop_rgb, dtype=np.uint8)

    result = Image.fromarray(
        np.dstack([foreground_rgb, np.asarray(alpha, dtype=np.uint8)]),
        mode="RGBA",
    )
    backdrop = None
    if detection.background_small is not None:
        backdrop = cv2.resize(
            detection.background_small, crop_rgb.size, interpolation=cv2.INTER_LINEAR)
    if protect:
        result = _preserve_in_strips(crop_rgb, result, protection, foreground_rgb, backdrop)

    result_arr = np.asarray(result, dtype=np.uint8).copy()
    if (bg_settings or {}).get("decontaminate_edges", True) and detection.background_small is not None:
        decontaminate_edges(result_arr, detection.background_small)
    output[y0:y1, x0:x1] = result_arr
    # A magnifier-lens frame printed on the photo is part of the picture:
    # put its white rim back whole instead of leaving cut-up pieces.
    try:
        from .glass_frame import restore_glass_frame
        detection.timings["glass_px"] = restore_glass_frame(
            output, np.asarray(source), bg_settings)
    except Exception:
        detection.timings["glass_px"] = 0
    detection.timings["protect_s"] = round(time.perf_counter() - t0, 2)
    return Image.fromarray(output, mode="RGBA"), detection


# ======================================================
# RESIZE IMAGE TO FIXED CANVAS
# ======================================================

def resize_to_canvas(
    image,
    bg_color_hex=None,
    target_width=None,
    target_height=None,
    auto_crop=False,
    padding_percent=0,
    max_upscale=1.0,
):
    """
    bg_color_hex = None  -> Transparent canvas.
    bg_color_hex = "#FFFFFF" -> White background canvas.

    auto_crop=True crops to the detected product first and centres it with
    ``padding_percent`` free space on each side, so every catalogue image is
    framed the same way regardless of how the photo was shot.
    """
    target_width = target_width or TARGET_WIDTH
    target_height = target_height or TARGET_HEIGHT

    if image.mode != "RGBA":
        image = image.convert("RGBA")
    else:
        image = image.copy()

    if auto_crop:
        image = crop_to_product(image)
        pad = min(max(float(padding_percent or 0), 0.0), 40.0) / 100.0
        inner_w = max(1, round(target_width * (1 - 2 * pad)))
        inner_h = max(1, round(target_height * (1 - 2 * pad)))
        scale = min(inner_w / image.width, inner_h / image.height)
        scale = min(scale, max(float(max_upscale or 1.0), 1.0))
        new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        if new_size != image.size:
            image = image.resize(new_size, Image.Resampling.LANCZOS)
    elif image.size != (target_width, target_height):
        # scale the whole photo to fit (never crop, never resample when the
        # upload already has the canvas size - pixels stay identical;
        # smaller photos are not blown up)
        scale = min(target_width / image.width, target_height / image.height, 1.0)
        new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        if new_size != image.size:
            image = image.resize(new_size, Image.Resampling.LANCZOS)

    if bg_color_hex:
        bg = hex_to_rgb(bg_color_hex)
        canvas = Image.new("RGBA", (target_width, target_height), bg + (255,))
    else:
        canvas = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))

    x = (target_width - image.width) // 2
    y = (target_height - image.height) // 2

    # Alpha-composite exactly once. paste(image, mask=image) applies the
    # alpha channel a second time and can make fine jewellery edges fade.
    canvas.alpha_composite(image, (x, y))

    return canvas


# ======================================================
# DRAW TEXT
# ======================================================

_FONT_CANDIDATES = (
    WINDOWS_TREBUCHET,
    "trebuc.ttf",
    "DejaVuSans.ttf",
    "LiberationSans-Regular.ttf",
    "arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/Library/Fonts/Arial.ttf",
)


def load_catalog_font(size):
    for candidate in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_text(img, lines, bg_color_hex=None):
    """Write the catalogue lines on the left, bottom-up from the corner."""
    lines = [l for l in (lines or []) if l]
    if not lines:
        return img
    draw = ImageDraw.Draw(img)

    # Black text on a light canvas, white on a dark one.
    fill = "white"
    if not bg_color_hex or (
        0.299 * hex_to_rgb(bg_color_hex)[0]
        + 0.587 * hex_to_rgb(bg_color_hex)[1]
        + 0.114 * hex_to_rgb(bg_color_hex)[2]
    ) > 140:
        fill = "black"

    size = max(18, round(img.height * 0.024))
    font = load_catalog_font(size)
    step = round(size * 1.3)
    margin = max(20, round(img.width * 0.015))

    x = margin
    y = img.height - step * len(lines) - margin
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += step

    return img


# ======================================================
# MAIN PROCESS FUNCTION
# ======================================================

def process_images(request):
    excel_file = request.FILES.get("excel_file")
    images = request.FILES.getlist("images")
    processing_mode = request.POST.get("processing_mode", "catalog").strip().lower()
    transparent_only = processing_mode == "transparent_only"

    itemwise_base = request.POST.get("itemwise_folder")
    style_base = request.POST.get("stylebio_folder")
    bg_color_hex = request.POST.get("bg_color") or "#FFFFFF"

    if not transparent_only and not excel_file:
        raise Exception("Excel file is required")

    if not images:
        raise Exception("Images are required")

    itemwise_base = resolve_output_directory(
        itemwise_base,
        os.path.join("media", "Itemwise"),
    )
    style_base = resolve_output_directory(
        style_base,
        os.path.join("media", "StyleBio"),
    )
    transparent_base = resolve_output_directory(
        None,
        os.path.join("media", "Transparent"),
    )

    if transparent_only:
        mapping, category_map, style_map = {}, {}, {}
    else:
        mapping, category_map, style_map = read_excel_mapping_cached(excel_file)
    results = []

    for img_file in images:

        try:
            filename = img_file.name
            ext = os.path.splitext(filename)[1].lower()
            name = os.path.splitext(filename)[0]
            # names used on disk; ``filename`` / ``name`` stay as uploaded
            # for the Excel lookup and the response
            save_name = safe_filename(name)
            save_filename = save_name + ext

            if ext not in SUPPORTED_EXTS:
                results.append({
                    "file": filename,
                    "status": "skipped",
                    "reason": f"Unsupported extension ({ext})"
                })
                continue

            preset_name = resolve_preset_name(request, filename=filename)
            preset = load_preset_settings(preset_name)
            meta = image_metadata(preset)          # author / software written into saved files
            image_settings = preset.get("image_processing", {})
            bg_settings = preset.get("background_removal", {})
            product_protection = preset.get("product_protection", {})
            output_settings = preset.get("output", {})

            key = base_token(name)
            # Transparent-only mode has no Excel mapping, so there is no
            # catalogue text to draw on the preview canvas.
            excel_key = find_excel_key(name, mapping)
            lines = mapping.get(excel_key) or []

            per_image_overrides = preset.get("per_image_overrides", {})
            item_override = {}
            if isinstance(per_image_overrides, dict):
                item_override = per_image_overrides.get(key)
                if isinstance(item_override, dict):
                    pp_override = item_override.get("product_protection", {})
                    if isinstance(pp_override, dict) and pp_override:
                        product_protection = {
                            **product_protection,
                            **pp_override,
                        }
                    output_override = item_override.get("output", {})
                    if isinstance(output_override, dict) and output_override:
                        output_settings = {
                            **output_settings,
                            **output_override,
                        }

            if not transparent_only and not lines:
                results.append({
                    "file": filename,
                    "status": "skipped",
                    "reason": "No Excel data"
                })
                continue

            # ---------------------------# ---------------------------
            # Load Image
            # ---------------------------
            with Image.open(img_file) as opened_image:
                source_rgba = opened_image.convert("RGBA")
                # keep the colour profile / camera data so the saved file
                # looks the same as the upload in every viewer
                source_icc = opened_image.info.get("icc_profile")
                source_exif = opened_image.info.get("exif")
                source_dpi = opened_image.info.get("dpi")
            im = source_rgba.convert("RGB")
            existing_alpha = source_rgba.getchannel("A")
            has_existing_transparency = existing_alpha.getextrema()[0] < 255

            # ---------------------------
            # Remove background and preserve product colour
            # ---------------------------
            detection = None
            reference_matte = (
                item_override.get("reference_matte")
                if isinstance(item_override, dict)
                else None
            )

            # Transparent-folder PNGs already contain the product matte. Do
            # not send them through RMBG again: a second segmentation pass
            # commonly removes the hollow diamond facets and white shine.
            if transparent_only and has_existing_transparency:
                transparent_im = source_rgba
            elif reference_matte:
                transparent_im = apply_reference_matte(
                    im,
                    reference_matte,
                    use_reference_rgb=bool(
                        item_override.get("use_reference_rgb", False)
                    ),
                    repair_regions=item_override.get("repair_regions"),
                )
            else:
                transparent_im, detection = remove_background_detailed(
                    im,
                    bg_settings={
                        **bg_settings,
                        "_validation": preset.get("validation", {}),
                    },
                    product_protection=product_protection,
                )
                if not detection.bbox and preset.get("validation", {}).get("require_product_visible", True):
                    results.append({
                        "file": filename,
                        "status": "skipped",
                        "reason": "No jewellery detected in the photo",
                    })
                    continue

            width = int(output_settings.get("width_px", TARGET_WIDTH) or TARGET_WIDTH)
            height = int(output_settings.get("height_px", TARGET_HEIGHT) or TARGET_HEIGHT)
            canvas_bg = output_settings.get("background_color", "transparent")
            canvas_color = None if canvas_bg.lower() == "transparent" else canvas_bg

            # Keep the transparent export at the actual source size.
            # The fixed canvas size is used only for white/itemwise outputs.
            original_size = im.size
            transparent_canvas = transparent_im.convert("RGBA")
            if transparent_canvas.size != original_size:
                transparent_canvas = transparent_canvas.resize(
                    original_size,
                    Image.Resampling.LANCZOS,
                )

            # ---------------------------
            # White Canvas
            # ---------------------------
            # keep_original_framing: the product stays exactly where it was in
            # the upload (same size, same position, no re-scaling), so the
            # saved JPG matches the uploaded photo. If the preset canvas has a
            # different size the whole photo is scaled once, never cropped.
            keep_framing = bool(image_settings.get("keep_original_framing", True))
            white_canvas = resize_to_canvas(
                transparent_im,
                bg_color_hex or canvas_color or "#FFFFFF",
                width,
                height,
                auto_crop=bool(image_settings.get("auto_crop", False)) and not keep_framing,
                padding_percent=(
                    image_settings.get("padding_percent", 0)
                    if image_settings.get("add_padding", True)
                    else 0
                ),
                max_upscale=float(image_settings.get("max_upscale", 2.0)),
            )

            # Add catalogue text only when Excel metadata exists.
            if lines:
                white_canvas = draw_text(
                    white_canvas,
                    lines,
                    bg_color_hex or canvas_color or "#FFFFFF",
                )

            # Small JPEG thumbnail for the browser (with the text on it);
            # the full files are on disk.
            preview = image_to_base64_jpeg(thumbnail(white_canvas, 900), quality=88)

            # ---------------------------
            # Save Transparent PNG
            # ---------------------------
            transparent_path = transparent_base / f"{save_name}.png"

            transparent_canvas.save(
                transparent_path,
                format="PNG",
                compress_level=1,
                dpi=tuple(source_dpi) if source_dpi else (300, 300),
                **({"icc_profile": source_icc} if source_icc else {}),
                **metadata_save_options("PNG", meta),
            )

            extra = {
                "preview_mime": "image/png" if transparent_only else "image/jpeg",
                "transparent_url": media_url_for(transparent_path),
                "warnings": list(detection.warnings) if detection else [],
                "detection": (
                    detection.as_dict(im.width, im.height) if detection else None
                ),
                "detection_preview": (
                    image_to_base64_jpeg(draw_detection_overlay(im, detection.boxes))
                    if detection and detection.boxes
                    else None
                ),
            }

            if transparent_only:
                results.append({
                    "file": filename,
                    "status": "success",
                    "preview": image_to_base64(thumbnail(transparent_canvas, 900, crop=True)),
                    "transparent_path": str(transparent_path),
                    "category": "",
                    "style": "",
                    "preset": preset_name,
                    **extra,
                })
                continue

            # ---------------------------
            # Save White Image
            # ---------------------------
            if ext in (".jpg", ".jpeg"):
                save_format = "JPEG"
            elif ext == ".webp":
                save_format = "WEBP"
            elif ext == ".bmp":
                save_format = "BMP"
            else:
                save_format = "PNG"

            save_options = {"dpi": tuple(source_dpi) if source_dpi else (300, 300)}
            if source_icc and save_format in ("JPEG", "PNG", "WEBP"):
                save_options["icc_profile"] = source_icc
            # author / software file properties (keeps the camera EXIF for JPEG)
            save_options.update(metadata_save_options(save_format, meta, source_exif))
            if save_format == "PNG":
                save_options["compress_level"] = 1
            elif save_format == "JPEG":
                # subsampling=0 (4:4:4): full colour resolution. Pillow's
                # default 4:2:0 halves colour detail and blurs thin gold
                # chains and diamond edges against the background.
                save_options.update({"quality": 100, "optimize": False, "subsampling": 0})
            elif save_format == "WEBP":
                save_options.update({"quality": 100, "method": 0})

            category = category_map.get(excel_key, "Others")
            style = style_map.get(excel_key, "OTH")

            cat_dir = itemwise_base / safe_filename(category, fallback="Others")
            style_dir = style_base / safe_filename(style, fallback="OTH")

            cat_dir.mkdir(parents=True, exist_ok=True)
            style_dir.mkdir(parents=True, exist_ok=True)

            cat_path = cat_dir / save_filename
            style_path = style_dir / save_filename
            white_rgb = white_canvas.convert("RGB")

            white_rgb.save(
                cat_path,
                format=save_format,
                **save_options,
            )

            white_rgb.save(
                style_path,
                format=save_format,
                **save_options,
            )

            # ---------------------------
            # API Response
            # ---------------------------
            results.append({
                "file": filename,
                "status": "success",
                "preview": preview,
                "category": category,
                "style": style,
                "preset": preset_name,
                "transparent_path": str(transparent_path),
                "category_path": str(cat_path),
                "stylebio_path": str(style_path),
                **extra,
            })

        except Exception as e:
            import traceback
            print(traceback.format_exc())

            results.append({
                "file": img_file.name,
                "status": "error",
                "error": str(e)
            })

    return results