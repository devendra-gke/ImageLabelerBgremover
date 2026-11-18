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
) -> Image.Image:
    """
    Preserve original jewellery colours and shine.
    Keep source RGB and model alpha, with optional conservative edge cleanup.
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
            highlight_mask = np.max(src, axis=2) >= highlight_threshold
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
        expanded_alpha = np.asarray(
            Image.fromarray(alpha_u8, mode="L").filter(
                ImageFilter.MaxFilter(size=kernel)
            ),
            dtype=np.uint8,
        )
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
            sparkle = np.max(src, axis=2) >= highlight_threshold
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
        bright_sparkle = np.max(src, axis=2) >= haze_highlight_threshold

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


def sanitize(name):
    return str(name).replace("/", "_").replace("\\", "_").strip()


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


def read_excel_mapping(excel_file):
    df = pd.read_excel(excel_file, engine="openpyxl", dtype=str)
    df.columns = [str(c).strip() for c in df.columns]

    key_series = get_named_series(
        df,
        "TAGNO",
        "TagNo",
        "Tag No",
        "tag no",
        "Image Tag",
    )
    category_series = get_named_series(df, "Category", "CATEGORY")
    style_series = get_named_series(df, "StyleBio", "Style Bio", "STYLEBIO")

    text_field_names = [
        "StyleBio",
        "TagNo",
        "DiaWt",
        "ChainWt",
        "ChainWT",
        "GrossWt",
        "StoneWt",
    ]
    text_columns = []
    seen_headers = set()
    for name in text_field_names:
        header = normalize_excel_header(name)
        if header in seen_headers:
            continue
        series = get_named_series(df, name)
        if series is not None:
            seen_headers.add(header)
            text_columns.append(series)

    mapping = {}
    category_map = {}
    style_map = {}

    for i in range(len(df)):
        key = key_series.iloc[i] if key_series is not None else None

        if pd.isna(key):
            continue

        key = base_token(str(key).strip())

        lines = []
        for col in text_columns:
            val = col.iloc[i]
            if pd.notna(val):
                text = str(val).strip()
                if text and text.lower() != "nan":
                    lines.append(text)

        if lines:
            mapping[key] = lines

        if category_series is not None:
            cat = category_series.iloc[i]
            if pd.notna(cat):
                category_map[key] = sanitize(cat)

        if style_series is not None:
            style = style_series.iloc[i]
            if pd.notna(style):
                style_map[key] = sanitize(style)[:3]

    return mapping, category_map, style_map


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
    if protect:
        result = preserve_original_product_pixels(
            crop_rgb,
            result,
            protection,
            estimated_rgb=foreground_rgb,
        )

    output[y0:y1, x0:x1] = np.asarray(result, dtype=np.uint8)
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
    else:
        image.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)

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

def draw_text(img, lines, bg_color_hex=None):
    lines = lines or []
    draw = ImageDraw.Draw(img)

    # White text on a light canvas is invisible, so pick black or white
    # depending on the background brightness.
    fill = "white"
    if bg_color_hex:
        r, g, b = hex_to_rgb(bg_color_hex)
        if (0.299 * r + 0.587 * g + 0.114 * b) > 140:
            fill = "black"

    try:
        font = ImageFont.truetype(WINDOWS_TREBUCHET, 28)
    except Exception:
        font = ImageFont.load_default()

    x = 40
    y = img.height - (35 * len(lines)) - 40

    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += 35

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

            if ext not in SUPPORTED_EXTS:
                results.append({
                    "file": filename,
                    "status": "skipped",
                    "reason": f"Unsupported extension ({ext})"
                })
                continue

            preset_name = resolve_preset_name(request, filename=filename)
            preset = load_preset_settings(preset_name)
            image_settings = preset.get("image_processing", {})
            bg_settings = preset.get("background_removal", {})
            product_protection = preset.get("product_protection", {})
            output_settings = preset.get("output", {})

            key = base_token(name)
            # Transparent-only mode has no Excel mapping, so there is no
            # catalogue text to draw on the preview canvas.
            lines = mapping.get(key) or []

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
            white_canvas = resize_to_canvas(
                transparent_im,
                bg_color_hex or canvas_color or "#FFFFFF",
                width,
                height,
                auto_crop=bool(image_settings.get("auto_crop", False)),
                padding_percent=(
                    image_settings.get("padding_percent", 0)
                    if image_settings.get("add_padding", True)
                    else 0
                ),
                max_upscale=float(image_settings.get("max_upscale", 2.0)),
            )

            # Small JPEG thumbnail for the browser; the full files are on disk.
            preview = image_to_base64_jpeg(thumbnail(white_canvas, 900), quality=88)

            # Add catalogue text only when Excel metadata exists.
            if lines:
                white_canvas = draw_text(
                    white_canvas,
                    lines,
                    bg_color_hex or canvas_color or "#FFFFFF",
                )

            # ---------------------------
            # Save Transparent PNG
            # ---------------------------
            transparent_path = transparent_base / f"{name}.png"

            transparent_canvas.save(
                transparent_path,
                format="PNG",
                compress_level=1,
                dpi=(300, 300)
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

            save_options = {"dpi": (300, 300)}
            if save_format == "PNG":
                save_options["compress_level"] = 1
            elif save_format == "JPEG":
                save_options.update({"quality": 100, "optimize": False})
            elif save_format == "WEBP":
                save_options.update({"quality": 100, "method": 0})

            category = category_map.get(key, "Others")
            style = style_map.get(key, "OTH")

            cat_dir = itemwise_base / category
            style_dir = style_base / style

            cat_dir.mkdir(parents=True, exist_ok=True)
            style_dir.mkdir(parents=True, exist_ok=True)

            cat_path = cat_dir / filename
            style_path = style_dir / filename
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