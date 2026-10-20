from __future__ import annotations

import base64
import json
import os
import ssl
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from transparent_background import Remover
import os
from huggingface_hub import InferenceClient
 
ssl._create_default_https_context = ssl._create_unverified_context
print("ssl done")

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
    "jewellery_image": "jewellery_image_settings.json",
}

# Initialize remover once
remover = Remover(mode="base")


# ======================================================
# HELPERS
# ======================================================


import numpy as np
from PIL import Image, ImageFilter
from io import BytesIO



def remove_background_huggingface(image: Image.Image) -> Image.Image:
    """
    Use a Hugging Face image-segmentation model to generate
    an alpha mask. The original jewellery RGB pixels are retained.

    HF_SEGMENTATION_MODEL must identify a compatible segmentation
    model, not an image-generation model such as Qwen-Image.
    """
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



def preserve_original_product_pixels(
    original: Image.Image,
    processed: Image.Image,
    product_protection: dict | None = None,
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

    # Start with the original image colour so stone tones stay intact.
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

            # Bright pixels near uncertain edges are likely real sparkle,
            # so raise alpha there using a localized expanded mask.
            sparkle = np.max(src, axis=2) >= highlight_threshold
            uncertain = alpha_u8 < 245
            keep_mask = sparkle & uncertain & (expanded_alpha > 0)

            boosted = np.minimum(expanded_alpha, sparkle_min_alpha)

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
                )
                alpha_u8[recover_zero] = np.maximum(
                    alpha_u8[recover_zero],
                    boosted[recover_zero],
                )

    if product_protection.get("preserve_facet_boundaries", True):
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
        if "jewel" in lowered or "jewellery" in lowered:
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
    return s.lower()


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

def read_excel_mapping(excel_file):
    df = pd.read_excel(excel_file, engine="openpyxl", dtype=str)
    df.columns = [str(c).strip().upper() for c in df.columns]

    key_series = get_series(df, "F")
    text_columns = [get_series(df, chr(c)) for c in range(ord("E"), ord("K") + 1)]
    category_series = get_series(df, "B")
    style_series = get_series(df, "E")

    mapping = {}
    category_map = {}
    style_map = {}

    for i in range(len(df)):
        key = key_series.iloc[i]

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

        cat = category_series.iloc[i]
        if pd.notna(cat):
            category_map[key] = sanitize(cat)

        style = style_series.iloc[i]
        if pd.notna(style):
            style_map[key] = sanitize(style)[:3]

    return mapping, category_map, style_map


# ======================================================
# BACKGROUND REMOVAL
# ======================================================

def remove_background(
    image: Image.Image,
    bg_settings: dict | None = None,
    product_protection: dict | None = None,
):
    source = image.convert("RGB")
    bg_settings = bg_settings or {}
    product_protection = product_protection or {}

    if not bg_settings.get("enabled", True):
        return source.convert("RGBA")

    selected_backend = str(
        bg_settings.get("backend") or BACKGROUND_REMOVAL_BACKEND
    ).strip().lower()

    if selected_backend == "huggingface":
        try:
            processed = remove_background_huggingface(source)
        except Exception:
            if not bg_settings.get("fallback_to_local", True):
                raise

            output = remover.process(source, type="rgba")

            if isinstance(output, Image.Image):
                processed = output.convert("RGBA")
            elif isinstance(output, (bytes, bytearray)):
                processed = Image.open(
                    BytesIO(output)
                ).convert("RGBA")
            else:
                processed = Image.fromarray(
                    np.asarray(output)
                ).convert("RGBA")
    else:
        output = remover.process(source, type="rgba")

        if isinstance(output, Image.Image):
            processed = output.convert("RGBA")
        elif isinstance(output, (bytes, bytearray)):
            processed = Image.open(
                BytesIO(output)
            ).convert("RGBA")
        else:
            processed = Image.fromarray(
                np.asarray(output)
            ).convert("RGBA")

    if processed.size != source.size:
        processed = processed.resize(
            source.size,
            Image.Resampling.LANCZOS,
        )

    if product_protection.get("preserve_original_color", True):
        processed = preserve_original_product_pixels(
            source,
            processed,
            product_protection=product_protection,
        )

    return processed


# ======================================================
# RESIZE IMAGE TO FIXED CANVAS
# ======================================================

def resize_to_canvas(image, bg_color_hex=None, target_width=None, target_height=None):
    """
    bg_color_hex = None  -> Transparent canvas.
    bg_color_hex = "#FFFFFF" -> White background canvas.
    """
    target_width = target_width or TARGET_WIDTH
    target_height = target_height or TARGET_HEIGHT

    if image.mode != "RGBA":
        image = image.convert("RGBA")

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

def draw_text(img, lines):
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(WINDOWS_TREBUCHET, 28)
    except Exception:
        font = ImageFont.load_default()

    x = 40
    y = img.height - (35 * len(lines)) - 40

    for line in lines:
        draw.text((x, y), line, fill="white", font=font)
        y += 35

    return img


# ======================================================
# MAIN PROCESS FUNCTION
# ======================================================

def process_images(request):
    excel_file = request.FILES.get("excel_file")
    images = request.FILES.getlist("images")

    itemwise_base = request.POST.get("itemwise_folder")
    style_base = request.POST.get("stylebio_folder")
    bg_color_hex = request.POST.get("bg_color") or "#FFFFFF"

    if not excel_file:
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

    mapping, category_map, style_map = read_excel_mapping(excel_file)
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
            lines = mapping.get(key)

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

            if not lines:
                results.append({
                    "file": filename,
                    "status": "skipped",
                    "reason": "No Excel data"
                })
                continue

            # ---------------------------# ---------------------------
            # Load Image
            # ---------------------------
            im = Image.open(img_file).convert("RGB")

            # ---------------------------
            # Remove background and preserve product colour
            # ---------------------------
            reference_matte = (
                item_override.get("reference_matte")
                if isinstance(item_override, dict)
                else None
            )

            if reference_matte:
                transparent_im = apply_reference_matte(
                    im,
                    reference_matte,
                    use_reference_rgb=bool(
                        item_override.get("use_reference_rgb", False)
                    ),
                    repair_regions=item_override.get("repair_regions"),
                )
            else:
                transparent_im = remove_background(
                    im,
                    bg_settings=bg_settings,
                    product_protection=product_protection,
                )

            width = int(output_settings.get("width_px", TARGET_WIDTH) or TARGET_WIDTH)
            height = int(output_settings.get("height_px", TARGET_HEIGHT) or TARGET_HEIGHT)
            canvas_bg = output_settings.get("background_color", "transparent")
            canvas_color = None if canvas_bg.lower() == "transparent" else canvas_bg

            # ---------------------------
            # Transparent Canvas
            # ---------------------------
            transparent_canvas = resize_to_canvas(transparent_im, None, width, height)

            # ---------------------------
            # White Canvas
            # ---------------------------
            white_canvas = resize_to_canvas(transparent_im, bg_color_hex or canvas_color or "#FFFFFF", width, height)

            preview = image_to_base64(white_canvas)

            # Add text only on white image
            white_canvas = draw_text(white_canvas, lines)

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
                "stylebio_path": str(style_path)
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