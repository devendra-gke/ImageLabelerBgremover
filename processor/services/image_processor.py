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
 
ssl._create_default_https_context = ssl._create_unverified_context
print("ssl done")

# ======================================================
# CONFIGURATION
# ======================================================

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


def preserve_original_product_pixels(original: Image.Image, processed: Image.Image) -> Image.Image:
    """Use the AI result only as an alpha mask and retain source RGB values.

    Do not restore bright pixels outside the mask: those can be background
    pixels and create white/grey halos around diamonds and fine gold wires.
    """
    source = original.convert("RGB")
    result = processed.convert("RGBA")

    if result.size != source.size:
        result = result.resize(source.size, Image.Resampling.LANCZOS)

    source_rgb = np.asarray(source, dtype=np.uint8)
    rgba = np.asarray(result, dtype=np.uint8).copy()
    alpha = rgba[..., 3].copy()

    # Keep the model's original soft alpha edges. Do not hard-threshold them;
    # tiny diamonds, prongs and thin chain links may occupy only a few pixels.
    rgba[..., :3] = source_rgb
    rgba[..., 3] = alpha
    rgba[alpha == 0, :3] = 0
    return Image.fromarray(rgba, mode="RGBA")

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
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def hex_to_rgb(hex_color):
    if not hex_color:
        return (255, 255, 255)

    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


# ======================================================
# EXCEL READER
# ======================================================

def read_excel_mapping(excel_file):
    df = pd.read_excel(excel_file, engine="openpyxl", dtype=str)
    df.columns = [str(c).strip().upper() for c in df.columns]

    key_series = get_series(df, "N")
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
    """Remove the background with the shared model and preserve source colour."""
    source = image.convert("RGB")
    bg_settings = bg_settings or {}
    product_protection = product_protection or {}

    if bg_settings.get("enabled", True) is False:
        return source.convert("RGBA")

    try:
        # Reuse the model initialized at module import; do not load one per image.
        try:
            output = remover.process(source, type="rgba")
        except TypeError:
            # Compatibility with transparent_background versions that don't
            # accept the `type` keyword.
            output = remover.process(source)

        if isinstance(output, Image.Image):
            processed = output.convert("RGBA")
        elif isinstance(output, (bytes, bytearray)):
            processed = Image.open(BytesIO(output)).convert("RGBA")
        else:
            processed = Image.fromarray(np.asarray(output)).convert("RGBA")

        if processed.size != source.size:
            processed = processed.resize(source.size, Image.Resampling.LANCZOS)

        if product_protection.get("preserve_original_color", True):
            processed = preserve_original_product_pixels(source, processed)

        return processed
    except Exception as exc:
        # Fail visibly instead of silently returning the original image with
        # its background still present.
        print("Transparent BG Error:", exc)
        raise RuntimeError(f"Background removal failed: {exc}") from exc


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

    if not itemwise_base:
        itemwise_base = os.path.join("media", "Itemwise")

    if not style_base:
        style_base = os.path.join("media", "StyleBio")

    transparent_base = os.path.join("media", "Transparent")

    os.makedirs(itemwise_base, exist_ok=True)
    os.makedirs(style_base, exist_ok=True)
    os.makedirs(transparent_base, exist_ok=True)

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
            transparent_path = os.path.join(
                transparent_base,
                name + ".png"
            )

            transparent_canvas.save(
                transparent_path,
                format="PNG",
                optimize=True,
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

            category = category_map.get(key, "Others")
            style = style_map.get(key, "OTH")

            cat_dir = os.path.join(itemwise_base, category)
            style_dir = os.path.join(style_base, style)

            os.makedirs(cat_dir, exist_ok=True)
            os.makedirs(style_dir, exist_ok=True)

            cat_path = os.path.join(cat_dir, filename)
            style_path = os.path.join(style_dir, filename)

            white_canvas.convert("RGB").save(
                cat_path,
                format=save_format,
                quality=100,
                optimize=True,
                dpi=(300, 300)
            )

            white_canvas.convert("RGB").save(
                style_path,
                format=save_format,
                quality=100,
                optimize=True,
                dpi=(300, 300)
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
                "transparent_path": transparent_path,
                "category_path": cat_path,
                "stylebio_path": style_path
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