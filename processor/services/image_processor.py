from __future__ import annotations

import os
import base64
from io import BytesIO

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from transparent_background import Remover
import ssl
 
ssl._create_default_https_context = ssl._create_unverified_context
print("ssl done")

# ======================================================
# CONFIGURATION
# ======================================================

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
WINDOWS_TREBUCHET = r"C:\Windows\Fonts\trebuc.ttf"

TARGET_WIDTH = 1750
TARGET_HEIGHT = 1160

# Initialize remover once
remover = Remover(mode="base")


# ======================================================
# HELPERS
# ======================================================

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

def remove_background(image: Image.Image):
    remover = Remover()
    try:
        image = image.convert("RGB")
        output = remover.process(image)

        if isinstance(output, Image.Image):
            return output.convert("RGBA")

        return Image.open(BytesIO(output)).convert("RGBA")

    except Exception as e:
        print("Transparent BG Error:", e)
        return image.convert("RGBA")


# ======================================================
# RESIZE IMAGE TO FIXED CANVAS
# ======================================================

def resize_to_canvas(image, bg_color_hex=None):
    """
    bg_color_hex = None  -> Transparent canvas.
    bg_color_hex = "#FFFFFF" -> White background canvas.
    """

    if image.mode != "RGBA":
        image = image.convert("RGBA")

    image.thumbnail((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)

    if bg_color_hex:
        bg = hex_to_rgb(bg_color_hex)
        canvas = Image.new("RGBA", (TARGET_WIDTH, TARGET_HEIGHT), bg + (255,))
    else:
        canvas = Image.new("RGBA", (TARGET_WIDTH, TARGET_HEIGHT), (0, 0, 0, 0))

    x = (TARGET_WIDTH - image.width) // 2
    y = (TARGET_HEIGHT - image.height) // 2

    canvas.paste(image, (x, y), image)

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
            # Remove Background (HF Model)
            # ---------------------------
            transparent_im = remove_background(im)

            # Ensure RGBA
            if transparent_im.mode != "RGBA":
                transparent_im = transparent_im.convert("RGBA")

            # ---------------------------
            # Transparent Canvas
            # ---------------------------
            transparent_canvas = resize_to_canvas(transparent_im, None)

            # ---------------------------
            # White Canvas
            # ---------------------------
            white_canvas = resize_to_canvas(transparent_im, bg_color_hex)
            

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