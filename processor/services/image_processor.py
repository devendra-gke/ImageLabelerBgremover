from __future__ import annotations

import os
import pandas as pd
import numpy as np
import torch
import time

from PIL import Image, ImageDraw, ImageFont, ImageOps
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# ---------------- CONFIG ----------------
SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
WINDOWS_TREBUCHET = r"C:\\Windows\\Fonts\\trebuc.ttf"

MODEL_ID = "ZhengPeng7/BiRefNet"
MODEL_IMAGE_SIZE = (1024, 1024)

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_float32_matmul_precision("high")

# ---------------- TRANSFORM ----------------
transform_image = transforms.Compose([
    transforms.Resize(MODEL_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

# ---------------- LOAD MODEL ----------------
model = AutoModelForImageSegmentation.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)
model.to(device)
model.eval()

# ---------------- HELPERS ----------------
def base_token(s: str) -> str:
    s = (s or "").strip()
    if "_" in s:
        s = s.split("_", 1)[0]
    return s.lower()


def get_series(df, col_letter):
    col_letter = col_letter.upper()
    if col_letter in df.columns:
        return df[col_letter]
    idx = ord(col_letter) - ord('A')
    return df.iloc[:, idx]


def sanitize(name):
    return str(name).replace("/", "_").replace("\\", "_").strip()


# ---------------- EXCEL READER ----------------
def read_excel_mapping(excel_file):
    df = pd.read_excel(excel_file, engine='openpyxl', dtype=str)
    df.columns = [str(c).strip().upper() for c in df.columns]

    key_series = get_series(df, 'N')
    text_columns = [get_series(df, chr(c)) for c in range(ord('E'), ord('K') + 1)]

    category_series = get_series(df, 'B')
    style_series = get_series(df, 'E')

    mapping = {}
    category_map = {}
    style_map = {}

    for i in range(len(df)):
        key = key_series.iloc[i]
        if pd.isna(key):
            continue

        key = base_token(str(key).strip())

        # TEXT
        lines = []
        for col in text_columns:
            val = col.iloc[i]
            if pd.notna(val):
                text = str(val).strip()
                if text and text.lower() != 'nan':
                    lines.append(text)

        if lines:
            mapping[key] = lines

        # CATEGORY
        cat = category_series.iloc[i]
        if pd.notna(cat):
            category_map[key] = sanitize(cat)

        # STYLE
        style = style_series.iloc[i]
        if pd.notna(style):
            style_map[key] = sanitize(style)[:3]

    return mapping, category_map, style_map


# ---------------- BACKGROUND REMOVAL ----------------
def preprocess_image(image: Image.Image):
    return transform_image(image).unsqueeze(0)


def predict_mask(image: Image.Image):
    input_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()

    pred = preds[0].squeeze()

    mask = transforms.ToPILImage()(pred)
    mask = mask.resize(image.size)

    return mask.convert("L")


def remove_background(image: Image.Image):
    image = image.convert("RGB")

    mask = predict_mask(image)

    cutout = image.copy()
    cutout.putalpha(mask)

    return cutout


# ---------------- DRAW TEXT ----------------
def draw_text(img, lines):
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(WINDOWS_TREBUCHET, 28)
    except:
        font = ImageFont.load_default()

    x, y = 50, img.height - (30 * len(lines)) - 50

    for line in lines:
        draw.text((x, y), line, fill="white", font=font)
        y += 30

    return img


import base64
from io import BytesIO

def image_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# ---------------- MAIN FUNCTION ----------------
def process_images(request):
    excel_file = request.FILES.get('excel_file')
    images = request.FILES.getlist('images')

    # ✅ FIXED (important)
    itemwise_base = request.POST.get('itemwise_folder')
    style_base = request.POST.get('stylebio_folder')

    bg_color_hex = request.POST.get('bg_color')

    if not excel_file:
        raise Exception("Excel file is required")

    if not images:
        raise Exception("Images are required")

    if not itemwise_base:
        itemwise_base = os.path.join("media", "Itemwise")

    if not style_base:
        style_base = os.path.join("media", "StyleBio")

    os.makedirs(itemwise_base, exist_ok=True)
    os.makedirs(style_base, exist_ok=True)

    mapping, category_map, style_map = read_excel_mapping(excel_file)

    results = []

    for img_file in images:
        try:
            filename = img_file.name

            if not filename.lower().endswith(SUPPORTED_EXTS):
                continue

            name = os.path.splitext(filename)[0]

            if not name.endswith("_1"):
                results.append({
                    "file": filename,
                    "status": "skipped",
                    "reason": "Not _1 image"
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

            # ---------------- IMAGE LOAD ----------------
            im = Image.open(img_file)
            im = ImageOps.exif_transpose(im).convert("RGB")

            # ---------------- BG REMOVE (SAFE) ----------------
            try:
                im = remove_background(im)
            except Exception as e:
                print("BG removal failed:", e)

            # ---------------- BLACK BG ----------------
            if im.mode == "RGBA":
                bg_color = hex_to_rgb(bg_color_hex)
                bg = Image.new("RGB", im.size, bg_color)
                bg.paste(im, mask=im.split()[3])
                im = bg

            # ---------------- PREVIEW ----------------
            preview = image_to_base64(im)

            # ---------------- TEXT ----------------
            im = draw_text(im, lines)

            # ---------------- CATEGORY SAVE ----------------
            category = category_map.get(key, "Others")
            cat_dir = os.path.join(itemwise_base, category)
            os.makedirs(cat_dir, exist_ok=True)

            cat_path = os.path.join(cat_dir, filename)
            if not os.path.exists(cat_path):
                im.save(cat_path)

            # ---------------- STYLE SAVE ----------------
            style = style_map.get(key, "OTH")
            style_dir = os.path.join(style_base, style)
            os.makedirs(style_dir, exist_ok=True)

            style_path = os.path.join(style_dir, filename)
            if not os.path.exists(style_path):
                im.save(style_path)

            results.append({
                "file": filename,
                "status": "success",
                "preview": preview,
                "category": category,
                "style": style,
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
