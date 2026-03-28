from __future__ import annotations

import os
import pandas as pd
import numpy as np
import torch
import random

from PIL import Image, ImageDraw, ImageFont, ImageOps
from io import BytesIO
from pathlib import Path

from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# ====== FLUX IMPORT ======
from diffusers.pipelines.flux2.pipeline_flux2_klein_kv import Flux2KleinKVPipeline

# ---------------- CONFIG ----------------
SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
WINDOWS_TREBUCHET = r"C:\\Windows\\Fonts\\trebuc.ttf"

MODEL_ID = "ZhengPeng7/BiRefNet"
MODEL_IMAGE_SIZE = (1024, 1024)

HF_TOKEN = "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"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")

# ---------------- LOAD BIREFNET ----------------
transform_image = transforms.Compose([
    transforms.Resize(MODEL_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

biref_model = AutoModelForImageSegmentation.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)
biref_model.to(device)
biref_model.eval()

# ---------------- LOAD FLUX ----------------
print("Loading FLUX model...")

flux_pipe = Flux2KleinKVPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9b-kv",
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN
)
flux_pipe.to(device)

# ---------------- HELPERS ----------------
def base_token(s: str) -> str:
    s = (s or "").strip()
    if "_" in s:
        s = s.split("_", 1)[0]
    return s.lower()

def sanitize(name):
    return str(name).replace("/", "_").replace("\\", "_").strip()

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# ---------------- EXCEL ----------------
def read_excel_mapping(excel_file):
    df = pd.read_excel(excel_file, engine='openpyxl', dtype=str)
    df.columns = [str(c).strip().upper() for c in df.columns]

    key_series = df.iloc[:, 13]  # column N
    category_series = df.iloc[:, 1]  # column B
    style_series = df.iloc[:, 4]  # column E

    mapping, category_map, style_map = {}, {}, {}

    for i in range(len(df)):
        key = key_series.iloc[i]
        if pd.isna(key):
            continue

        key = base_token(str(key))
        lines = []

        for col in range(4, 11):
            val = df.iloc[i, col]
            if pd.notna(val):
                lines.append(str(val).strip())

        if lines:
            mapping[key] = lines

        if pd.notna(category_series.iloc[i]):
            category_map[key] = sanitize(category_series.iloc[i])

        if pd.notna(style_series.iloc[i]):
            style_map[key] = sanitize(style_series.iloc[i])[:3]

    return mapping, category_map, style_map

# ---------------- BIREFNET BG REMOVE ----------------
def remove_background_biref(image: Image.Image):
    image = image.convert("RGB")
    input_tensor = transform_image(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = biref_model(input_tensor)[-1].sigmoid().cpu()

    mask = preds[0].squeeze()
    mask = transforms.ToPILImage()(mask).resize(image.size).convert("L")

    cutout = image.copy()
    cutout.putalpha(mask)

    return cutout

# ---------------- FLUX BG REMOVE ----------------
def remove_background_flux(image: Image.Image):
    prompt = """
    Remove background from this image.
    Keep only the main object.
    Clean, studio background.
    Sharp edges, no artifacts.
    """

    generator = torch.Generator(device=device).manual_seed(random.randint(0, 999999))

    result = flux_pipe(
        prompt=prompt,
        image=[image],
        num_inference_steps=4,
        height=image.height,
        width=image.width,
        generator=generator
    ).images[0]

    return result

# ---------------- SAFE BG REMOVE ----------------
def remove_background(image: Image.Image):
    try:
        return remove_background_biref(image)
    except Exception as e:
        print("BiRefNet failed, using FLUX:", e)
        return remove_background_flux(image)

# ---------------- TEXT ----------------
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

# ---------------- MAIN FUNCTION ----------------
def process_images(request):
    excel_file = request.FILES.get('excel_file')
    images = request.FILES.getlist('images')

    itemwise_base = request.POST.get('itemwise_folder') or "media/Itemwise"
    style_base = request.POST.get('stylebio_folder') or "media/StyleBio"
    bg_color_hex = request.POST.get('bg_color') or "#000000"

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
                continue

            key = base_token(name)
            lines = mapping.get(key)

            if not lines:
                continue

            # LOAD IMAGE
            im = Image.open(img_file)
            im = ImageOps.exif_transpose(im).convert("RGB")

            # REMOVE BG
            im = remove_background(im)

            # APPLY BG COLOR
            if im.mode == "RGBA":
                bg_color = hex_to_rgb(bg_color_hex)
                bg = Image.new("RGB", im.size, bg_color)
                bg.paste(im, mask=im.split()[3])
                im = bg

            # ADD TEXT
            im = draw_text(im, lines)

            # SAVE CATEGORY
            category = category_map.get(key, "Others")
            cat_dir = os.path.join(itemwise_base, category)
            os.makedirs(cat_dir, exist_ok=True)

            cat_path = os.path.join(cat_dir, filename)
            im.save(cat_path)

            # SAVE STYLE
            style = style_map.get(key, "OTH")
            style_dir = os.path.join(style_base, style)
            os.makedirs(style_dir, exist_ok=True)

            style_path = os.path.join(style_dir, filename)
            im.save(style_path)

            results.append({
                "file": filename,
                "status": "success",
                "category": category,
                "style": style
            })

        except Exception as e:
            results.append({
                "file": img_file.name,
                "status": "error",
                "error": str(e)
            })

    return results