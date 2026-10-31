"""
Jewelry background remover for a single image -> transparent PNG.

Pipeline:
  1. BiRefNet predicts a soft foreground mask.
  2. Alpha matting (PyMatting closed-form) refines edges / fine chains and
     decontaminates edge colors.
  3. SAM 2 is used as a fallback (auto when the BiRefNet mask looks wrong) or
     as a manual-correction tool (--sam2 / --point / --neg-point / --box).

Usage:
  python jewelry_bgremove.py input.jpg
  python jewelry_bgremove.py input.jpg -o out.png
  python jewelry_bgremove.py input.jpg --sam2 --box 100,80,900,700
  python jewelry_bgremove.py input.jpg --sam2 --point 450,300 --neg-point 50,50
  python jewelry_bgremove.py input.jpg --sam2 --sam2-mode union --point 450,300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

BIREFNET_ID = "ZhengPeng7/BiRefNet"
BIREFNET_SIZE = (1024, 1024)
SAM2_ID = "facebook/sam2.1-hiera-large"
MATTING_MAX_SIDE = 1400

# Auto-fallback thresholds (foreground area / image area)
MIN_FG_RATIO = 0.005
MAX_FG_RATIO = 0.90

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_birefnet = None
_sam2 = None


# ---------------------------------------------------------------- BiRefNet --
def _load_birefnet():
    global _birefnet
    if _birefnet is None:
        from transformers import AutoModelForImageSegmentation

        model = AutoModelForImageSegmentation.from_pretrained(
            BIREFNET_ID, trust_remote_code=True
        )
        model.to(DEVICE).eval()
        if DEVICE == "cuda":
            model.half()
        _birefnet = model
    return _birefnet


@torch.no_grad()
def birefnet_mask(image: Image.Image) -> np.ndarray:
    """Soft mask (H, W) float32 in [0, 1] at the original image size."""
    model = _load_birefnet()
    w, h = image.size

    resized = image.resize(BIREFNET_SIZE, Image.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    if DEVICE == "cuda":
        tensor = tensor.half()

    pred = model(tensor)[-1].sigmoid().float()
    pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
    return pred[0, 0].clamp(0, 1).cpu().numpy()


# ------------------------------------------------------------------- SAM 2 --
def _load_sam2(model_id: str):
    global _sam2
    if _sam2 is None:
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise RuntimeError(
                "SAM 2 is not installed. Run: pip install sam2"
            ) from exc
        _sam2 = SAM2ImagePredictor.from_pretrained(model_id, device=DEVICE)
    return _sam2


@torch.no_grad()
def sam2_mask(
    image: Image.Image,
    base_mask: np.ndarray | None,
    points: list[tuple[int, int]],
    neg_points: list[tuple[int, int]],
    box: tuple[int, int, int, int] | None,
    model_id: str = SAM2_ID,
) -> np.ndarray:
    """Soft mask (H, W) float32 in [0, 1] from SAM 2 using the given prompts.

    When no prompt is given, the bounding box of the BiRefNet mask is used,
    or the image center if BiRefNet found nothing.
    """
    predictor = _load_sam2(model_id)
    w, h = image.size
    rgb = np.asarray(image.convert("RGB"))

    if box is None and not points:
        if base_mask is not None and (base_mask > 0.5).any():
            ys, xs = np.where(base_mask > 0.5)
            box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        else:
            points = [(w // 2, h // 2)]

    point_coords = point_labels = None
    all_pts = [*points, *neg_points]
    if all_pts:
        point_coords = np.array(all_pts, dtype=np.float32)
        point_labels = np.array([1] * len(points) + [0] * len(neg_points), dtype=np.int32)

    box_arr = np.array(box, dtype=np.float32) if box is not None else None

    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if DEVICE == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    with autocast:
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_arr,
            multimask_output=(box_arr is None and len(all_pts) == 1),
            return_logits=True,
        )

    best = masks[int(np.argmax(scores))].astype(np.float32)
    return (1.0 / (1.0 + np.exp(-best))).astype(np.float32)


# ------------------------------------------------------------ Alpha matting --
def _odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 == 1 else n + 1


def _make_trimap(coarse: np.ndarray) -> np.ndarray:
    side = max(coarse.shape)
    k = _odd(side / 250)
    fg = cv2.erode((coarse > 0.9).astype(np.uint8), np.ones((k, k), np.uint8))
    maybe = cv2.dilate((coarse > 0.02).astype(np.uint8), np.ones((2 * k + 1,) * 2, np.uint8))

    trimap = np.full(coarse.shape, 0.5, dtype=np.float64)
    trimap[maybe == 0] = 0.0
    trimap[fg == 1] = 1.0
    return trimap


def refine_with_matting(image: Image.Image, coarse: np.ndarray):
    """Returns (alpha float32 [0,1] full-res, foreground RGB float32 [0,1] full-res)."""
    from pymatting import estimate_alpha_cf, estimate_foreground_ml

    w, h = image.size
    scale = min(1.0, MATTING_MAX_SIDE / max(w, h))
    mw, mh = max(1, round(w * scale)), max(1, round(h * scale))

    small_img = image.resize((mw, mh), Image.LANCZOS) if scale < 1 else image
    small_rgb = np.ascontiguousarray(np.asarray(small_img, dtype=np.float64) / 255.0)
    small_coarse = cv2.resize(coarse, (mw, mh), interpolation=cv2.INTER_LINEAR) if scale < 1 else coarse

    trimap = _make_trimap(small_coarse)
    alpha = estimate_alpha_cf(small_rgb, trimap)
    alpha = np.clip(alpha, 0, 1).astype(np.float32)

    if scale < 1:
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_CUBIC)
        alpha = np.clip(alpha, 0, 1)

    alpha[alpha < 0.01] = 0.0

    full_rgb = np.ascontiguousarray(np.asarray(image, dtype=np.float64) / 255.0)
    fg = estimate_foreground_ml(full_rgb, alpha.astype(np.float64))
    return alpha, np.clip(fg, 0, 1).astype(np.float32)


# ------------------------------------------------------------------ Pipeline --
def remove_background(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    use_sam2: bool = False,
    auto_fallback: bool = True,
    points: list[tuple[int, int]] | None = None,
    neg_points: list[tuple[int, int]] | None = None,
    box: tuple[int, int, int, int] | None = None,
    sam2_mode: str = "replace",
    sam2_model: str = SAM2_ID,
    matting: bool = True,
    save_mask: bool = False,
) -> Path:
    """Remove the background of one image and save a transparent PNG."""
    input_path = Path(input_path)
    output_path = (
        Path(output_path) if output_path else input_path.with_name(f"{input_path.stem}_transparent.png")
    )
    points = points or []
    neg_points = neg_points or []

    image = ImageOps.exif_transpose(Image.open(input_path)).convert("RGB")
    h, w = image.size[1], image.size[0]

    print(f"[1/3] BiRefNet on {DEVICE} ...")
    mask = birefnet_mask(image)

    ratio = float((mask > 0.5).mean())
    manual = use_sam2 or bool(points or neg_points or box)
    if not manual and auto_fallback and not (MIN_FG_RATIO <= ratio <= MAX_FG_RATIO):
        print(f"      BiRefNet foreground ratio {ratio:.3f} looks wrong -> SAM 2 fallback")
        manual = True

    if manual:
        print("[2/3] SAM 2 ...")
        sam = sam2_mask(image, mask, points, neg_points, box, sam2_model)
        if sam2_mode == "union":
            mask = np.maximum(mask, sam)
        else:
            mask = sam
    else:
        print("[2/3] SAM 2 not needed")

    if matting:
        print("[3/3] Alpha matting ...")
        alpha, fg = refine_with_matting(image, mask)
    else:
        print("[3/3] Alpha matting skipped")
        alpha, fg = mask, np.asarray(image, dtype=np.float32) / 255.0

    rgba = np.dstack([(fg * 255).round().astype(np.uint8), (alpha * 255).round().astype(np.uint8)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(output_path, format="PNG")

    if save_mask:
        Image.fromarray((alpha * 255).round().astype(np.uint8)).save(
            output_path.with_name(f"{output_path.stem}_mask.png")
        )

    print(f"Saved: {output_path}  ({w}x{h})")
    return output_path


# ------------------------------------------------------------------------ CLI --
def _pt(s: str) -> tuple[int, int]:
    x, y = s.split(",")
    return int(x), int(y)


def _box(s: str) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(v) for v in s.split(","))
    return x1, y1, x2, y2


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Jewelry background remover (BiRefNet + matting + SAM 2)")
    p.add_argument("input", help="input image path")
    p.add_argument("-o", "--output", help="output PNG path (default: <input>_transparent.png)")
    p.add_argument("--sam2", action="store_true", help="force SAM 2 instead of BiRefNet-only")
    p.add_argument("--no-auto-fallback", action="store_true", help="disable automatic SAM 2 fallback")
    p.add_argument("--point", type=_pt, action="append", default=[], metavar="X,Y",
                   help="positive SAM 2 click (object to keep); repeatable")
    p.add_argument("--neg-point", type=_pt, action="append", default=[], metavar="X,Y",
                   help="negative SAM 2 click (area to remove); repeatable")
    p.add_argument("--box", type=_box, metavar="X1,Y1,X2,Y2", help="SAM 2 box around the jewelry")
    p.add_argument("--sam2-mode", choices=["replace", "union"], default="replace",
                   help="replace BiRefNet mask with SAM 2, or union them (adds missed chains)")
    p.add_argument("--sam2-model", default=SAM2_ID, help=f"SAM 2 HF model id (default {SAM2_ID})")
    p.add_argument("--no-matting", action="store_true", help="skip alpha matting")
    p.add_argument("--save-mask", action="store_true", help="also save the alpha mask")
    a = p.parse_args(argv)

    remove_background(
        a.input,
        a.output,
        use_sam2=a.sam2,
        auto_fallback=not a.no_auto_fallback,
        points=a.point,
        neg_points=a.neg_point,
        box=a.box,
        sam2_mode=a.sam2_mode,
        sam2_model=a.sam2_model,
        matting=not a.no_matting,
        save_mask=a.save_mask,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
