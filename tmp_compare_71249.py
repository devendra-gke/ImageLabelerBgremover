from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(r"d:\Development\Image LabelerBgremove\ImageLabelerBgremover")
orig_path = BASE / "media" / "Transparent" / "71249.png"
tuned_path = BASE / "media" / "Transparent" / "71249_tuned.png"
opt_path = BASE / "media" / "Transparent" / "71249_tuned_opt.png"
preview_path = BASE / "media" / "Transparent" / "71249_tuned_preview.png"

orig = Image.open(orig_path).convert("RGBA")
tuned = Image.open(tuned_path).convert("RGBA")

tuned.save(opt_path, format="PNG", optimize=True)

preview = tuned.copy()
preview.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
preview.save(preview_path, format="PNG", optimize=True)

a = np.array(orig)[:, :, 3]
b = np.array(tuned)[:, :, 3]

print("orig_nonzero", int((a > 0).sum()))
print("tuned_nonzero", int((b > 0).sum()))
print("orig_hi", int((a >= 200).sum()))
print("tuned_hi", int((b >= 200).sum()))
print(f"saved: {opt_path}")
print(f"saved: {preview_path}")
