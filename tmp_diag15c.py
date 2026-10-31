import numpy as np
from PIL import Image

from processor.services import image_processor as ip
from processor.services import hole_cleanup as hc

Image.MAX_IMAGE_PIXELS = None
src = Image.open(r"d:\Development\Image LabelerBgremove\Test\15.jpg").convert("RGB")
src.thumbnail((1600, 1600))
preset = ip.load_preset_settings("jewellery")
bgs = dict(preset["background_removal"])
bgs["remove_inner_holes"] = False
raw = ip.remove_background(src, bg_settings=bgs, product_protection=preset["product_protection"])
raw.save("tmp15_raw.png")
cleaned = hc.remove_enclosed_background(src, raw)
cleaned.save("tmp15_clean.png")

a = np.asarray(raw)[..., 3]
print("raw alpha>0", (a > 0).mean(), "cleaned alpha>0", (np.asarray(cleaned)[..., 3] > 0).mean())

ys, xs = np.where(a > 0)
box = (xs.min(), ys.min(), xs.max(), ys.max())
panels = []
for im in (raw, cleaned):
    m = Image.new("RGBA", im.size, (255, 0, 255, 255))
    m.alpha_composite(im)
    panels.append(m.convert("RGB").crop(box))
w, h = panels[0].size
s = 520 / w
size = (round(w * s), round(h * s))
canvas = Image.new("RGB", (size[0] * 2, size[1]))
for i, p in enumerate(panels):
    canvas.paste(p.resize(size, Image.LANCZOS), (i * size[0], 0))
canvas.save("tmp15_compare.jpg", quality=85)
print(canvas.size)
