import numpy as np
from PIL import Image

from processor.services import hole_cleanup as hc

raw = Image.open("tmp15_raw.png").convert("RGBA")
src = Image.open(r"d:\Development\Image LabelerBgremove\Test\15.jpg").convert("RGB")
src.thumbnail(raw.size)
cleaned = hc.remove_enclosed_background(src, raw)
cleaned.save("tmp15_clean.png")

a0 = np.asarray(raw)[..., 3]
a1 = np.asarray(cleaned)[..., 3]
print("removed px", int(((a0 > 0) & (a1 == 0)).sum()))

ys, xs = np.where(a0 > 0)
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
