import numpy as np
import cv2
from PIL import Image
from processor.services import hole_cleanup as hc

Image.MAX_IMAGE_PIXELS = None
src = Image.open(r"d:\Development\Image LabelerBgremove\Test\15.jpg").convert("RGB")
cut = Image.open(
    r"d:\Development\Image LabelerBgremove\ImageLabelerBgremover\media\Transparent\15.png"
).convert("RGBA")
print(src.size, cut.size)
if cut.size != src.size:
    cut = cut.resize(src.size, Image.LANCZOS)

a = np.asarray(cut)[..., 3]
ys, xs = np.where(a > 0)
x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
print("bbox", x0, y0, x1, y1)
pad = 40
box = (max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, src.width), min(y1 + pad, src.height))
s = src.crop(box)
c = cut.crop(box)
scale = 900 / max(s.size)
size = (round(s.width * scale), round(s.height * scale))
s = s.resize(size, Image.LANCZOS)
c = c.resize(size, Image.LANCZOS)
m = Image.new("RGBA", size, (255, 0, 255, 255))
m.alpha_composite(c)
canvas = Image.new("RGB", (size[0] * 2, size[1]))
canvas.paste(s, (0, 0))
canvas.paste(m.convert("RGB"), (size[0], 0))
canvas.save(r"d:\Development\Image LabelerBgremove\ImageLabelerBgremover\tmp_inspect15_side.jpg", quality=85)
print("saved", canvas.size)
