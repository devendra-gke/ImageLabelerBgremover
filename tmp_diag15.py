import numpy as np
import cv2
from PIL import Image

from processor.services import hole_cleanup as hc

Image.MAX_IMAGE_PIXELS = None
src = Image.open(r"d:\Development\Image LabelerBgremove\Test\15.jpg").convert("RGB")
print("src", src.size)
src.thumbnail((1400, 1400))
cut = Image.open(
    r"d:\Development\Image LabelerBgremove\ImageLabelerBgremover\media\Transparent\15.png"
).convert("RGBA").resize(src.size, Image.LANCZOS)

s = np.asarray(src, dtype=np.float32)
alpha = np.asarray(cut)[..., 3]
print("alpha>0", (alpha > 0).mean(), "alpha==255", (alpha == 255).mean())

bg = hc._estimate_background(s, alpha)
print("bg estimate", bg)
dist = np.sqrt(((s - bg) ** 2).sum(axis=2))

# Pixels that are still opaque but look like background.
opaque = alpha > 200
for tol in (12, 30, 50, 80):
    print("opaque & dist<=%d" % tol, ((dist <= tol) & opaque).sum())

gray = cv2.cvtColor(s.astype(np.uint8), cv2.COLOR_RGB2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
grad = np.sqrt(gx * gx + gy * gy) / 4.0
print("grad pct of opaque & dist<=30", np.percentile(grad[(dist <= 30) & opaque], [50, 90, 99]) if ((dist <= 30) & opaque).any() else None)

out = hc.remove_enclosed_background(src, cut)
a2 = np.asarray(out)[..., 3]
print("removed px by cleanup", int(((alpha > 0) & (a2 == 0)).sum()))

# Sample colours of opaque pixels that are brightest/least saturated (hollow candidates)
hsv = cv2.cvtColor(s.astype(np.uint8), cv2.COLOR_RGB2HSV)
cand = opaque & (hsv[..., 1] < 40) & (hsv[..., 2] > 150)
print("low-sat bright opaque px", cand.sum())
if cand.any():
    print("their mean colour", s[cand].mean(axis=0), "mean dist", dist[cand].mean())

vis = np.asarray(src).copy()
vis[opaque & (dist <= 30)] = (255, 0, 0)
Image.fromarray(vis).save(r"d:\Development\Image LabelerBgremove\ImageLabelerBgremover\tmp_inspect15_red.png")
