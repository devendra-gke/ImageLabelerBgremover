import sys
import numpy as np
from PIL import Image

from processor.services.image_processor import load_preset_settings, remove_background

path = sys.argv[1]
preset = load_preset_settings("jewellery")
im = Image.open(path).convert("RGB")
im.thumbnail((1400, 1400))

out = remove_background(
    im,
    bg_settings=preset["background_removal"],
    product_protection=preset["product_protection"],
)
out.save("tmp_prompt_test.png")

bg = Image.new("RGBA", out.size, (255, 0, 255, 255))
bg.alpha_composite(out)
bg.convert("RGB").save("tmp_prompt_test_magenta.png")
a = np.asarray(out)[..., 3]
print("opaque", (a > 127).mean(), "haze", ((a > 0) & (a < 40)).mean())
