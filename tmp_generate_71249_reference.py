from pathlib import Path

from PIL import Image

from processor.services.image_processor import apply_reference_matte, resize_to_canvas

project = Path(__file__).parent
source = Image.open(project.parent / "Test" / "71249.jpg").convert("RGB")
cutout = apply_reference_matte(
    source,
    "media/Transparent/71249-P.png",
)
output = resize_to_canvas(cutout, None, 2400, 1610)
target = project / "media" / "Transparent" / "71249_refined.png"
output.save(target, format="PNG", optimize=True)
print(f"Saved {target}")
