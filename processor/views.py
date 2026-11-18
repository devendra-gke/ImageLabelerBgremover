import base64
import traceback
from io import BytesIO

from django.http import JsonResponse
from django.shortcuts import render
from django.views import View
from PIL import Image, ImageOps

from .services.image_processor import (
    load_preset_settings,
    remove_background_detailed,
    resolve_preset_name,
    process_images,
)


def upload_page(request):
    return render(request, "image_upload.html")


def _png_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buffer = BytesIO()
    img.save(buffer, format=fmt, compress_level=1) if fmt == "PNG" else img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode()


class ImageProcessAPI(View):
    def post(self, request):
        try:
            results = process_images(request)

            return JsonResponse({
                "data": results
            })

        except Exception as e:
            print(traceback.format_exc())  # VERY IMPORTANT (debugging)

            return JsonResponse({
                "error": str(e)
            }, status=500)


class Make3DAPI(View):
    """Build the textures for the 3D viewer from one jewellery photo.

    POST fields:
      image       - photo or transparent PNG
      depth_mode  - auto | shape | ai | hybrid
      roundness   - 0.3 .. 2.0 (shape profile, default 1)
      remove_bg   - auto | yes | no  (auto: skip if the PNG is already transparent)
      preset      - optional preset name (jewellery / diamond)
    """

    def post(self, request):
        from .services.depth3d import build_relief

        upload = request.FILES.get("image")
        if not upload:
            return JsonResponse({"error": "Please choose an image."}, status=400)
        try:
            with Image.open(upload) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGBA")

            remove_bg = (request.POST.get("remove_bg") or "auto").lower()
            has_alpha = image.getchannel("A").getextrema()[0] < 250
            warnings = []
            detection = None
            if remove_bg == "yes" or (remove_bg == "auto" and not has_alpha):
                preset = load_preset_settings(resolve_preset_name(request, filename=upload.name))
                cutout, result = remove_background_detailed(
                    image.convert("RGB"),
                    bg_settings={**preset.get("background_removal", {}), "_validation": preset.get("validation", {})},
                    product_protection=preset.get("product_protection", {}),
                )
                if not result.bbox:
                    return JsonResponse({"error": "No jewellery detected in this photo."}, status=422)
                warnings.extend(result.warnings)
                detection = result.as_dict(image.width, image.height)
                image = cutout

            try:
                roundness = float(request.POST.get("roundness") or 1.0)
            except ValueError:
                roundness = 1.0
            relief = build_relief(
                image,
                mode=request.POST.get("depth_mode") or "auto",
                roundness=min(max(roundness, 0.3), 2.0),
            )
            warnings.extend(relief["warnings"])
            return JsonResponse({
                "name": upload.name.rsplit(".", 1)[0],
                "cutout": _png_b64(relief["cutout"]),
                "depth": _png_b64(relief["depth"]),
                "detail": _png_b64(relief["detail"]),
                "width": relief["width"],
                "height": relief["height"],
                "depth_mode": relief["mode"],
                "detection": detection,
                "warnings": warnings,
            })
        except Exception as exc:
            print(traceback.format_exc())
            return JsonResponse({"error": str(exc)}, status=500)


class DepthStatusAPI(View):
    def get(self, request):
        from .services.depth3d import ai_depth_available

        return JsonResponse({"ai_depth_ready": ai_depth_available()})


class SelectOutputFolderAPI(View):
    """Open a native folder picker for the local Windows application."""

    def post(self, request):
        if request.META.get("REMOTE_ADDR") not in {"127.0.0.1", "::1"}:
            return JsonResponse({"error": "Folder selection is local-only."}, status=403)

        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected_folder = filedialog.askdirectory(
                parent=root,
                title="Choose output folder",
                mustexist=True,
            )
            root.destroy()

            return JsonResponse({
                "path": selected_folder,
                "cancelled": not bool(selected_folder),
            })
        except Exception as exc:
            return JsonResponse(
                {"error": f"Could not open the folder picker: {exc}"},
                status=500,
            )
