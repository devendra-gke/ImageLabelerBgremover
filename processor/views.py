from .services.image_processor import process_images
from django.http import JsonResponse
from django.views import View
import traceback
from PIL import Image
import numpy as np
from rembg import new_session, remove as rembg_remove

rembg_session = new_session("bria-rmbg")

def remove_background(image):
    """
    Same logic as Hugging Face Space.
    Accepts PIL Image or NumPy array.
    Returns a PIL Image with transparent background.
    """

    if isinstance(image, Image.Image):
        output = rembg_remove(image.convert("RGB"), session=rembg_session)

    elif isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
        output = rembg_remove(image_pil.convert("RGB"), session=rembg_session)

    else:
        raise TypeError("Unsupported image type")

    return output



##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

def upload_page(request):
    from django.shortcuts import render
    return render(request, "image_upload.html")


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
