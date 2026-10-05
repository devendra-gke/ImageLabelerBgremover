from .services.image_processor import process_images
from django.http import JsonResponse
from django.views import View
import traceback
from transparent_background import Remover
from PIL import Image
import numpy as np

# Initialize once (recommended for Django)
remover = Remover()

def remove_background(image):
    """
    Same logic as Hugging Face Space.
    Accepts PIL Image or NumPy array.
    Returns a PIL Image with transparent background.
    """

    if isinstance(image, Image.Image):
        output = remover.process(image)

    elif isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
        output = remover.process(image_pil)

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
