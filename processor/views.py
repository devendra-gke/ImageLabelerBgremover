from .services.image_processor import process_images
from django.http import JsonResponse
from django.views import View
import traceback


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
