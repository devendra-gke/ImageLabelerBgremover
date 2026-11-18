from django.urls import path
from .views import (
    DepthStatusAPI,
    ImageProcessAPI,
    Make3DAPI,
    SelectOutputFolderAPI,
    upload_page,
)

urlpatterns = [
    path('', upload_page),
    path('process-images/', ImageProcessAPI.as_view()),
    path('make-3d/', Make3DAPI.as_view()),
    path('depth-status/', DepthStatusAPI.as_view()),
    path('select-output-folder/', SelectOutputFolderAPI.as_view()),
]
