from django.urls import path
from .views import ImageProcessAPI, SelectOutputFolderAPI, upload_page

urlpatterns = [
    path('', upload_page),
    path('process-images/', ImageProcessAPI.as_view()),
    path('select-output-folder/', SelectOutputFolderAPI.as_view()),
]

