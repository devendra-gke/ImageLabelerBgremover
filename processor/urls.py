from django.urls import path
from .views import ImageProcessAPI, upload_page

urlpatterns = [
    path('', upload_page),
    path('process-images/', ImageProcessAPI.as_view()),
]