from django.urls import path
from .views import ImageProcessAPI, upload_page

urlpatterns = [
    path('', upload_page),
    path('process-images/', ImageProcessAPI.as_view()),
]

# from django.urls import path

# from .views import (
#     upload_page,
#     ParseExcelAPI,
#     SaveProcessedAPI,
# )

# urlpatterns = [

#     path(
#         "",
#         upload_page,
#         name="upload_page"
#     ),

#     path(
#         "parse-excel/",
#         ParseExcelAPI.as_view(),
#         name="parse_excel"
#     ),

#     path(
#         "save-processed/",
#         SaveProcessedAPI.as_view(),
#         name="save_processed"
#     ),
# ]

