from django.urls import path
from . import views

urlpatterns = [
    path('parse_image', views.ImageProcessingView.as_view(), name='parse_image')
]