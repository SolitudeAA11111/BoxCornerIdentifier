from django.urls import path
from . import views

urlpatterns = [
    path('upload', views.ImageProcessingView.as_view(), name='upload')
]