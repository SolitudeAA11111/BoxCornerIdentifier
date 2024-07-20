from django.http import HttpResponse
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.viewsets import GenericViewSet
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from django.core.files.base import ContentFile
from drf_spectacular.utils import extend_schema, OpenApiParameter

from app import imagedetector
from .models import Image
from .serializers import ImageSerializer

class ImageProcessingView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    @extend_schema(
        request=ImageSerializer,
        responses={201: ImageSerializer},
        description="Upload an image to process",
    )
    def post(self, request, *args, **kwargs):
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            image_instance = serializer.save()
            original_image_path = image_instance.image.path
            json_response = imagedetector.execute(original_image_path)
            return Response(json_response, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
