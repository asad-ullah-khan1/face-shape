from django.urls import path
from .views import upload_image, api_view

urlpatterns = [
    path('', upload_image, name='upload'),
    path('/api', api_view, name='api'),
]
