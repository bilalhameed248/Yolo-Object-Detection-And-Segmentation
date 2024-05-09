from django.contrib import admin
from django.urls import path
from object_removal.views import upload_file
from .import views

urlpatterns = [
    path('index/', views.home, name='index'),
    path('upload_file/', views.upload_file, name='upload_file'),
]