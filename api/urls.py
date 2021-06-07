from django.urls import path, include
from django.contrib import admin
from django.urls import path
from api.views import api

urlpatterns = [
    path('addcsv/', api.as_view()),
]
