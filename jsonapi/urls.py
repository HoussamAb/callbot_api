from django.urls import path, include
from django.contrib import admin
from django.urls import path
from jsonapi.views import jsonapi

urlpatterns = [
    path('data/', jsonapi.as_view()),
]
