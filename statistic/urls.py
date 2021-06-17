from django.urls import path, include
from django.contrib import admin
from django.urls import path
from statistic.views import statistic

urlpatterns = [
    path('statistic/', statistic.as_view()),
]
