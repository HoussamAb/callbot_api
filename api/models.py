from django.db import models


# Create your models here.
class api(models.Model):
    task = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True, auto_now=False, blank=True)
    completed = models.BooleanField(default=False, blank=True)
    updated = models.DateTimeField(auto_now=True, blank=True)
    questions = models.CharField(max_length=255)
    reponses = models.CharField(max_length=255)


def __str__(self):
    return self.task
