from django.db import models


# Create your models here.
class statistic(models.Model):
    task = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True, auto_now=False, blank=True)
    completed = models.BooleanField(default=False, blank=True)
    updated = models.DateTimeField(auto_now=True, blank=True)
    algoName = models.CharField(max_length=255)
    trainSize = models.IntegerField(blank=True)
    testSize = models.IntegerField(blank=True)
    Score = models.IntegerField(blank=True)
    NLPmethode = models.CharField(max_length=255, blank=True)
    details = models.CharField(max_length=255, blank=True)


def __str__(self):
    return self.task
