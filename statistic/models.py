from django.db import models


# Create your models here.
class statistic(models.Model):
    task = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True, auto_now=False, blank=True)
    completed = models.BooleanField(default=False, blank=True)
    updated = models.DateTimeField(auto_now=True, blank=True)
    algoName = models.CharField(max_length=255)
    trainSize = models.IntegerField()
    testSize = models.IntegerField()
    Score = models.IntegerField()
    NLPmethode = models.CharField(max_length=255)
    details = models.CharField(max_length=255)


def __str__(self):
    return self.task
