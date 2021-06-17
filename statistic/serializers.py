from django.core import serializers

from models import statistic

class StatisticSerializer(serializers.ModelSerializer):
    class Meta:
        model = statistic
        fields = ["task", "completed", "timestamp", "updated", "algoName","trainSize","testSize","Score","NLPmethode","details"]
