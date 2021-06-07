from django.core import serializers

from models import api

class apiSerializer(serializers.ModelSerializer):
    class Meta:
        model = api
        fields = ["task", "completed", "timestamp", "updated", "user"]