from rest_framework import serializers
from .models import Item

class ItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = Item
        fields = ['id', 'name', 'count', 'created_at', 'updated_at']
        read_only_fields = ['created_at', 'updated_at']


