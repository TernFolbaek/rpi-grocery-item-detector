from rest_framework import generics
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from .models import Item
from .serializers import ItemSerializer

class ItemListView(generics.ListCreateAPIView):
    """
    API endpoint that allows items to be viewed or created
    """
    queryset = Item.objects.all()
    serializer_class = ItemSerializer

class ItemDetailView(generics.RetrieveUpdateDestroyAPIView):
    """
    API endpoint that allows items to be viewed, updated or destroyed
    """
    queryset = Item.objects.all()
    serializer_class = ItemSerializer

@api_view(['POST'])
def detect_item(request):
    """
    Expected JSON format:
    {
        "name" : "Item Name"
    }

    Response
    {
    "id" : int,
    "name" : {item name},
    "count" : {nr of observations of item},
    "created_at" : Date,
    "updated_at" : Date
    }
    """

    name = request.data.get('name')

    if not name:
        return Response({'error': 'Item name is required'})
    
    # Try to find an existing item with the same name in the db

    item = Item.objects.filter(name=name).first()

    if item:
        # If item exists, increment the count
        item.count += 1
        item.save()
    else:
        item = Item.objects.create(name=name)
    
    serializer = ItemSerializer(item)
    return Response(serializer.data)
