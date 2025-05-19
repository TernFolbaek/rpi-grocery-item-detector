from django.db import models


class Item(models.Model):
    """
    Model for storing detected items
    """

    name = models.CharField(max_length=255)
    count = models.PositiveIntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} (Count: {self.count})"