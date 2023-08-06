from django.db import models

class TextEntry(models.Model):
    text = models.TextField()
    status = models.CharField(max_length=255)