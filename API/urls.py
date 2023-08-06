from django.urls import path
from . import views

urlpatterns = [
    path('train/', views.train_model, name='train_model'),
    path('text-status/', views.get_text_status, name='get_text_status'),
]
