from django.db import models
from django.contrib.auth.models import User 

    
class History(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    input_sequence = models.TextField()
    prediction_result = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)