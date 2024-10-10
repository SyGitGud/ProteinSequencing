from django.contrib import admin
from django.urls import path, include  # Make sure to include include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('sequence/', include('sequence.urls')),  # Include the sequence app URLs
    path('', include('sequence.urls')),  # Redirect root to sequence app (if you have views for it)
]