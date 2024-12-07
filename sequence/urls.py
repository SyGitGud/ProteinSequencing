from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'), 
    path('login/', views.loginpage, name='loginpage'),  
    path('infopage/', views.infopage, name = 'infopage')
]
