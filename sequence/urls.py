from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'), 
    path('', views.login_views, name = 'loginpage'),
    path('', views.infopage, name = 'infopage')
]