from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'), 
    path('login/', views.loginpage, name='loginpage'),  
    path('infopage/', views.infopage, name = 'infopage'),
    path('logout/', views.log_out, name='log_out'),
    path('history/', views.history, name='history'),
    path('delete/', views.delete_account, name='delete_account')
]
