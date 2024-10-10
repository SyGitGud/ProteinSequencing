from django.shortcuts import render
from django.http import HttpResponse

def home(request):
    return home(request, 'home.html') #send back our home document
