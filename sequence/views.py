from django.shortcuts import render

def home(request):
    return render(request, 'home.html') #send back our home document
