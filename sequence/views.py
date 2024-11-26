from django.shortcuts import render, redirect
from django.contrib import messages
from . import login

def home(request):
    return render(request, 'home.html') #send back our home document
def login_views(request):
    username = ''
    password = ''
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
    user = login.loginProcess(username, password)
    if(user == True):
        return redirect('home.html')
    else:
        messages.error(request, 'Username or Password is incorrect!')
    return render(request, 'loginpage.html')