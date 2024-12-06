from django.shortcuts import render, redirect
from django.contrib import messages
from .knn_model_for_protein_scaffold.knn_model import predict_sequence, knn_classifier
from . import login

def login_views(request):
    username = ''
    password = ''
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
    user = login.loginProcess(username, password)
    if 'login_button' in request.POST:
        user = login.loginProcess(username, password)
        if(user == True):
            return redirect('home.html')
        else:
            messages.error(request, "Username or Password is incorrect!")
    if 'register_button' in request.POST:
        user=login.registerProcess(username, password)
        if(user==True):
            messages.success(request, "Registration successful please log in.")
        
    return render(request, 'loginpage.html')
def home(request):
    prediction = None  # Default value for prediction

    if request.method == 'POST':
        user_seq = request.POST.get('scaffold')  # Get the sequence input from the form
        
        if user_seq:  # Only process if a sequence is provided
            prediction = gap_fill_sequence(user_seq, knn_classifier)  # Pass the sequence and model

    # Render the home page with the prediction
    return render(request, 'home.html', {'prediction': prediction})
    
