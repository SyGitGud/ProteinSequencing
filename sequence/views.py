from django.shortcuts import render, redirect
from django.contrib import messages
from .knn_model import predict_sequence, gap_fill_sequence, knn_model
from . import login

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
def home(request):
    prediction = None  # Default value for prediction

    if request.method == 'POST':
        user_seq = request.POST.get('user_seq')  # Get the sequence input from the form
        
        if user_sequence:  # Only process if a sequence is provided
            # Use the gap_fill_sequence function to predict the filled sequence
            prediction = gap_fill_sequence(user_seq, knn_model)  # Pass the sequence and model

    # Render the home page with the prediction
    return render(request, 'home.html', {'prediction': prediction})
