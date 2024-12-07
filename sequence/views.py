from django.apps import apps
from .apps import SequenceConfig  # Import from apps
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from .knn_model_for_protein_scaffold.knn_model import predict_sequence, loaded_model
from .models import History

def loginpage(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        if 'login_button' in request.POST:
            user = authenticate(request, username=username, password=password)
            if user: #check if this user is successfully made and added to Django User model
                login(request, user) #logged in user session
                messages.success(request, f"Welcome back, {user.username}!")
                return redirect('home')  # Redirect to the home view
            else:
                messages.error(request, "Username or Password is incorrect!")
                return render(request, 'loginpage.html')  # Render the login page again

        elif 'register_button' in request.POST:
            # Check if username already exists
            if User.objects.filter(username=username).exists():
                messages.error(request, "Username already exists. Please choose another.")
            else:
                # Create a new user
                registered_user = User.objects.create(username=username)
                registered_user.set_password(password)
                registered_user.save()
                messages.success(request, "Registration successful. Please log in.")
                return redirect('loginpage')  #allow them to login now that there is a user in table
    return render(request, 'loginpage.html')


@login_required
def home(request):
    prediction = None  # Default value for prediction
    history = None

    if request.method == 'POST':
        print("Form submitted")
        user_seq = request.POST.get('scaffold')  # Get the sequence input from the form
        
        if user_seq:
            print(user_seq)
            config_seq = apps.get_app_config('sequence')
            knn_classifier, CHAR_TO_INT, INT_TO_CHAR =  config_seq.get_knn_model()
            prediction = predict_sequence(user_seq, knn_classifier, CHAR_TO_INT, INT_TO_CHAR)
            if prediction:
                History.objects.create(user = request.user, input_sequence=user_seq, prediction_result=prediction)
                print(f"Predicted Sequence: {prediction}")  
    
    
    # Home page with prediction (if applicable)
    return render(request, 'home.html', {'prediction': prediction})

def infopage(request):

    if 'home_button' in request.POST:
        return redirect('home')
    if 'login_button' in request.POST:
        return redirect('loginpage')
    return render(request, 'information_page.html')
