from django.apps import apps
from .apps import SequenceConfig  # Import from apps
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.password_validation import validate_password
from django.contrib.auth.models import User
from django.http import HttpResponse
import csv
from django.contrib.auth.decorators import login_required
from .knn_model_for_protein_scaffold.knn_model import predict_seq, calc_accuracy
from .models import History


def log_out(request):
    logout(request)
    return redirect(loginpage)

def loginpage(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        if 'login_button' in request.POST:
            user = authenticate(request, username=username, password=password) #authenticat them
            if user: #check if this user is successfully made and added to Django User model
                login(request, user) #logged in user session use django built in
                return redirect('home')  # Redirect to the home view
            else:
                messages.error(request, "Username or Password is incorrect!")
                return render(request, 'loginpage.html')  # Render the login page again

        elif 'register_button' in request.POST:
            # Check if username already exists
            if User.objects.filter(username=username).exists():
                messages.error(request, "Username already exists. Please choose another.")
            else:
                try:
                    validate_password(password)
                    # Create a new user
                    registered_user = User.objects.create(username=username) #pass by object in Django to prevent sql injection
                    registered_user.set_password(password) #have to use set password for encryption and safety
                    registered_user.save() #save the user and maintain the user sessions
                    messages.success(request, "Registration successful. Please log in.")
                    return redirect('loginpage')  #allow them to login now that there is a user in table
                except Exception as e:
                    messages.error(request, f"Password Denied: {e}")
    return render(request, 'loginpage.html')


def history(request): #downloading history and clearing it
    user = request.user

    if request.method == 'POST':
        hist = request.POST.get('hist') #getting the values by name
        if 'Download' in hist: 
            history_file = HttpResponse(content_type='text/csv')
            history_file['Content-Disposition'] = 'attachment; filename="scaffold_history.csv"' #name and create a csv file
            csv_edit = csv.writer(history_file)
            csv_edit.writerow(['timestamp', 'gap scaffold', 'filled sequence'])
            records = History.objects.filter(user=user).all()

            for record in records:
                csv_edit.writerow([record.created_at, record.input_sequence, record.prediction_result]) #populate the csv file
            
            return history_file

        if 'Clear' in hist:
                records = History.objects.filter(user=user) #delete all of that user's entries
                records.delete()
                return redirect('home')


@login_required
def delete_account(request):
        user = request.user
        user.delete()
        return redirect('log_out')




@login_required
def home(request):
    prediction = None  # Default value for prediction
    accurate = None
    train_accuracy = None
    history = None
    user_seq = None




    if request.method == 'POST':
        print("Form submitted")
        user_seq = request.POST.get('scaffold')  # Get the sequence input from the form
        
        if user_seq:
            print(user_seq)
            prediction = predict_seq(user_seq)
            if prediction:
                History.objects.create(user = request.user, input_sequence=user_seq, prediction_result=prediction)
                print(f"Predicted Sequence: {prediction}")
                train_accuracy, accurate = calc_accuracy()
                train_accuracy = f"{train_accuracy *100:.2f}%"
                accurate = f"{accurate*100:.2f}%" 
   





    return render(request, 'home.html', {'prediction': prediction, 'train_accuracy': train_accuracy, 'accurate': accurate, 'scaffold': user_seq, 'user': request.user.username})
    
    # Home page with prediction (if applicable)


def infopage(request): #information page

    if 'home_button' in request.POST:
        return redirect('home')
    if 'login_button' in request.POST:
        return redirect('loginpage')
    return render(request, 'information_page.html')
