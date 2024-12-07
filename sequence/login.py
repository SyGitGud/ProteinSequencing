import sqlite3
from django.urls import path
from . import views
from django.shortcuts import redirect
import re

# Connect to SQLite database (it will create db.sqlite3 if it doesn't exist)
mydb = sqlite3.connect('db.sqlite3')  # Using SQLite, which creates a file in the project directory
cursor = mydb.cursor()

# Function to sanitize the input username and password
def sanitization(username, password):
    if not username:
        print("Username cannot be empty.")
        return ['','']
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        print("Username can only contain letters, numbers, and underscores.")
        return ['','']
    if not password or len(password) < 8:
        print("Password must be at least 8 characters long.")
        return ['','']
    if not (re.search(r'[A-Z]', password) and
            re.search(r'[a-z]', password) and
            re.search(r'[0-9]', password) and
            re.search(r'[!@#$%^&*(),.?":{}|<>]', password)):
        print("Password must include uppercase, lowercase, a number, and a special character.")
        return ['','']
    
    return [username, password]

# Function to handle the login process
def loginProcess(username, password):
    sanitizedData = sanitization(username, password)
    sanitizedUsername = sanitizedData[0]
    sanitizedPass = sanitizedData[1]
    
    if not sanitizedUsername or not sanitizedPass:
        print("Invalid username or password")
        return False
    
    # SQLite query to check if the user exists
    query = "SELECT * FROM user_logins WHERE user_name = ? AND user_pass = ?"
    user = (sanitizedUsername, sanitizedPass)
    cursor.execute(query, user)
    result = cursor.fetchall()
    
    print("Query executed successfully.")
    print("Result:", result)  
    
    if len(result) > 0:
        print("Right pass")
        return True
    else:
        print("Wrong pass")
        return False

# Function to handle the registration process
def registerProcess(username, password):
    # SQLite query to insert a new user (user role is set as 'user' by default)
    sqlFormula = "INSERT INTO user_logins (user_name, user_pass, user_role) VALUES (?, ?, ?)"
    user = (username, password, "user")
    cursor.execute(sqlFormula, user)
    mydb.commit()  # Commit the transaction to save the changes
    print("User registered successfully!")
