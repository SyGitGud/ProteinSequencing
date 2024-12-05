import mysql.connector
from django.urls import path
from . import views
from django.shortcuts import redirect
import re

mydb = mysql.connector.connect(
    host ="localhost",
    user ="root",
    password ="BuddyBoy1",
    database ="logins"
)

cursor = mydb.cursor()

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

def loginProcess(username, password):
    sanitizedData = sanitization(username, password)
    sanitizedUsername = sanitizedData[0]
    sanitizedPass = sanitizedData[1]
    
    if not sanitizedUsername or not sanitizedPass:
        print("Invalid username or password")
        return False
    query =  query = "SELECT * FROM user_logins WHERE username = %s AND password = %s"
    user = (sanitizedUsername, sanitizedPass)
    cursor.execute(query, user)
    result = cursor.fetchall()
    print("Query executed successfully.")
    print("Result:", result)  
          
    if(len(result) > 0):
        print("Right pass")
        return True
    else:
        print("wrong Pass")
        return False
    
        
def registerProcess(username, password):
    # Query for mysql, user role is default for everyuser
    sqlFormula = "INSERT INTO user_logins (user_name, user_pass, user_role) VALUES(%s, %s, %s)"
    user = (username, password, "user")
    #Tells database to execute written query for creating account
    cursor.execute(sqlFormula, user)
    mydb.commit()
