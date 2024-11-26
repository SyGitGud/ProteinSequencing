import mysql.connector
from django.urls import path
from . import views
from django.shortcuts import redirect

mydb = mysql.connector.connect(
    host ="localhost",
    user ="root",
    password ="BuddyBoy1",
    database ="logins"
)

cursor = mydb.cursor()

def loginProcess(username, password):
    userExists = False
    passExists = False
    query = "SELECT user_name FROM user_logins"
    cursor.execute(query)        
    result = cursor.fetchall()
    for x in result:
        if(username == x ):  
            userExists = True
            break
    query = "SELECT user_pass FROM user_logins" 
    cursor.execute(query)
    result = cursor
    for x in result:
        if(password == x):
            userExists = True
            break
    if (userExists == True and passExists == True):
        return True
    if (userExists == False or passExists == False):
        return False
    
        
def registerProcess(username, password):
    # Query for mysql, user role is default for everyuser
    sqlFormula = "INSERT INTO user_logins (user_name, user_pass, user_role) VALUES(%s, %s, %s)"
    user = (username, password, "user")
    #Tells database to execute written query for creating account
    cursor.execute(sqlFormula, user)
    mydb.commit()
