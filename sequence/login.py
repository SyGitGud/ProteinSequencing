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
    ##passExists = False
    query = "SELECT * FROM user_logins WHERE user_pass = %s AND user_pass = %s"
    user = (username, password)
    cursor.execute(query, user)        
    if(len(cursor.fetchall()) > 0):
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
