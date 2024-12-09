# ProteinSequencing
Our senior project where we use an AI transformer model to take an incomplete protein scaffold and fill in the scaffold gaps

## Advisors
- Dr. Letu Qingge
- TA Kushal Badal

## Instructions:
1. Clone the repository or download and extract the zip file.
2. Navigate to project directory in command line of choice. 
3. Create and activate a virtual environment:
     ```powershell
     #create
   python -m venv venv
     #activate for windows:
     .\venv\Scripts\activate
     # activate for linux/mac
     source venv/bin/activate 
   ```
4. Install required packages using:
   `pip install -r requirements.txt`
5. Set up environment variables:
  - ### For Linux/Mac, Run:
    ```bash
    chmod +x gen_env.sh
    ./gen_env.sh
    ```
  - ### For Windows Powershell, Run:
    ```bash
    ./gen_env.ps1
    ```
6. If there are any errors with steps 4 and 5 and the next step, install the packages below using:
    ```powershell
    # Use if there is an error with setting up the environmental variables
    pip install Django
    # Use the following if there is an error with running migrations
    pip install python-decouple
    pip install numpy
    pip install Scikit-learn
    pip install tqdm
    ```
7. ### Run Migrations and run the server:
        python manage.py migrate
        python manage.py runserver
