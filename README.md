# ProteinSequencing
Our senior project where we use an AI transformer model to take an incomplete protein scaffold and fill in the scaffold gaps

## Advisors
- Dr. Letu Qingge
- TA Kushal Badal

## Instructions:
1. Clone the repository or download and extract the zip file.
2. Navigate to project directory in command line of choice. 
3. Create and activate a virtual environment
4. Install required packages using 'pip install -r requirements.txt'
5. Set up environemnt variables:
  - ### For Linux/Mac, Run:
        chmod +x gen_env.sh
        ./gen_env.sh
  - ### For Windows, Run:
        .gen_env.p21
6. ### Run Migrations and run the server:
        python manage.py migrate
        python manage.py runserver
