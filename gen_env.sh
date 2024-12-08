#!/bin/bash
SECRET_KEY=$(python -c "from django.core.management.utils import get_random_secret_key; 
get_random_secret_key()")
echo "SECRET_KEY=$SECRET_KEY" > .env
