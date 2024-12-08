Write-Host "secret key being made!..."
$SECRET_KEY = python -c "from django.core.management.utils import get_random_secret_key; get_random_secret_key()"
Set-Content -Path .env -Value "SECRET_KEY=$SECRET_KEY"
