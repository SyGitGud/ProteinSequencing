Write-Host "secret key being made!..."
try {
    $SECRET_KEY = python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
    Set-Content -Path .env -Value "SECRET_KEY=$SECRET_KEY"
    Write-Host "SECRET_KEY created."
}
catch {
    Write-Host "UH OH!"
}



