$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    py -3 -m venv .venv
}

$python = Join-Path ".venv" "Scripts\python.exe"
& $python -m pip install --quiet --upgrade pip
& $python -m pip install --quiet -r requirements.txt

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

Write-Host "Starting Catalyst Propagation at http://127.0.0.1:8000"
& $python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
