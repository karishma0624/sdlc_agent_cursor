
Write-Host "Starting SDLC Backend..."

# Ensure we are in the project root (relative to this script)
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptPath

# Add backend to PYTHONPATH so 'import simple_builder' works from main.py
# We add both absolute path to backend and current root
$env:PYTHONPATH = "$scriptPath\backend;$scriptPath"

# Activate venv
if (Test-Path ".venv") {
    Write-Host "Activating root venv..."
    & .venv\Scripts\Activate.ps1
} elseif (Test-Path "backend\.venv") {
     Write-Host "Activating backend venv..."
    & backend\.venv\Scripts\Activate.ps1
}

Write-Host "Running Backend Server..."
# We run backend.main so uvicorn finds it. CWD stays at root so 'runs/' calls go to root.
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
