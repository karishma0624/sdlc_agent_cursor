
Write-Host "Launching Development Environment..."

# Get the script's directory (project root)
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition

Write-Host "Project Root: $scriptPath"

# Launch Backend in a new window
# We pass full path to script and set working directory explicitly
Start-Process powershell -ArgumentList "-NoExit", "-File", "$scriptPath\run_backend.ps1" -WorkingDirectory $scriptPath

# Launch Frontend in a new window
Start-Process powershell -ArgumentList "-NoExit", "-File", "$scriptPath\run_frontend.ps1" -WorkingDirectory $scriptPath

Write-Host "Backend and Frontend are starting in new windows."
Write-Host "Backend URL: http://localhost:8000"
Write-Host "Frontend URL: http://localhost:5173 (or 3000 depending on config)"
