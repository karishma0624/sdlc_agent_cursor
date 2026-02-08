
Write-Host "Starting SDLC Frontend..."
Set-Location frontend

if (-not (Test-Path "node_modules")) {
    Write-Host "Installing dependencies... (this may take a minute)"
    npm install
}

npm run dev
