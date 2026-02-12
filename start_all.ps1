# SDLC Agent - Quick Start Script
# This script starts both backend and frontend servers

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SDLC Agent - Starting Services" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if .env exists
if (-not (Test-Path ".env")) {
    Write-Host "❌ ERROR: .env file not found!" -ForegroundColor Red
    Write-Host "Please create .env file with required API keys." -ForegroundColor Yellow
    exit 1
}

# Check required keys
$envContent = Get-Content .env -Raw
$requiredKeys = @("MISTRAL_API_KEY", "GEMINI_API_KEY", "V0_API_KEY")
$missingKeys = @()


foreach ($key in $requiredKeys) {
    if ($envContent -notmatch "$key=\w+") {
        $missingKeys += $key
    }
}

if ($missingKeys.Count -gt 0) {
    Write-Host "❌ ERROR: Missing required API keys:" -ForegroundColor Red
    foreach ($key in $missingKeys) {
        Write-Host "  - $key" -ForegroundColor Yellow
    }
    exit 1
}

Write-Host "✅ Environment configuration verified" -ForegroundColor Green
Write-Host ""

# Start Backend
Write-Host "🚀 Starting Backend Server..." -ForegroundColor Cyan
$backendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    if (Test-Path ".venv\Scripts\Activate.ps1") {
        & .\.venv\Scripts\Activate.ps1
    }
    Set-Location backend
    python main.py
}

Write-Host "   Backend starting (Job ID: $($backendJob.Id))" -ForegroundColor Gray
Start-Sleep -Seconds 3

# Start Frontend
Write-Host "🚀 Starting Frontend Server..." -ForegroundColor Cyan
$frontendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    Set-Location frontend
    npm run dev
}

Write-Host "   Frontend starting (Job ID: $($frontendJob.Id))" -ForegroundColor Gray
Start-Sleep -Seconds 3

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  ✅ Services Started Successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "📍 Access Points:" -ForegroundColor Cyan
Write-Host "   Frontend:  http://localhost:5173" -ForegroundColor White
Write-Host "   Backend:   http://localhost:8000" -ForegroundColor White
Write-Host "   API Docs:  http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "📊 Provider Status:" -ForegroundColor Cyan
Write-Host "   Check: http://localhost:8000/providers" -ForegroundColor White
Write-Host ""
Write-Host "⚠️  Press Ctrl+C to stop all services" -ForegroundColor Yellow
Write-Host ""

# Monitor jobs
try {
    while ($true) {
        Start-Sleep -Seconds 5
        
        # Check if jobs are still running
        $backendStatus = Get-Job -Id $backendJob.Id
        $frontendStatus = Get-Job -Id $frontendJob.Id
        
        if ($backendStatus.State -eq "Failed") {
            Write-Host "❌ Backend crashed!" -ForegroundColor Red
            Receive-Job -Id $backendJob.Id
            break
        }
        
        if ($frontendStatus.State -eq "Failed") {
            Write-Host "❌ Frontend crashed!" -ForegroundColor Red
            Receive-Job -Id $frontendJob.Id
            break
        }
    }
}
finally {
    Write-Host ""
    Write-Host "🛑 Stopping services..." -ForegroundColor Yellow
    Stop-Job -Id $backendJob.Id -ErrorAction SilentlyContinue
    Stop-Job -Id $frontendJob.Id -ErrorAction SilentlyContinue
    Remove-Job -Id $backendJob.Id -ErrorAction SilentlyContinue
    Remove-Job -Id $frontendJob.Id -ErrorAction SilentlyContinue
    Write-Host "✅ All services stopped" -ForegroundColor Green
}
