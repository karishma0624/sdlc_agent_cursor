Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "       SDLC AGENT: RESEARCH EVALUATION RUNNER" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# Ensure dependencies
Write-Host "Checking dependencies..."
try {
    pip show matplotlib | Out-Null
}
catch {
    Write-Host "Installing matplotlib..." -ForegroundColor Yellow
    pip install matplotlib
}

Write-Host "Running Benchmark Suite..." -ForegroundColor Green
python backend/research/evaluate.py

Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host "Evaluation Completed. Check 'evaluation_results/' folder." -ForegroundColor White
Write-Host "To view, open the folder using Explorer."
Write-Host "========================================================" -ForegroundColor Cyan
