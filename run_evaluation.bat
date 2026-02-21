@echo off
echo ========================================================
echo        SDLC AGENT: RESEARCH EVALUATION RUNNER
echo ========================================================
echo.
echo Running Benchmark Suite...
echo.
python backend/research/evaluate.py
echo.
echo ========================================================
echo Evaluation Completed. Results in:
echo   - evaluation_results/statistics.json
echo   - evaluation_results/benchmark_summary.json
echo   - evaluation_results/*.png
echo.
echo View the summary log for details on failures/successes.
echo ========================================================
pause
