@echo off
echo Starting SDLC Agent Services...

:: Start Backend
start "Backend API" cmd /k "call .venv\Scripts\activate & python -m uvicorn backend.main:app --reload --port 8000"

:: Start Frontend
start "Frontend UI" cmd /k "call .venv\Scripts\activate & streamlit run frontend/app.py"

echo Services started in new windows.
echo Frontend: http://localhost:8501
echo Backend: http://localhost:8000
pause
