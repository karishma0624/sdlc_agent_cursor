@echo off
echo Starting SDLC Agent Streamlit Interface...
echo.
echo Make sure your API server is running on port 8080
echo If not, start it with: python -m uvicorn backend.main:app --host 127.0.0.1 --port 8080 --reload
echo.
streamlit run streamlit_app.py --server.port 8501 --server.address 127.0.0.1
pause
