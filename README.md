Autonomous SDLC Agent (Streamlit + FastAPI)

Overview
- End-to-end agent that takes a natural language task and executes SDLC: requirements → design → build → test → deploy → docs.
- Multi-provider model routing with fallbacks (OpenAI, Gemini, Mistral, Groq, Hugging Face, Perplexity, Ollama) with a safe local baseline for offline/dev.
- Backend FastAPI, Frontend Streamlit, SQLite for app DB, lightweight RAG memory.

Quickstart
1) Create virtual environment and install:
   - Windows PowerShell:
     - python -m venv .venv && .venv\Scripts\Activate
     - pip install -r requirements.txt
2) Copy .env.example to .env and fill keys as available (safe to leave empty to use local baseline).
3) Run services:
   - uvicorn backend.main:app --reload
   - streamlit run frontend/app.py

Docker
- docker compose up --build

Project Layout
- backend/: FastAPI app, adapters, DB, RAG
- frontend/: Streamlit UI
- tests/: basic API tests

Environment
- See .env.example. Never commit real keys.

Commands
- Setup: pip install -r requirements.txt
- Run API: uvicorn backend.main:app --reload
- Run UI: streamlit run frontend/app.py
- Test: pytest -q
- Docker: docker compose up --build

Notes
- The adapters route to cloud providers if keys are present; otherwise fallback to a simple local baseline classifier. This keeps the project runnable out-of-the-box.


