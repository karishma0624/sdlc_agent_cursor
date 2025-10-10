Autonomous SDLC Agent (FastAPI + Streamlit + v0.dev Frontend Generator)

Overview
- End-to-end agent that executes the SDLC: requirements → design → build → test → deploy → docs.
- Multi-provider routing with fallbacks (OpenAI, Gemini, Mistral, Groq, HF, Perplexity, Ollama). Safe, runnable local baseline.
- Backend FastAPI, Streamlit control UI, plus generated frontend (via v0.dev or Vite + Tailwind fallback) inside `runs/`.

Quickstart
1) Create a venv and install deps (Windows PowerShell):
   - python -m venv .venv && .venv\Scripts\Activate
   - pip install -r requirements.txt
2) (Optional) Create `.env` and add provider keys. Safe to leave empty for local baseline.
3) Run services:
   - Backend: uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   - Streamlit UI: streamlit run frontend/app.py

Docker
- docker compose up --build

APIs (base = http://localhost:8000)
- GET `/` — API index (endpoints + docs links)
- GET `/health` — Health check
- GET `/providers` — Provider availability
- GET `/logs` — Recent logs
- POST `/predict` — Image classification demo (multipart: file, notes)
- POST `/task` — Generic text/code task { prompt }
- POST `/build` — High-level build orchestrator { prompt }
- POST `/sdlc/build` — Async full SDLC builder { prompt }
- GET `/sdlc/status` — Latest or specific job status (?job_id=...)
- GET `/sdlc/report` — Fetch run_report.json (?job_id=... or ?run_dir=...)
- POST `/dispatch` — Classify and run best-fit tool { prompt, free_only }
- POST `/materialize` — Write generated files to runs/ { prompt, free_only }
- POST `/diagnostics` — Run pytest and flake8
- POST `/save_spec` — Persist requirements markdown { content, filename? }
- CRUD: Students, Events, Requirements

What gets generated in a build (example `runs/YYYYMMDD-HHMMSS-<slug>/`)
- `backend/`
  - `main.py`, `requirements.txt`, `routes/`, `models/`, `services/`, `tests/`, `tests/test_health.py`
- `frontend/` (v0.dev preferred; otherwise Vite + Tailwind scaffold)
  - `package.json`, `index.html`, `vite.config.js`, `tailwind.config.js`, `postcss.config.js`
  - `src/main.jsx`, `src/index.css`, `src/App.jsx` (Home shows API status, providers, API list)
- `requirements/` — `requirements.md`, `requirements.json`
- `docs/` — `index.md`, plus `README.md`, `mkdocs.yml`
- `pytest.ini`, `Dockerfile`, `docker-compose.yml`, `run_report.json`

How to run the generated frontend
```bash
cd runs/<latest-run>/frontend
npm install
npm run dev
```

Diagnostics
```bash
cd runs/<latest-run>
pytest -q
flake8
```

Docs
```bash
cd runs/<latest-run>
pip install mkdocs
mkdocs serve
```

Docker (combined)
```bash
cd runs/<latest-run>
docker compose up --build
```

Environment variables (optional)
- VITE_API_BASE (frontend): default `http://localhost:8000`
- Provider keys: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`, `GROQ_API_KEY`, `HUGGINGFACE_API_KEY`/`HF_API_KEY`, `PERPLEXITY_API_KEY`, `V0_API_KEY`/`V0_DEV_API_KEY`, `OLLAMA_BASE_URL`

Notes
- The builder prefers v0.dev for a polished frontend; any missing files are filled with a Vite + Tailwind scaffold so the app is always runnable.
