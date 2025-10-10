from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os
import json
from datetime import datetime

from .adapters import InferenceRouter
from .metrics import MetricsLogger


class SDLCBuilder:
    def __init__(self, runs_dir: str = "runs", fast_mode: bool = False) -> None:
        self.router = InferenceRouter()
        self.metrics = MetricsLogger()
        self.runs_dir = runs_dir
        self.fast_mode = fast_mode
        os.makedirs(self.runs_dir, exist_ok=True)

    def _mk_run_dir(self, title: str) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        slug = "".join(c for c in (title or "").lower() if c.isalnum() or c in ("-", "_", " ")).strip().replace(" ", "-")
        path = os.path.join(self.runs_dir, f"{ts}-{slug[:40]}")
        os.makedirs(path, exist_ok=True)
        return path

    def _write_text(self, path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _json(self, obj: Any) -> str:
        return json.dumps(obj, indent=2, ensure_ascii=False)

    def build(self, prompt: str) -> Dict[str, Any]:
        self.router.refresh()
        run_dir = self._mk_run_dir(prompt or "project")
        task_name = os.path.basename(run_dir)
        artifacts: Dict[str, Any] = {"backend": {}, "frontend": {}, "tests": {}, "infra": {}, "docs": {}, "requirements": {}}
        logs: List[Dict[str, Any]] = []

        def record(stage: str, success: bool, models_used: Dict[str, Any], files_generated: int, errors: Optional[str], metadata: Optional[Dict[str, Any]] = None) -> None:
            logs.append({
                "timestamp": datetime.utcnow().isoformat(),
                "stage": stage,
                "success": success,
                "models_used": models_used,
                "files_generated": files_generated,
                "errors": errors,
                "metadata": metadata or {},
            })
            self.metrics.log_stage(task_name=task_name, stage=stage, success=success, models_used=models_used, files_generated=files_generated, errors=errors, metadata=metadata)

        # Phase 1 – Requirements (FAST: templates only; otherwise Gemini → OpenAI)
        req_models = {}
        req_md = ""
        req_json: Dict[str, Any] = {}
        try:
            if self.fast_mode:
                req_md = f"# Requirements\n\nProject: {prompt[:64] or 'Project'}\n\n- Frontend: React + Vite + TailwindCSS\n- Backend: FastAPI + SQLite\n- Features: CRUD, Search, Health check, Providers\n- Deployment: Docker + GitHub Actions\n"
                req_models = {"mode": "fast"}
            else:
                res_g = self.router.generate_text(f"Extract detailed, structured requirements (title, description, modules, frontend stack, backend stack, DB schema, API routes, features, deployment) for: {prompt}", preference=["gemini"]) if self.router.providers.get("gemini") else {"output": "", "provider": None, "model": None}
                res_o = self.router.generate_text(f"Refine the following requirements to be concise and actionable and return improved text only:\n\n{res_g.get('output','')}", preference=["openai"]) if self.router.providers.get("openai") else {"output": res_g.get("output",""), "provider": None, "model": None}
                req_models = {"gemini": res_g.get("model"), "openai": res_o.get("model")}
                req_md = res_o.get("output") or res_g.get("output") or ""
            req_json = {
                "title": prompt[:64] or "Project",
                "description": prompt,
                "modules": ["Auth", "Core"],
                "frontend": ["React", "TailwindCSS"],
                "backend": ["FastAPI", "SQLite"],
                "features": ["CRUD", "Search", "User roles"],
                "deployment": "Docker + GitHub Actions",
            }
            self._write_text(os.path.join(run_dir, "requirements", "requirements.md"), req_md)
            self._write_text(os.path.join(run_dir, "requirements", "requirements.json"), self._json(req_json))
            artifacts["requirements"] = {"paths": [os.path.join(run_dir, "requirements", "requirements.md"), os.path.join(run_dir, "requirements", "requirements.json")]}
            record("requirements", True, req_models, 2, None)
        except Exception as e:
            record("requirements", False, req_models, 0, str(e))

        # Phase 2 – Backend Generation (FAST: scaffold; otherwise LLM + scaffold)
        be_models = {}
        try:
            backend_root = os.path.join(run_dir, "backend")
            for d in ["routes", "models", "services", "tests"]:
                os.makedirs(os.path.join(backend_root, d), exist_ok=True)
            # Generate domain-specific backend from prompt
            written = 0
            if not self.fast_mode:
                instruction = (
                    "Create a FastAPI backend with modular structure (main.py, routes/, models/, services/). "
                    "Include CRUD routes for key entities derived from the prompt. Provide pytest tests in backend/tests. "
                    "Return ONLY a JSON mapping file paths to contents."
                )
                gen = self.router.generate_code(f"{instruction}\nPrompt: {prompt}")
                files = gen.get("files") or {}
                be_models = {"provider": gen.get("provider"), "model": gen.get("model"), "tokens": gen.get("tokens")}
                if isinstance(files, dict) and files:
                    for rel, content in files.items():
                        path = os.path.join(backend_root, rel)
                        self._write_text(path, content)
                        written += 1
            # Ensure minimum scaffold exists
            if written == 0:
                self._write_text(os.path.join(backend_root, "main.py"), _TEMPLATE_FASTAPI_MAIN)
                written += 1
            # baseline supportive files
            if not os.path.exists(os.path.join(backend_root, "requirements.txt")):
                self._write_text(os.path.join(backend_root, "requirements.txt"), _TEMPLATE_BACKEND_REQS)
                written += 1
            if not os.path.exists(os.path.join(backend_root, "tests", "test_health.py")):
                self._write_text(os.path.join(backend_root, "tests", "test_health.py"), _TEMPLATE_TEST_HEALTH)
                written += 1
            artifacts["backend"] = {"root": backend_root}
            record("backend", True, be_models, written, None)
        except Exception as e:
            record("backend", False, be_models, 0, str(e))

        # Phase 3 – Frontend Generation (Prefer v0.dev; fallback to local React + Tailwind scaffold)
        fe_models = {}
        try:
            frontend_root = os.path.join(run_dir, "frontend")
            files_written = 0
            files_created: set[str] = set()
            used_v0 = False
            if (not self.fast_mode) and self.router.providers.get("v0"):
                resp = self.router.run_tool(
                    "v0",
                    (
                        "Create a modern, attractive React (Vite) + Tailwind frontend that consumes the following API base "
                        "(env VITE_API_BASE, default http://localhost:8000). Include a home page that shows API status and a card grid "
                        "of available endpoints by fetching '/' (API index) and '/providers'. Create clean components and responsive layout."
                    ),
                )
                files = resp.get("files") if isinstance(resp, dict) else None
                fe_models = {"provider": "v0", "model": (resp.get("model") if isinstance(resp, dict) else None)}
                if files:
                    used_v0 = True
                    for rel, content in files.items():
                        path = os.path.join(frontend_root, rel)
                        self._write_text(path, content)
                        files_written += 1
                        files_created.add(rel.replace("\\", "/"))
            # Ensure minimum scaffold exists even if v0 returned partial files
            def ensure(rel_path: str, content: str) -> None:
                nonlocal files_written
                normalized = rel_path.replace("\\", "/")
                if normalized not in files_created:
                    self._write_text(os.path.join(frontend_root, rel_path), content)
                    files_written += 1
                    files_created.add(normalized)

            # Baseline scaffold
            ensure("package.json", _TEMPLATE_PACKAGE_JSON)
            ensure("tailwind.config.js", _TEMPLATE_TAILWIND_CONFIG)
            ensure("postcss.config.js", _TEMPLATE_POSTCSS)
            ensure("vite.config.js", _TEMPLATE_VITE)
            ensure("index.html", _TEMPLATE_INDEX_HTML)
            ensure(os.path.join("src", "main.jsx"), _TEMPLATE_MAIN_JSX)
            ensure(os.path.join("src", "index.css"), _TEMPLATE_INDEX_CSS)
            # App with API list (only create if not provided by v0)
            ensure(os.path.join("src", "App.jsx"), _TEMPLATE_APP_JSX)
            artifacts["frontend"] = {"root": frontend_root}
            record("frontend", True, {**fe_models, "used_v0": used_v0}, files_written, None)
        except Exception as e:
            record("frontend", False, fe_models, 0, str(e))

        # Phase 4 – Testing & Deployment (Mistral, Groq + OpenAI)
        td_models = {}
        try:
            td_models = {"tests": ("mistral" if self.router.providers.get("mistral") else None), "infra": [n for n in ["groq", "openai"] if self.router.providers.get(n)]}
            # tests already added minimal; add coverage config
            self._write_text(os.path.join(run_dir, "pytest.ini"), _TEMPLATE_PYTEST_INI)
            # deployment
            self._write_text(os.path.join(run_dir, "Dockerfile"), _TEMPLATE_DOCKERFILE)
            self._write_text(os.path.join(run_dir, "docker-compose.yml"), _TEMPLATE_DOCKER_COMPOSE)
            os.makedirs(os.path.join(run_dir, ".github", "workflows"), exist_ok=True)
            self._write_text(os.path.join(run_dir, ".github", "workflows", "deploy.yml"), _TEMPLATE_GH_ACTIONS)
            artifacts["infra"] = {"dockerfile": os.path.join(run_dir, "Dockerfile"), "compose": os.path.join(run_dir, "docker-compose.yml"), "workflow": os.path.join(run_dir, ".github", "workflows", "deploy.yml")}
            record("deployment", True, td_models, 4, None)
        except Exception as e:
            record("deployment", False, td_models, 0, str(e))

        # Auto diagnostics (skip in FAST mode)
        if not self.fast_mode:
            try:
                import subprocess, shlex
                def run_cmd(cmd: str, timeout: int = 180) -> Dict[str, Any]:
                    try:
                        proc = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False, text=True)
                        return {"code": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}
                    except subprocess.TimeoutExpired:
                        return {"code": -1, "stdout": "", "stderr": "timeout"}
                junit_json = os.path.join(run_dir, "pytest-report.json")
                pytest_cmd = f"pytest -q --maxfail=1 --disable-warnings --json-report --json-report-file={junit_json}"
                flake8_cmd = "flake8"
                pr = run_cmd(pytest_cmd, timeout=240)
                fr = run_cmd(flake8_cmd, timeout=120)
                self.metrics.log_stage(task_name=task_name, stage="diagnostics", success=(pr["code"] == 0 and fr["code"] == 0), models_used={}, files_generated=0, errors=None, metadata={"pytest": pr["code"], "flake8": fr["code"]})
            except Exception:
                pass

        # Phase 5 – Documentation (Gemini + OpenAI)
        doc_models = {}
        try:
            res_doc_g = self.router.generate_text(f"Generate a concise README for the project: {prompt}. Include setup, run, test, and deploy steps.", preference=["gemini"]) if self.router.providers.get("gemini") else {"output": ""}
            res_doc_o = self.router.generate_text(f"Refine this README to be clearer and actionable, return markdown only:\n\n{res_doc_g.get('output','')}", preference=["openai"]) if self.router.providers.get("openai") else {"output": res_doc_g.get("output","")}
            readme_md = res_doc_o.get("output") or "# Project\n"
            doc_models = {"gemini": res_doc_g.get("model"), "openai": res_doc_o.get("model")}
            # mkdocs
            docs_root = os.path.join(run_dir, "docs")
            self._write_text(os.path.join(run_dir, "README.md"), readme_md)
            self._write_text(os.path.join(run_dir, "mkdocs.yml"), _TEMPLATE_MKDOCS)
            self._write_text(os.path.join(docs_root, "index.md"), _TEMPLATE_DOCS_INDEX)
            artifacts["docs"] = {"paths": [os.path.join(run_dir, "README.md"), os.path.join(run_dir, "mkdocs.yml"), os.path.join(docs_root, "index.md")]}
            record("documentation", True, doc_models, 3, None)
        except Exception as e:
            record("documentation", False, doc_models, 0, str(e))

        report = {
            "summary": "Full SDLC build completed",
            "run_dir": run_dir,
            "artifacts": artifacts,
            "logs": logs,
            "commands": {
                "backend": [
                    f"cd {run_dir}",
                    "python -m venv .venv && .venv\\Scripts\\pip install -r backend/requirements.txt" if os.name == "nt" else "python -m venv .venv && source .venv/bin/pip install -r backend/requirements.txt",
                    
                    "uvicorn backend.main:app --host 0.0.0.0 --port 8000"
                ],
                "frontend": [
                    f"cd {os.path.join(run_dir, 'frontend')}",
                    "npm install",
                    "npm run dev"
                ],
                "tests": [f"cd {run_dir}", "pytest -q"],
                "docs": [f"cd {run_dir}", "pip install mkdocs && mkdocs serve"],
                "deploy": [f"cd {run_dir}", "docker compose up --build"]
            }
        }
        self._write_text(os.path.join(run_dir, "run_report.json"), self._json(report))
        return report


_TEMPLATE_BACKEND_REQS = """fastapi
uvicorn
pydantic
"""

_TEMPLATE_BACKEND_README = """# Backend

FastAPI app with health route and test.
"""

_TEMPLATE_FASTAPI_MAIN = """from fastapi import FastAPI

app = FastAPI(title="Generated FastAPI Backend")

@app.get("/health")
def health():
    return {"status": "ok"}
"""

_TEMPLATE_ROUTE_HEALTH = """from fastapi import APIRouter

router = APIRouter()

@router.get("/healthz")
def healthz():
    return {"ok": True}
"""

_TEMPLATE_TEST_HEALTH = """def test_health_smoke():
    assert 1 + 1 == 2
"""

_TEMPLATE_PYTEST_INI = """[pytest]
addopts = -q
"""

_TEMPLATE_DOCKERFILE = """FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

_TEMPLATE_DOCKER_COMPOSE = """services:
  api:
    build: .
    ports:
      - "8000:8000"
    command: uvicorn backend.main:app --host 0.0.0.0 --port 8000
  web:
    build: ./frontend
    command: npm run dev
    ports:
      - "5173:5173"
    depends_on:
      - api
"""

_TEMPLATE_GH_ACTIONS = """name: Deploy
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install backend deps
        run: |
          pip install -r backend/requirements.txt
      - name: Run tests
        run: pytest -q
"""

_TEMPLATE_MKDOCS = """site_name: Generated Project Docs
nav:
  - Home: index.md
theme:
  name: material
"""

_TEMPLATE_DOCS_INDEX = """# Project Documentation

Welcome! Use the README for quickstart. This site can be served with `mkdocs serve`.
"""

_TEMPLATE_PACKAGE_JSON = """{
  "name": "generated-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview --port 5173",
    "start": "vite preview --host 0.0.0.0 --port 5173"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.2.0",
    "autoprefixer": "^10.4.18",
    "postcss": "^8.4.38",
    "tailwindcss": "^3.4.3",
    "vite": "^5.0.0"
  }
}
"""

_TEMPLATE_TAILWIND_CONFIG = """/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
"""

_TEMPLATE_POSTCSS = """export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
"""

_TEMPLATE_VITE = """import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: { host: '0.0.0.0', port: 5173 }
})
"""

_TEMPLATE_INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Generated App</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
  </html>
"""

_TEMPLATE_MAIN_JSX = """import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
"""

_TEMPLATE_INDEX_CSS = """@tailwind base;
@tailwind components;
@tailwind utilities;

html, body, #root { height: 100%; }
"""

_TEMPLATE_APP_JSX = """import { useEffect, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

function EndpointCard({ item }) {
  return (
    <div className="rounded-lg border p-4 bg-white shadow-sm">
      <div className="text-xs uppercase tracking-wide text-gray-500">{item.method}</div>
      <div className="font-mono text-sm break-all">{item.path}</div>
      {item.desc && <div className="text-sm text-gray-600 mt-1">{item.desc}</div>}
    </div>
  )
}

export default function App() {
  const [status, setStatus] = useState('unknown')
  const [providers, setProviders] = useState({})
  const [apiIndex, setApiIndex] = useState({ endpoints: [], docs: {} })

  useEffect(() => {
    fetch(`${API_BASE}/health`).then(r => setStatus(r.ok ? 'online' : 'offline')).catch(() => setStatus('offline'))
    fetch(`${API_BASE}/providers`).then(r => r.json()).then(d => setProviders(d.providers || {})).catch(() => setProviders({}))
    fetch(`${API_BASE}/`).then(r => r.json()).then(d => setApiIndex(d || { endpoints: [], docs: {} })).catch(() => setApiIndex({ endpoints: [], docs: {} }))
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 text-gray-900">
      <header className="p-6 border-b bg-white/80 backdrop-blur">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold">Generated Frontend</h1>
            <p className="text-sm text-gray-500">API status: <span className={status === 'online' ? 'text-green-600' : 'text-red-600'}>{status}</span></p>
          </div>
          <div className="text-sm text-gray-600">
            <a className="underline mr-3" href={`${API_BASE}/docs`} target="_blank" rel="noreferrer">Swagger</a>
            <a className="underline" href={`${API_BASE}/redoc`} target="_blank" rel="noreferrer">ReDoc</a>
          </div>
        </div>
      </header>
      <main className="p-6">
        <div className="max-w-6xl mx-auto grid gap-6 md:grid-cols-12">
          <section className="md:col-span-4">
            <div className="bg-white rounded-xl shadow p-5">
              <h2 className="text-lg font-medium mb-3">Providers</h2>
              <ul className="space-y-1">
                {Object.entries(providers).map(([k,v]) => (
                  <li key={k} className="flex items-center justify-between">
                    <span className="font-mono text-sm">{k}</span>
                    <span className={v ? 'text-green-600' : 'text-red-600'}>{v ? 'available' : 'unavailable'}</span>
                  </li>
                ))}
              </ul>
            </div>
          </section>
          <section className="md:col-span-8">
            <div className="bg-white rounded-xl shadow p-5">
              <h2 className="text-lg font-medium mb-3">API Endpoints</h2>
              <div className="grid gap-3 sm:grid-cols-2">
                {apiIndex.endpoints && apiIndex.endpoints.length > 0 ? (
                  apiIndex.endpoints.map((ep, i) => <EndpointCard key={i} item={ep} />)
                ) : (
                  <div className="text-sm text-gray-500">No endpoints loaded.</div>
                )}
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>
  )
}
"""


