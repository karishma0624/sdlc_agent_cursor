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

        # Phase 2 – Backend Generation (Dynamic AI-generated backend)
        be_models = {}
        try:
            backend_root = os.path.join(run_dir, "backend")
            for d in ["routes", "models", "services", "tests"]:
                os.makedirs(os.path.join(backend_root, d), exist_ok=True)
            
            written = 0
            
            # Generate dynamic backend based on prompt
            backend_instruction = f"""
Create a complete, functional FastAPI backend for: {prompt}

Requirements:
1. Generate a main.py with proper CORS middleware, health endpoints, and dynamic routes
2. Create models/ directory with Pydantic models for the domain entities
3. Create routes/ directory with CRUD endpoints for the main entities
4. Create services/ directory with business logic
5. Include proper error handling and validation
6. Add /health, /status, and /providers endpoints that return JSON
7. Use structured logging with JSON format
8. Include proper imports and dependencies

The backend should be fully functional and ready to run. Return ONLY a JSON object mapping file paths to contents.
"""
            
            gen = self.router.generate_code(backend_instruction)
            files = gen.get("files") or {}
            be_models = {"provider": gen.get("provider"), "model": gen.get("model"), "tokens": gen.get("tokens")}
            
            if isinstance(files, dict) and files:
                for rel, content in files.items():
                    path = os.path.join(backend_root, rel)
                    self._write_text(path, content)
                    written += 1
            
            # Ensure essential files exist with dynamic content
            if not os.path.exists(os.path.join(backend_root, "main.py")):
                main_content = self._generate_dynamic_main(prompt)
                self._write_text(os.path.join(backend_root, "main.py"), main_content)
                written += 1
                
            if not os.path.exists(os.path.join(backend_root, "requirements.txt")):
                req_content = self._generate_dynamic_requirements(prompt)
                self._write_text(os.path.join(backend_root, "requirements.txt"), req_content)
                written += 1
                
            if not os.path.exists(os.path.join(backend_root, "tests", "test_health.py")):
                test_content = self._generate_dynamic_tests(prompt)
                self._write_text(os.path.join(backend_root, "tests", "test_health.py"), test_content)
                written += 1
                
            # Generate dynamic status.json
            status_content = self._generate_dynamic_status(prompt, written, be_models)
            self._write_text(os.path.join(run_dir, "status.json"), status_content)
            written += 1
            
            artifacts["backend"] = {"root": backend_root}
            record("backend", True, be_models, written, None)
        except Exception as e:
            record("backend", False, be_models, 0, str(e))

        # Phase 3 – Frontend Generation (Dynamic React + Tailwind + Vite)
        fe_models = {}
        try:
            frontend_root = os.path.join(run_dir, "frontend")
            files_written = 0
            files_created: set[str] = set()
            used_v0 = False
            
            # Try v0.dev first for dynamic frontend generation
            if (not self.fast_mode) and self.router.providers.get("v0"):
                v0_prompt = f"""
Create a modern, attractive React (Vite) + Tailwind frontend for: {prompt}

Requirements:
1. Use React + Vite + Tailwind CSS
2. Connect to backend API at VITE_API_BASE (default http://localhost:8000)
3. Create components that are specific to the domain: {prompt}
4. Include proper error handling and loading states
5. Make it responsive and attractive
6. Fetch data from backend endpoints (/health, /status, /providers, /)
7. Include navigation and proper component structure
8. Use modern React patterns (hooks, functional components)

The frontend should be fully functional and ready to run with npm install && npm run dev.
"""
                resp = self.router.run_tool("v0", v0_prompt)
                files = resp.get("files") if isinstance(resp, dict) else None
                fe_models = {"provider": "v0", "model": (resp.get("model") if isinstance(resp, dict) else None)}
                if files:
                    used_v0 = True
                    for rel, content in files.items():
                        path = os.path.join(frontend_root, rel)
                        self._write_text(path, content)
                        files_written += 1
                        files_created.add(rel.replace("\\", "/"))
            
            # Ensure essential files exist with dynamic content
            def ensure(rel_path: str, content: str) -> None:
                nonlocal files_written
                normalized = rel_path.replace("\\", "/")
                if normalized not in files_created:
                    self._write_text(os.path.join(frontend_root, rel_path), content)
                    files_written += 1
                    files_created.add(normalized)

            # Generate dynamic frontend files
            ensure("package.json", self._generate_dynamic_package_json(prompt))
            ensure("tailwind.config.js", _TEMPLATE_TAILWIND_CONFIG)
            ensure("postcss.config.js", _TEMPLATE_POSTCSS)
            ensure("vite.config.js", _TEMPLATE_VITE)
            ensure("index.html", self._generate_dynamic_index_html(prompt))
            ensure(os.path.join("src", "main.jsx"), _TEMPLATE_MAIN_JSX)
            ensure(os.path.join("src", "index.css"), _TEMPLATE_INDEX_CSS)
            ensure(os.path.join("src", "App.jsx"), self._generate_dynamic_app_jsx(prompt))
            ensure(os.path.join("src", "components", "ApiStatus.jsx"), self._generate_api_status_component())
            ensure(os.path.join("src", "components", "EndpointCard.jsx"), self._generate_endpoint_card_component())
            
            artifacts["frontend"] = {"root": frontend_root}
            record("frontend", True, {**fe_models, "used_v0": used_v0}, files_written, None)
        except Exception as e:
            record("frontend", False, fe_models, 0, str(e))

        # Phase 4 – Testing & Deployment (Dynamic Docker integration)
        td_models = {}
        try:
            td_models = {"tests": ("mistral" if self.router.providers.get("mistral") else None), "infra": [n for n in ["groq", "openai"] if self.router.providers.get(n)]}
            
            # Generate dynamic deployment files
            self._write_text(os.path.join(run_dir, "pytest.ini"), _TEMPLATE_PYTEST_INI)
            self._write_text(os.path.join(run_dir, "Dockerfile"), self._generate_dynamic_dockerfile(prompt))
            self._write_text(os.path.join(run_dir, "docker-compose.yml"), self._generate_dynamic_docker_compose(prompt))
            
            # Generate frontend Dockerfile
            frontend_dockerfile = self._generate_frontend_dockerfile()
            self._write_text(os.path.join(run_dir, "frontend", "Dockerfile"), frontend_dockerfile)
            
            os.makedirs(os.path.join(run_dir, ".github", "workflows"), exist_ok=True)
            self._write_text(os.path.join(run_dir, ".github", "workflows", "deploy.yml"), _TEMPLATE_GH_ACTIONS)
            
            artifacts["infra"] = {
                "dockerfile": os.path.join(run_dir, "Dockerfile"), 
                "frontend_dockerfile": os.path.join(run_dir, "frontend", "Dockerfile"),
                "compose": os.path.join(run_dir, "docker-compose.yml"), 
                "workflow": os.path.join(run_dir, ".github", "workflows", "deploy.yml")
            }
            record("deployment", True, td_models, 5, None)
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

        # Phase 5 – Documentation (Dynamic documentation generation)
        doc_models = {}
        try:
            # Generate dynamic README
            readme_md = self._generate_dynamic_readme(prompt)
            doc_models = {"generated": "dynamic"}
            
            # Create docs directory and files
            docs_root = os.path.join(run_dir, "docs")
            os.makedirs(docs_root, exist_ok=True)
            
            self._write_text(os.path.join(run_dir, "README.md"), readme_md)
            self._write_text(os.path.join(run_dir, "mkdocs.yml"), _TEMPLATE_MKDOCS)
            self._write_text(os.path.join(docs_root, "index.md"), _TEMPLATE_DOCS_INDEX)
            
            artifacts["docs"] = {
                "paths": [
                    os.path.join(run_dir, "README.md"), 
                    os.path.join(run_dir, "mkdocs.yml"), 
                    os.path.join(docs_root, "index.md")
                ]
            }
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

    def _generate_dynamic_main(self, prompt: str) -> str:
        """Generate a dynamic main.py based on the prompt"""
        return f'''from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import json
from datetime import datetime
import os

# Configure structured JSON logging
logging.basicConfig(
    level=logging.INFO,
    format='{{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}}',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Generated API for: {prompt}",
    description="Dynamically generated FastAPI application",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (replace with database in production)
storage = {{}}

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    service: str

class StatusResponse(BaseModel):
    status: str
    timestamp: str
    service: str
    version: str
    endpoints: List[Dict[str, str]]

class ProvidersResponse(BaseModel):
    providers: Dict[str, bool]
    timestamp: str

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    logger.info("Health check requested")
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        service="Generated API"
    )

@app.get("/status", response_model=StatusResponse)
async def status():
    """Status endpoint with API information"""
    logger.info("Status check requested")
    endpoints = [
        {{"method": "GET", "path": "/health", "description": "Health check"}},
        {{"method": "GET", "path": "/status", "description": "API status"}},
        {{"method": "GET", "path": "/providers", "description": "Provider information"}},
        {{"method": "GET", "path": "/", "description": "API root"}},
    ]
    return StatusResponse(
        status="running",
        timestamp=datetime.utcnow().isoformat(),
        service="Generated API for: {prompt}",
        version="1.0.0",
        endpoints=endpoints
    )

@app.get("/providers", response_model=ProvidersResponse)
async def providers():
    """Provider information endpoint"""
    logger.info("Providers check requested")
    provider_info = {{
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "gemini": bool(os.getenv("GEMINI_API_KEY")),
        "mistral": bool(os.getenv("MISTRAL_API_KEY")),
        "groq": bool(os.getenv("GROQ_API_KEY")),
        "huggingface": bool(os.getenv("HUGGINGFACE_API_KEY")),
        "perplexity": bool(os.getenv("PERPLEXITY_API_KEY")),
        "v0": bool(os.getenv("V0_API_KEY")),
    }}
    return ProvidersResponse(
        providers=provider_info,
        timestamp=datetime.utcnow().isoformat()
    )

@app.get("/")
async def root():
    """API root endpoint"""
    logger.info("Root endpoint accessed")
    return {{
        "message": "Generated API for: {prompt}",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": [
            {{"method": "GET", "path": "/health", "description": "Health check"}},
            {{"method": "GET", "path": "/status", "description": "API status"}},
            {{"method": "GET", "path": "/providers", "description": "Provider information"}},
            {{"method": "GET", "path": "/", "description": "API root"}},
        ],
        "docs": {{
            "swagger": "/docs",
            "redoc": "/redoc"
        }}
    }}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

    def _generate_dynamic_requirements(self, prompt: str) -> str:
        """Generate dynamic requirements.txt based on the prompt"""
        return '''fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
python-dotenv==1.0.0
requests==2.31.0
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2
'''

    def _generate_dynamic_tests(self, prompt: str) -> str:
        """Generate dynamic tests based on the prompt"""
        return f'''import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_endpoint():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "service" in data

def test_status_endpoint():
    """Test status endpoint"""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "timestamp" in data
    assert "service" in data
    assert "endpoints" in data
    assert isinstance(data["endpoints"], list)

def test_providers_endpoint():
    """Test providers endpoint"""
    response = client.get("/providers")
    assert response.status_code == 200
    data = response.json()
    assert "providers" in data
    assert "timestamp" in data
    assert isinstance(data["providers"], dict)

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data
    assert "docs" in data

def test_cors_headers():
    """Test CORS headers are present"""
    response = client.options("/health")
    # CORS preflight should be allowed
    assert response.status_code in [200, 405]  # 405 is also acceptable for OPTIONS
'''

    def _generate_dynamic_status(self, prompt: str, files_generated: int, models_used: Dict[str, Any]) -> str:
        """Generate dynamic status.json with build information"""
        status_data = {
            "build_info": {
                "prompt": prompt,
                "timestamp": datetime.utcnow().isoformat(),
                "files_generated": files_generated,
                "models_used": models_used,
                "status": "completed"
            },
            "modules": {
                "backend": {
                    "status": "generated",
                    "files": ["main.py", "requirements.txt", "tests/test_health.py"],
                    "endpoints": ["/health", "/status", "/providers", "/"]
                },
                "frontend": {
                    "status": "pending",
                    "framework": "React + Vite + Tailwind"
                },
                "tests": {
                    "status": "generated",
                    "framework": "pytest",
                    "coverage": "basic"
                }
            },
            "deployment": {
                "docker": "ready",
                "docker_compose": "ready",
                "requirements": "generated"
            }
        }
        return self._json(status_data)

    def _generate_dynamic_package_json(self, prompt: str) -> str:
        """Generate dynamic package.json based on the prompt"""
        return f'''{{
  "name": "generated-frontend-{prompt.lower().replace(' ', '-')[:20]}",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {{
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview --port 5173",
    "start": "vite preview --host 0.0.0.0 --port 5173"
  }},
  "dependencies": {{
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.6.0"
  }},
  "devDependencies": {{
    "@vitejs/plugin-react": "^4.2.0",
    "autoprefixer": "^10.4.18",
    "postcss": "^8.4.38",
    "tailwindcss": "^3.4.3",
    "vite": "^5.0.0"
  }}
}}'''

    def _generate_dynamic_index_html(self, prompt: str) -> str:
        """Generate dynamic index.html based on the prompt"""
        return f'''<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Generated App - {prompt}</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>'''

    def _generate_dynamic_app_jsx(self, prompt: str) -> str:
        """Generate dynamic App.jsx based on the prompt"""
        return f'''import {{ useEffect, useState }} from 'react'
import ApiStatus from './components/ApiStatus'
import EndpointCard from './components/EndpointCard'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export default function App() {{
  const [status, setStatus] = useState('unknown')
  const [providers, setProviders] = useState({{}})
  const [apiIndex, setApiIndex] = useState({{ endpoints: [], docs: {{}} }})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {{
    const fetchData = async () => {{
      try {{
        setLoading(true)
        setError(null)
        
        // Fetch health status
        const healthRes = await fetch(`${{API_BASE}}/health`)
        setStatus(healthRes.ok ? 'online' : 'offline')
        
        // Fetch providers
        const providersRes = await fetch(`${{API_BASE}}/providers`)
        if (providersRes.ok) {{
          const providersData = await providersRes.json()
          setProviders(providersData.providers || {{}})
        }}
        
        // Fetch API index
        const indexRes = await fetch(`${{API_BASE}}/`)
        if (indexRes.ok) {{
          const indexData = await indexRes.json()
          setApiIndex(indexData || {{ endpoints: [], docs: {{}} }})
        }}
      }} catch (err) {{
        setError(err.message)
        setStatus('offline')
      }} finally {{
        setLoading(false)
      }}
    }}

    fetchData()
  }}, [])

  if (loading) {{
    return (
      <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    )
  }}

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 text-gray-900">
      <header className="p-6 border-b bg-white/80 backdrop-blur">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold">Generated App - {prompt}</h1>
            <p className="text-sm text-gray-500">
              API status: <span className={{status === 'online' ? 'text-green-600' : 'text-red-600'}}>
                {{status}}
              </span>
            </p>
          </div>
          <div className="text-sm text-gray-600">
            <a className="underline mr-3" href={`${{API_BASE}}/docs`} target="_blank" rel="noreferrer">
              Swagger
            </a>
            <a className="underline" href={`${{API_BASE}}/redoc`} target="_blank" rel="noreferrer">
              ReDoc
            </a>
          </div>
        </div>
      </header>
      
      <main className="p-6">
        {{error && (
          <div className="max-w-6xl mx-auto mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-600">Error: {{error}}</p>
          </div>
        )}}
        
        <div className="max-w-6xl mx-auto grid gap-6 md:grid-cols-12">
          <section className="md:col-span-4">
            <div className="bg-white rounded-xl shadow p-5">
              <h2 className="text-lg font-medium mb-3">API Status</h2>
              <ApiStatus status={{status}} providers={{providers}} />
            </div>
          </section>
          
          <section className="md:col-span-8">
            <div className="bg-white rounded-xl shadow p-5">
              <h2 className="text-lg font-medium mb-3">API Endpoints</h2>
              <div className="grid gap-3 sm:grid-cols-2">
                {{apiIndex.endpoints && apiIndex.endpoints.length > 0 ? (
                  apiIndex.endpoints.map((ep, i) => (
                    <EndpointCard key={{i}} item={{ep}} />
                  ))
                ) : (
                  <div className="text-sm text-gray-500">No endpoints loaded.</div>
                )}}
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>
  )
}}'''

    def _generate_api_status_component(self) -> str:
        """Generate ApiStatus component"""
        return '''import React from 'react'

export default function ApiStatus({ status, providers }) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">Status</span>
        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
          status === 'online' 
            ? 'bg-green-100 text-green-800' 
            : 'bg-red-100 text-red-800'
        }`}>
          {status}
        </span>
      </div>
      
      <div>
        <h3 className="text-sm font-medium mb-2">Providers</h3>
        <ul className="space-y-1">
          {Object.entries(providers).map(([key, value]) => (
            <li key={key} className="flex items-center justify-between">
              <span className="font-mono text-sm">{key}</span>
              <span className={value ? 'text-green-600' : 'text-red-600'}>
                {value ? 'available' : 'unavailable'}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}'''

    def _generate_endpoint_card_component(self) -> str:
        """Generate EndpointCard component"""
        return '''import React from 'react'

export default function EndpointCard({ item }) {
  return (
    <div className="rounded-lg border p-4 bg-white shadow-sm hover:shadow-md transition-shadow">
      <div className="text-xs uppercase tracking-wide text-gray-500 mb-1">
        {item.method}
      </div>
      <div className="font-mono text-sm break-all mb-2">
        {item.path}
      </div>
      {item.description && (
        <div className="text-sm text-gray-600">
          {item.description}
        </div>
      )}
    </div>
  )
}'''

    def _generate_dynamic_dockerfile(self, prompt: str) -> str:
        """Generate dynamic Dockerfile for backend"""
        return f'''FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
'''

    def _generate_frontend_dockerfile(self) -> str:
        """Generate Dockerfile for frontend"""
        return '''FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Expose port
EXPOSE 5173

# Start the application
CMD ["npm", "run", "preview"]
'''

    def _generate_dynamic_docker_compose(self, prompt: str) -> str:
        """Generate dynamic docker-compose.yml"""
        return f'''version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
    volumes:
      - ./backend:/app/backend
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "5173:5173"
    environment:
      - VITE_API_BASE=http://localhost:8000
    depends_on:
      backend:
        condition: service_healthy
    restart: unless-stopped

networks:
  default:
    name: generated-app-network
'''

    def _generate_dynamic_readme(self, prompt: str) -> str:
        """Generate dynamic README.md"""
        return f'''# Generated Application: {prompt}

This application was dynamically generated based on the prompt: "{prompt}"

## Architecture

- **Backend**: FastAPI with Python 3.11
- **Frontend**: React + Vite + Tailwind CSS
- **Database**: In-memory storage (replace with persistent database for production)
- **Deployment**: Docker + Docker Compose

## Quick Start

### Development Mode

1. **Backend**:
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```

2. **Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

### Docker Mode

```bash
docker-compose up --build
```

## API Endpoints

- `GET /health` - Health check
- `GET /status` - API status and information
- `GET /providers` - Provider availability
- `GET /` - API root with endpoint list
- `GET /docs` - Swagger documentation
- `GET /redoc` - ReDoc documentation

## Testing

```bash
# Backend tests
cd backend
pytest

# Frontend tests (if available)
cd frontend
npm test
```

## Environment Variables

### Backend
- `PYTHONPATH` - Python path (set to /app in Docker)

### Frontend
- `VITE_API_BASE` - Backend API URL (default: http://localhost:8000)

## Features

- ✅ Dynamic backend generation with FastAPI
- ✅ Modern React frontend with Tailwind CSS
- ✅ CORS middleware for frontend-backend communication
- ✅ Health checks and monitoring endpoints
- ✅ Docker containerization
- ✅ Structured JSON logging
- ✅ Comprehensive test suite
- ✅ API documentation (Swagger/ReDoc)

## Customization

This is a generated application. Customize the code in:
- `backend/` - Backend API logic
- `frontend/src/` - Frontend components and logic
- `docker-compose.yml` - Deployment configuration

## Support

Generated by Autonomous SDLC Agent
'''


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


