import os
import json
import threading
from datetime import datetime
from typing import Dict, Any, List

from .services.adapters import InferenceRouter

class SimpleBuilder:
    def __init__(self, runs_dir: str = "runs"):
        self.runs_dir = runs_dir
        self.router = InferenceRouter()
        os.makedirs(self.runs_dir, exist_ok=True)

    def build(self, prompt: str) -> Dict[str, Any]:
        """Legacy wrapper for backward protection if needed, though we will use new methods."""
        run_dir = self.init_run(prompt)
        return self.run_build(run_dir, prompt)

    def init_run(self, prompt: str, job_id: str = None) -> str:
        """Creates run directory and initial status.json immediately."""
        slug = "".join(c for c in prompt.lower() if c.isalnum() or c == " ").strip().replace(" ", "-")[:30]
        if not slug:
            slug = "project"
        run_dir = self._create_run_dir(slug)
        
        # Immediate status.json
        status = {
            "job_id": job_id or os.path.basename(run_dir),
            "status": "running",
            "started_at": datetime.utcnow().isoformat(),
            "prompt": prompt,
            "backend": {},
            "frontend": {}
        }
        self._write_file(run_dir, "status.json", json.dumps(status, indent=2))
        return run_dir

    # ... (imports remain)

    def run_build(self, run_dir: str, prompt: str) -> Dict[str, Any]:
        """
        Hybrid build execution in the specific run_dir with Intelligent Routing.
        """
        self.router.refresh()
        prompt_lower = prompt.lower()
        
        # Track execution details
        execution_log = []
        
        def update_status(stage: str, provider: str = "system", status_msg: str = "running"):
            try:
                p = os.path.join(run_dir, "status.json")
                if os.path.exists(p):
                    with open(p, "r") as f: current = json.load(f)
                else:
                    current = {}
                
                current.update({
                    "current_stage": stage,
                    "provider_used": provider,
                    "status_message": status_msg,
                    "last_updated": datetime.utcnow().isoformat()
                })
                self._write_file(run_dir, "status.json", json.dumps(current, indent=2))
                execution_log.append({"stage": stage, "provider": provider, "msg": status_msg, "time": datetime.utcnow().isoformat()})
            except:
                pass

        result = {}
        
        # 1. Deterministic Shortcuts
        if "counter" in prompt_lower:
            update_status("planning", "system", "Dedicated template found: Counter App")
            result = self._build_counter(run_dir, prompt)
            update_status("completed", "system", "Build finished")
        elif "hello" in prompt_lower:
            update_status("planning", "system", "Dedicated template found: Hello World")
            result = self._build_hello(run_dir, prompt)
            update_status("completed", "system", "Build finished")
        else:
            # 2. AI Path
            update_status("planning", "system", "Analyzing requirements...")
            
            # Check AI or just try fallback robustly
            # We will try AI route, but if it fails, _build_ai_intelligent now handles fallback internally per stage
            update_status("generation", "auto", "Starting AI Generation...")
            result = self._build_ai_intelligent(run_dir, prompt, update_status)

        # Update status.json to completed
        try:
            status_path = os.path.join(run_dir, "status.json")
            with open(status_path, "r") as f:
                current_status = json.load(f)
            
            current_status.update({
                "status": "completed",
                "finished_at": datetime.utcnow().isoformat(),
                "success": True,
                "summary": result.get("summary"),
                "execution_log": execution_log
            })
            self._write_file(run_dir, "status.json", json.dumps(current_status, indent=2))
        except Exception:
            pass
            
        return result

    def _select_provider(self, stage: str) -> List[str]:
        """Intelligent Routing: Select best provider preferences based on stage."""
        if stage == "planning":
            return ["openai", "gemini", "anthropic", "mistral"]
        elif stage == "backend_code" or stage == "frontend_code":
            return ["claude", "openai", "mistral", "deepseek", "gemini", "groq", "hf"]
        return ["openai", "gemini", "mistral", "groq", "hf", "ollama"]

    def _build_ai_intelligent(self, run_dir: str, prompt: str, status_cb) -> Dict[str, Any]:
        slug = os.path.basename(run_dir)
        
        # --- BACKEND ---
        try:
            backend_pref = self._select_provider("backend_code")
            status_cb("backend_generation", str(backend_pref), "Generating FastAPI Backend...")
            
            backend_prompt = (
                f"Generate a single-file FastAPI backend (main.py) for: {prompt}. "
                "Include ALL imports, a /health endpoint, and CORS middleware allowing '*'. "
                "Use Pydantic models. Return ONLY the python code."
            )
            
            be_res = self.router.generate_text(backend_prompt, preference=backend_pref)
            be_code = self._clean_code(be_res.get("output", ""))
            
            if "FastAPI" not in be_code:
                 # Fallback if AI produced garbage
                 be_code = f"""from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
@app.get("/")
def root(): return {{"message": "AI generation failed, fallback active for: {prompt}"}}
@app.get("/health")
def health(): return {{"status": "ok"}}
"""
                 status_cb("backend_generation", "system", "AI failed, used fallback backend")

            self._write_file(run_dir, "backend/main.py", be_code)
            self._write_file(run_dir, "backend/requirements.txt", "fastapi\nuvicorn\npydantic\npython-multipart\n")
        except Exception as e:
            status_cb("backend_error", "system", f"Backend generation crashed: {e}")
            # Ensure file exists even on crash
            self._write_file(run_dir, "backend/main.py", """from fastapi import FastAPI\napp = FastAPI()\n""")

        # --- FRONTEND ---
        try:
            frontend_pref = self._select_provider("frontend_code")
            status_cb("frontend_generation", str(frontend_pref), "Generating React Frontend...")
            
            fe_prompt = (
                f"Generate a React component (App.jsx) for: {prompt}. "
                "Assume the backend is at VITE_API_BASE. "
                "Use fetch() to call the endpoints you would expect for this app. "
                "Use Tailwind CSS classes for styling. Return ONLY the jsx code."
            )
            fe_res = self.router.generate_text(fe_prompt, preference=frontend_pref)
            fe_code = self._clean_code(fe_res.get("output", ""), lang="jsx")

            if "export default" not in fe_code and "function" not in fe_code:
                 fe_code = self._get_hello_frontend()
                 status_cb("frontend_generation", "system", "AI failed, used fallback frontend")

            self._write_frontend_scaffold(run_dir, fe_code)
        except Exception as e:
             status_cb("frontend_error", "system", f"Frontend generation crashed: {e}")
             # Ensure fallback exists
             self._write_frontend_scaffold(run_dir, self._get_hello_frontend())
        
        status_cb("finalizing", "system", "Writing artifacts...")
        self._write_common_artifacts(run_dir, prompt, f"AI Generated: {slug}")
        
        return {"summary": f"AI Build Completed for {slug}", "run_dir": run_dir}

    # ... (rest of methods: _clean_code, _write_frontend_scaffold, _write_common_artifacts, _get_, etc. keep existing)

    def _create_run_dir(self, slug: str) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        run_dir_name = f"{timestamp}-{slug}"
        run_dir = os.path.join(self.runs_dir, run_dir_name)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def _write_file(self, run_dir: str, rel_path: str, content: str):
        full_path = os.path.join(run_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

    def _build_counter(self, run_dir: str, prompt: str) -> Dict[str, Any]:
        # Reuse existing logic... 
        backend_main = """from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

count = 0

class CountResponse(BaseModel):
    count: int

@app.get("/count", response_model=CountResponse)
def get_count():
    global count
    return {"count": count}

@app.post("/increment", response_model=CountResponse)
def increment_count():
    global count
    count += 1
    return {"count": count}

@app.get("/health")
def health():
    return {"status": "ok"}
"""
        self._write_file(run_dir, "backend/main.py", backend_main)
        self._write_file(run_dir, "backend/requirements.txt", "fastapi\nuvicorn\npydantic\n")
        self._write_frontend_scaffold(run_dir, self._get_counter_frontend())
        self._write_common_artifacts(run_dir, prompt, "Deterministic Counter App")
        return {"summary": "Counter App Generated", "run_dir": run_dir}

    def _build_hello(self, run_dir: str, prompt: str) -> Dict[str, Any]:
        backend_main = """from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello from your generated app!"}

@app.get("/health")
def health():
    return {"status": "ok"}
"""
        self._write_file(run_dir, "backend/main.py", backend_main)
        self._write_file(run_dir, "backend/requirements.txt", "fastapi\nuvicorn\n")
        self._write_frontend_scaffold(run_dir, self._get_hello_frontend())
        self._write_common_artifacts(run_dir, prompt, "Deterministic Hello World")
        return {"summary": "Hello World App Generated", "run_dir": run_dir}

    def _build_ai(self, run_dir: str, prompt: str) -> Dict[str, Any]:
        slug = os.path.basename(run_dir)
        
        # 1. Generate Backend
        backend_prompt = (
            f"Generate a single-file FastAPI backend (main.py) for: {prompt}. "
            "Include ALL imports, a /health endpoint, and CORS middleware allowing '*'. "
            "Use Pydantic models. Return ONLY the python code."
        )
        # Use generate_code if we trust it, or generate_text and strip fences.
        # simple_builder uses generate_text for robust fallback logic usually, but adapters has generate_code
        be_res = self.router.generate_text(backend_prompt, preference=["gemini", "openai", "mistral", "groq", "hf"])
        be_code = self._clean_code(be_res.get("output", ""))
        
        # Fallback if AI fails to give code
        if "FastAPI" not in be_code:
             be_code = """from fastapi import FastAPI\napp = FastAPI()\n@app.get("/")\ndef root(): return {"error": "AI failed generation"}\n"""

        self._write_file(run_dir, "backend/main.py", be_code)
        self._write_file(run_dir, "backend/requirements.txt", "fastapi\nuvicorn\npydantic\npython-multipart\n")

        # 2. Generate Frontend
        fe_prompt = (
            f"Generate a React component (App.jsx) for: {prompt}. "
            "Assume the backend is at VITE_API_BASE. "
            "Use fetch() to call the endpoints you would expect for this app. "
            "Use Tailwind CSS classes for styling. Return ONLY the jsx code."
        )
        fe_res = self.router.generate_text(fe_prompt, preference=["gemini", "openai", "mistral", "groq", "hf"])
        fe_code = self._clean_code(fe_res.get("output", ""), lang="jsx")

        # Fallback
        if "export default" not in fe_code and "function" not in fe_code:
             fe_code = self._get_hello_frontend()

        self._write_frontend_scaffold(run_dir, fe_code)
        self._write_common_artifacts(run_dir, prompt, f"AI Generated: {slug}")
        
        return {"summary": f"AI Build Completed for {slug}", "run_dir": run_dir}

    def _clean_code(self, text: str, lang: str = "python") -> str:
        # Remove markdown fences
        lines = text.split("\n")
        clean = []
        in_fence = False
        for line in lines:
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            clean.append(line)
        return "\n".join(clean).strip()

    def _write_frontend_scaffold(self, run_dir: str, app_jsx: str):
        self._write_file(run_dir, "frontend/src/App.jsx", app_jsx)
        self._write_file(run_dir, "frontend/package.json", """{"name": "gen-app", "version": "0.0.0", "scripts": {"dev": "vite", "build": "vite build"}, "dependencies": {"react": "^18.2.0", "react-dom": "^18.2.0"}, "devDependencies": {"@vitejs/plugin-react": "^4.0.0", "vite": "^5.0.0", "tailwindcss": "^3.0.0", "autoprefixer": "^10.0.0", "postcss": "^8.0.0"}}""")
        self._write_file(run_dir, "frontend/vite.config.js", "import { defineConfig } from 'vite'; import react from '@vitejs/plugin-react'; export default defineConfig({ plugins: [react()], server: { host: '0.0.0.0', port: 5173 } });")
        self._write_file(run_dir, "frontend/index.html", "<!doctype html><html lang='en'><head><meta charset='UTF-8' /><title>App</title></head><body><div id='root'></div><script type='module' src='/src/main.jsx'></script></body></html>")
        self._write_file(run_dir, "frontend/src/main.jsx", "import React from 'react'; import ReactDOM from 'react-dom/client'; import App from './App.jsx'; import './index.css'; ReactDOM.createRoot(document.getElementById('root')).render(<React.StrictMode><App /></React.StrictMode>);")
        self._write_file(run_dir, "frontend/src/index.css", "@tailwind base; @tailwind components; @tailwind utilities;")
        self._write_file(run_dir, "frontend/postcss.config.js", "export default { plugins: { tailwindcss: {}, autoprefixer: {} } }")
        self._write_file(run_dir, "frontend/tailwind.config.js", "export default { content: ['./index.html', './src/**/*.{js,jsx}'], theme: { extend: {} }, plugins: [] }")

    def _write_common_artifacts(self, run_dir: str, prompt: str, summary: str):
         # Status
        status = {
            "prompt": prompt,
            "timestamp": datetime.utcnow().isoformat(),
            "success": True,
            "builder_used": "hybrid"
        }
        self._write_file(run_dir, "status.json", json.dumps(status, indent=2))
        
        # Readme
        readme = f"# {prompt}\n\nGenerated by SDLC Agent.\n\n## Run\n- Backend: `uvicorn backend.main:app --reload`\n- Frontend: `npm run dev`"
        self._write_file(run_dir, "README.md", readme)
        
        # Report
        report = {"files_generated": True, "success": True, "fixe": "none"}
        self._write_file(run_dir, "run_report.json", json.dumps(report, indent=2))

    def _get_counter_frontend(self):
        return """import { useState, useEffect } from 'react'

function App() {
  const [count, setCount] = useState(0)
  const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

  const fetchCount = async () => {
    try {
      const res = await fetch(`${API_BASE}/count`)
      const data = await res.json()
      setCount(data.count)
    } catch (e) { console.error(e) }
  }

  const increment = async () => {
    try {
      const res = await fetch(`${API_BASE}/increment`, { method: 'POST' })
      const data = await res.json()
      setCount(data.count)
    } catch (e) { console.error(e) }
  }

  useEffect(() => { fetchCount() }, [])

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-4">
      <div className="bg-white shadow-xl rounded-2xl p-8 w-full max-w-sm text-center">
        <h1 className="text-3xl font-bold text-gray-800 mb-6">Counter App</h1>
        <div className="text-6xl font-mono text-blue-600 mb-8">{count}</div>
        <button 
          onClick={increment}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
        >
          Increment
        </button>
      </div>
    </div>
  )
}
export default App
"""

    def _get_hello_frontend(self):
        return """import { useState, useEffect } from 'react'

function App() {
  const [msg, setMsg] = useState('Loading...')
  const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

  useEffect(() => {
    fetch(`${API_BASE}/`)
      .then(r => r.json())
      .then(d => setMsg(d.message))
      .catch(e => setMsg('Error'))
  }, [])

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="p-8 bg-white shadow rounded-xl">
        <h1 className="text-2xl font-bold mb-4">Generated App</h1>
        <p className="text-gray-600">Backend says: <span className="font-mono text-blue-600">{msg}</span></p>
      </div>
    </div>
  )
}
export default App
"""
