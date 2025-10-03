from __future__ import annotations
from typing import Dict, Any, List
from datetime import datetime
import os
import json

from .adapters import InferenceRouter
from .storage import DatabaseService
from .rag import RagService


class AgentOrchestrator:
	def __init__(self, base_dir: str = "runs") -> None:
		self.router = InferenceRouter()
		self.db = DatabaseService()
		self.rag = RagService(self.db)
		self.base_dir = base_dir
		os.makedirs(self.base_dir, exist_ok=True)

	def _mk_run_dir(self, title: str) -> str:
		ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
		slug = "".join(c for c in title.lower() if c.isalnum() or c in ("-","_"," ")).strip().replace(" ", "-")
		path = os.path.join(self.base_dir, f"{ts}-{slug[:40]}")
		os.makedirs(path, exist_ok=True)
		return path

	def run(self, prompt: str) -> Dict[str, Any]:
		# Always refresh providers in case env changed since startup
		self.router.refresh()
		run_dir = self._mk_run_dir(prompt or "task")
		logs: List[Dict[str, Any]] = []

		def log(stage: str, success: bool, message: str, metadata: Dict[str, Any]):
			entry = {
				"timestamp": datetime.utcnow().isoformat(),
				"stage": stage,
				"success": success,
				"message": message,
				"metadata": metadata or {},
			}
			logs.append(entry)
			self.db.save_log(stage=stage, provider=metadata.get("provider"), model=metadata.get("model"), success=success, message=message, metadata=metadata)

		# Requirements (prefer gemini/perplexity)
		req = self.router.generate_text(
			f"Extract concise functional/non-functional requirements, constraints, acceptance criteria for: {prompt}",
			preference=["gemini","perplexity"],
		)
		self._write_text(os.path.join(run_dir, "requirements.md"), req.get("output", ""))
		log("requirements", True, "requirements extracted", req)

		# Design (prefer gemini/perplexity)
		design = self.router.generate_text(
			f"Create a Mermaid system diagram and an OpenAPI high-level outline for: {prompt}",
			preference=["gemini","perplexity"],
		)
		self._write_text(os.path.join(run_dir, "design.md"), design.get("output", ""))
		log("design", True, "design generated", design)

		# Build (scaffold minimal runnable app tailored to prompt)
		backend_dir = os.path.join(run_dir, "backend")
		frontend_dir = os.path.join(run_dir, "frontend")
		os.makedirs(backend_dir, exist_ok=True)
		os.makedirs(frontend_dir, exist_ok=True)
		self._write_text(os.path.join(backend_dir, "main.py"), self._template_fastapi(prompt))
		# Attempt v0.dev frontend generation when key present, fallback to minimal UI
		files = None
		try:
			if self.router.providers.get("v0"):
				resp = self.router.run_tool("v0", f"TASK: frontend_only\nSTACK: react + tailwind\nOUTCOME: '{prompt}'")
				files = resp.get("files") if isinstance(resp, dict) else None
		except Exception:
			files = None
		if files:
			for path, content in files.items():
				fpath = os.path.join(frontend_dir, path)
				os.makedirs(os.path.dirname(fpath), exist_ok=True)
				self._write_text(fpath, content)
		else:
			self._write_text(os.path.join(frontend_dir, "app.py"), self._template_streamlit(prompt))
		log("build", True, "scaffold created", {"provider": req.get("provider"), "model": req.get("model")})

		# Tests
		tests_dir = os.path.join(run_dir, "tests")
		os.makedirs(tests_dir, exist_ok=True)
		self._write_text(os.path.join(tests_dir, "test_smoke.py"), self._template_test())
		log("test", True, "tests scaffolded", {})

		# Deploy
		self._write_text(os.path.join(run_dir, "Dockerfile"), self._template_dockerfile())
		self._write_text(os.path.join(run_dir, "docker-compose.yml"), self._template_compose())
		log("deploy", True, "deploy artifacts created", {})

		# Docs
		self._write_text(os.path.join(run_dir, "README.md"), self._template_readme(prompt))
		log("docs", True, "readme generated", {})

		self.rag.index_text(f"Run for: {prompt}")

		report = {
			"summary": "Autonomous build scaffolded successfully",
			"routing": [
				{"stage": "requirements", "provider": req.get("provider"), "model": req.get("model"), "reason": "planning", "fallbackUsed": req.get("fallback", False)},
				{"stage": "design", "provider": design.get("provider"), "model": design.get("model"), "reason": "design", "fallbackUsed": design.get("fallback", False)},
			],
			"artifacts": {
				"backend": {"paths": [os.path.join(run_dir, "backend", "main.py")]},
				"frontend": {"paths": [os.path.join(run_dir, "frontend", "app.py")]},
				"tests": {"paths": [os.path.join(run_dir, "tests", "test_smoke.py")]},
				"infra": {"dockerfile": os.path.join(run_dir, "Dockerfile"), "compose": os.path.join(run_dir, "docker-compose.yml"), "other": []},
				"docs": {"readme": os.path.join(run_dir, "README.md"), "design": os.path.join(run_dir, "design.md")},
			},
			"commands": {
				"setup": ["pip install -r requirements.txt"],
				"run": ["uvicorn backend.main:app --reload", "streamlit run frontend/app.py"],
				"test": ["pytest -q"],
				"deploy": ["docker compose up --build"],
			},
			"logs": logs,
			"nextActions": ["Customize generated code to your domain needs"]
		}
		self._write_text(os.path.join(run_dir, "run_report.json"), json.dumps(report, indent=2))
		return report

	def _write_text(self, path: str, content: str) -> None:
		os.makedirs(os.path.dirname(path), exist_ok=True)
		with open(path, "w", encoding="utf-8") as f:
			f.write(content)

	def _template_fastapi(self, prompt: str) -> str:
		return """
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Generated Service")

class Task(BaseModel):
	input: str

@app.get('/health')
def health():
	return {'status': 'ok'}

@app.post('/run')
def run(task: Task):
	# Stub using the input; replace with domain logic.
	return {'echo': task.input}
"""

	def _template_streamlit(self, prompt: str) -> str:
		return """
import streamlit as st
import requests

st.title('Generated UI')

def get_backend_default():
	return 'http://127.0.0.1:8000'

if 'backend_url' not in st.session_state:
	st.session_state.backend_url = get_backend_default()

with st.sidebar:
	st.subheader('Settings')
	st.session_state.backend_url = st.text_input('Backend URL', value=st.session_state.backend_url)
	backend = st.session_state.backend_url
	status = 'unknown'
	try:
		r = requests.get(f"{backend}/health", timeout=5)
		status = 'online' if r.ok else 'unreachable'
	except Exception:
		status = 'offline'
	st.caption(f'API status: {status}')

text = st.text_input('Enter input')
if st.button('Run', disabled=(status != 'online')):
	try:
		resp = requests.post(f"{backend}/run", json={'input': text}, timeout=30)
		st.json(resp.json())
	except Exception:
		st.error('Cannot reach backend. Check the Backend URL and that the API is running.')
"""

	def _template_test(self) -> str:
		return (
			"def test_placeholder():\n\tassert True\n"
		)

	def _template_dockerfile(self) -> str:
		return (
			"FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt ./\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nEXPOSE 8000 8501\n"
		)

	def _template_compose(self) -> str:
		return (
			"services:\n  api:\n    image: generated-api\n    build: .\n    command: uvicorn backend.main:app --host 0.0.0.0 --port 8000\n    ports:\n      - '8000:8000'\n  ui:\n    image: generated-ui\n    build: .\n    command: streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0\n    ports:\n      - '8501:8501'\n    depends_on:\n      - api\n"
		)

	def _template_readme(self, prompt: str) -> str:
		return (
			"# Generated Project\n\n"
			f"Prompt: {prompt}\n\n"
			"Run:\n\n"
			"- pip install -r requirements.txt\n- uvicorn backend.main:app --reload\n- streamlit run frontend/app.py\n"
		)


