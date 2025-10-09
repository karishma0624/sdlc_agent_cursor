from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import threading
import uuid
import io
from PIL import Image
import base64
import os
from datetime import datetime
import logging
from dotenv import load_dotenv

from .services.adapters import InferenceRouter
from .services.storage import DatabaseService
from .services.rag import RagService
from .services.agent import AgentOrchestrator
from .services.sdlc_builder import SDLCBuilder
from .services.classify import classify_prompt, pick_tool
from .services.security import scrub_files, append_audit
import json
import pathlib
import subprocess
import shlex

app = FastAPI(title="Autonomous SDLC Agent API")
logger = logging.getLogger("uvicorn.error")


app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


class PredictResponse(BaseModel):
	label: str
	confidence: float
	image_id: Optional[int]
	log_id: Optional[int]
	run_report: Dict[str, Any]


class TextTask(BaseModel):
	prompt: str


router = InferenceRouter()
db = DatabaseService()
rag = RagService(db)
agent = AgentOrchestrator()
builder = SDLCBuilder()

# --- Simple background job registry for SDLC builds ---
_BUILD_JOBS_LOCK = threading.Lock()
_BUILD_JOBS: Dict[str, Dict[str, Any]] = {}


@app.get("/health")
def health() -> Dict[str, str]:
	logger.info("health check")
	return {"status": "ok"}


@app.get("/logs")
def list_logs(limit: int = 50) -> Dict[str, Any]:
	from .services.security import scrub_secrets_in_text
	items = db.list_logs(limit)
	for it in items:
		if isinstance(it.get("message"), str):
			it["message"] = scrub_secrets_in_text(it["message"]) 
		if isinstance(it.get("metadata"), dict):
			# Shallow redact values
			it["metadata"] = {k: (scrub_secrets_in_text(v) if isinstance(v, str) else v) for k, v in it["metadata"].items()}
	return {"items": items}


@app.get("/providers")
def providers() -> Dict[str, Any]:
	"""Return detected provider availability and relevant environment hints (redacted)."""
	router.refresh()
	info = {
		"providers": router.providers,
		"env": {
			"OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
			"GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY")),
			"MISTRAL_API_KEY": bool(os.getenv("MISTRAL_API_KEY")),
			"GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
			"HUGGINGFACE_API_KEY": bool(os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_API_KEY")),
			"PERPLEXITY_API_KEY": bool(os.getenv("PERPLEXITY_API_KEY")),
			"V0_API_KEY": bool(os.getenv("V0_API_KEY") or os.getenv("V0_DEV_API_KEY")),
			"OLLAMA_BASE_URL": bool(os.getenv("OLLAMA_BASE_URL")),
		},
	}
	return info


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), notes: Optional[str] = Form(None), request: Request = None):
	logger.info("/predict called: filename=%s, client=%s", file.filename, getattr(getattr(request, 'client', None), 'host', None))
	image_bytes = await file.read()
	image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

	# Route to provider
	result = router.classify_image(image)
	logger.info("classification result: provider=%s model=%s label=%s conf=%.3f", result.get("provider"), result.get("model"), result.get("label"), result.get("confidence", 0.0))

	# Persist image and prediction
	image_id = db.save_image(image_bytes, file.filename)
	pred_id = db.save_prediction(image_id=image_id, label=result["label"], confidence=result["confidence"]) 
	logger.info("saved image_id=%s prediction_id=%s", image_id, pred_id)

	# Log operation + RAG index
	log_id = db.save_log(
		stage="predict",
		provider=result.get("provider"),
		model=result.get("model"),
		success=True,
		message="prediction completed",
		metadata=result,
	)
	rag.index_text(f"Prediction: {result['label']} conf={result['confidence']}")

	run_report = {
		"summary": "Image classified",
		"routing": [
			{
				"stage": "build",
				"provider": result.get("provider"),
				"model": result.get("model"),
				"reason": "vision classification",
				"fallbackUsed": result.get("fallback", False),
			}
		],
		"artifacts": {"backend": {"paths": []}},
		"commands": {"setup": [], "run": [], "test": [], "deploy": []},
		"logs": [
			{
				"timestamp": datetime.utcnow().isoformat(),
				"stage": "predict",
				"success": True,
				"message": "classification done",
				"metadata": {"filename": file.filename},
			}
		],
		"nextActions": [],
	}

	return PredictResponse(
		label=result["label"],
		confidence=result["confidence"],
		image_id=image_id,
		log_id=log_id,
		run_report=run_report,
	)


@app.post("/task")
def run_text_task(req: TextTask, request: Request = None) -> Dict[str, Any]:
	logger.info("/task called: client=%s", getattr(getattr(request, 'client', None), 'host', None))
	# Prefer non-OpenAI providers first to utilize all configured keys
	preference = ["mistral", "groq", "hf", "ollama", "openai", "gemini", "perplexity"]
	response = router.generate_text(req.prompt, preference=preference)
	log_id = db.save_log(
		stage="task",
		provider=response.get("provider"),
		model=response.get("model"),
		success=True,
		message="text task completed",
		metadata=response,
	)
	rag.index_text(f"Task: {req.prompt}\nOutput: {response.get('output','')}")
	logger.info("text task: provider=%s model=%s log_id=%s", response.get("provider"), response.get("model"), log_id)
	return {
		"output": response.get("output"),
		"log_id": log_id,
	}


class BuildRequest(BaseModel):
	prompt: str


@app.post("/build")
def build_project(req: BuildRequest, request: Request = None) -> Dict[str, Any]:
	logger.info("/build called: client=%s prompt_len=%s", getattr(getattr(request, 'client', None), 'host', None), len(req.prompt or ""))
	report = agent.run(req.prompt)
	logger.info("/build completed: summary=%s", report.get("summary"))
	return report


class SDLCBuildRequest(BaseModel):
    prompt: str


@app.post("/sdlc/build")
def sdlc_build(req: SDLCBuildRequest, request: Request = None) -> Dict[str, Any]:
    logger.info("/sdlc/build called: client=%s prompt_len=%s", getattr(getattr(request, 'client', None), 'host', None), len(req.prompt or ""))

    job_id = uuid.uuid4().hex
    started_at = datetime.utcnow().isoformat()

    with _BUILD_JOBS_LOCK:
        _BUILD_JOBS[job_id] = {
            "job_id": job_id,
            "status": "running",
            "prompt_len": len(req.prompt or ""),
            "started_at": started_at,
            "finished_at": None,
            "run_dir": None,
            "error": None,
        }

    def _run_build(job_id: str, prompt: str) -> None:
        try:
            logger.info("[job %s] build started", job_id)
            result = builder.build(prompt)
            run_dir = result.get("run_dir")
            # Persist status to run directory
            try:
                if run_dir:
                    status_path = pathlib.Path(run_dir) / "status.json"
                    status_obj = {
                        "job_id": job_id,
                        "status": "completed",
                        "started_at": started_at,
                        "finished_at": datetime.utcnow().isoformat(),
                        "run_dir": run_dir,
                        "summary": result.get("summary"),
                    }
                    status_path.write_text(json.dumps(status_obj, indent=2), encoding="utf-8")
            except Exception as e:
                logger.warning("[job %s] failed to write status.json: %s", job_id, e)
            finally:
                with _BUILD_JOBS_LOCK:
                    if job_id in _BUILD_JOBS:
                        _BUILD_JOBS[job_id]["status"] = "completed"
                        _BUILD_JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat()
                        _BUILD_JOBS[job_id]["run_dir"] = run_dir
            logger.info("/sdlc/build completed: dir=%s job_id=%s", run_dir, job_id)
        except Exception as e:
            logger.exception("[job %s] build failed: %s", job_id, e)
            with _BUILD_JOBS_LOCK:
                if job_id in _BUILD_JOBS:
                    _BUILD_JOBS[job_id]["status"] = "failed"
                    _BUILD_JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat()
                    _BUILD_JOBS[job_id]["error"] = str(e)

    t = threading.Thread(target=_run_build, args=(job_id, req.prompt), daemon=True)
    t.start()

    return {"status": "build_started", "job_id": job_id}


@app.get("/sdlc/status")
def sdlc_status(job_id: Optional[str] = None) -> Dict[str, Any]:
    with _BUILD_JOBS_LOCK:
        if job_id:
            job = _BUILD_JOBS.get(job_id)
            return job or {"error": "job_not_found", "job_id": job_id}
        # Return latest job by started_at
        if not _BUILD_JOBS:
            return {"jobs": []}
        latest = max(_BUILD_JOBS.values(), key=lambda j: j.get("started_at") or "")
        return {"latest": latest, "jobs_count": len(_BUILD_JOBS)}


@app.get("/sdlc/report")
def sdlc_report(job_id: Optional[str] = None, run_dir: Optional[str] = None) -> Dict[str, Any]:
    """Return the run_report.json for a completed build.

    One of job_id or run_dir must be provided.
    """
    target_dir: Optional[str] = None
    if job_id:
        with _BUILD_JOBS_LOCK:
            job = _BUILD_JOBS.get(job_id)
            if job and isinstance(job.get("run_dir"), str):
                target_dir = job.get("run_dir")
    if not target_dir and run_dir:
        target_dir = run_dir
    if not target_dir:
        return {"error": "missing_job_id_or_run_dir"}
    report_path = pathlib.Path(str(target_dir)) / "run_report.json"
    if not report_path.exists():
        return {"error": "report_not_found", "run_dir": str(target_dir)}
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        return data
    except Exception as e:
        return {"error": "report_read_error", "message": str(e)}


class DispatchRequest(BaseModel):
	prompt: str
	free_only: bool = True


@app.post("/dispatch")
def dispatch(req: DispatchRequest, request: Request = None) -> Dict[str, Any]:
	kind = classify_prompt(req.prompt)
	tool = pick_tool(kind, req.free_only)
	logger.info("/dispatch: kind=%s tool=%s", kind, tool)
	result = router.run_tool(tool, req.prompt)
	if isinstance(result, dict) and "files" in result:
		result["files"] = scrub_files(result["files"])
	append_audit("dispatch", {"kind": kind, "tool": tool})
	return {
		"kind": kind,
		"tool": tool,
		"result": result,
	}


class MaterializeRequest(BaseModel):
	prompt: str
	free_only: bool = True


@app.post("/materialize")
def materialize(req: MaterializeRequest) -> Dict[str, Any]:
	kind = classify_prompt(req.prompt)
	tool = pick_tool(kind, req.free_only)
	result = router.run_tool(tool, req.prompt)
	files = result.get("files") if isinstance(result, dict) else None
	if not files:
		return {"error": "No files returned by tool", "kind": kind, "tool": tool}
	files = scrub_files(files)
	# write to runs/
	root = pathlib.Path("runs")
	root.mkdir(parents=True, exist_ok=True)
	dirname = datetime.utcnow().strftime("preview-%Y%m%d-%H%M%S")
	outdir = root / dirname
	for path, content in files.items():
		p = outdir / path
		p.parent.mkdir(parents=True, exist_ok=True)
		p.write_text(content, encoding="utf-8")
	append_audit("materialize", {"dir": str(outdir), "kind": kind, "tool": tool})
	commands = [
		"cd " + str(outdir).replace("\\", "/"),
		"npm install (if package.json exists)",
		"npm run dev (for Next/Vite) or npx serve .",
	]
	return {"kind": kind, "tool": tool, "outdir": str(outdir), "commands": commands}


@app.post("/diagnostics")
def diagnostics() -> Dict[str, Any]:
	"""Run pytest (with JSON report) and flake8, returning summarized results.
	Note: For full isolation, run in Docker as per compose. This runs locally with timeouts.
	"""
	logs_dir = pathlib.Path("logs")
	logs_dir.mkdir(parents=True, exist_ok=True)
	pytest_json = logs_dir / "pytest-report.json"

	def run_cmd(cmd: str, timeout: int = 120) -> Dict[str, Any]:
		try:
			proc = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False, text=True)
			return {"code": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}
		except subprocess.TimeoutExpired:
			return {"code": -1, "stdout": "", "stderr": "timeout"}

	pytest_cmd = f"pytest -q --maxfail=1 --disable-warnings --json-report --json-report-file={pytest_json.as_posix()}"
	flake8_cmd = "flake8"
	pytest_res = run_cmd(pytest_cmd, timeout=180)
	flake8_res = run_cmd(flake8_cmd, timeout=120)

	pytest_report: Dict[str, Any] = {}
	if pytest_json.exists():
		try:
			pytest_report = json.loads(pytest_json.read_text(encoding="utf-8"))
		except Exception:
			pytest_report = {}

	return {
		"run_id": datetime.utcnow().strftime("diag-%Y%m%d-%H%M%S"),
		"timestamp": datetime.utcnow().isoformat(),
		"checks": {
			"pytest": {"exit_code": pytest_res["code"], "summary": pytest_report.get("summary", {}), "paths": {"json": pytest_json.as_posix()}},
			"flake8": {"exit_code": flake8_res["code"], "stderr": flake8_res.get("stderr", "")[:1000]},
		},
		"model_used": {},
		"actions": [pytest_cmd, flake8_cmd],
		"human_approvals": [],
	}


class SaveSpecRequest(BaseModel):
	content: str
	filename: str | None = None


@app.post("/save_spec")
def save_spec(req: SaveSpecRequest) -> Dict[str, Any]:
	root = pathlib.Path("runs")
	root.mkdir(parents=True, exist_ok=True)
	name = req.filename or ("requirements-" + datetime.utcnow().strftime("%Y%m%d-%H%M%S") + ".md")
	path = root / name
	path.write_text(req.content or "", encoding="utf-8")
	append_audit("save_spec", {"path": str(path)})
	return {"path": str(path)}


@app.on_event("startup")
def _on_startup() -> None:
	load_dotenv()
	logging.basicConfig(level=logging.INFO)
	logger.info("API startup: DB=%s, RAG optional=%s", os.getenv("PRIMARY_DB_URL", "sqlite:///app.db"), "enabled")
	router.refresh()
	agent.router.refresh()


# --- CRUD: Students ---
class StudentIn(BaseModel):
	name: str
	email: Optional[str] = None


@app.post("/students")
def create_student(payload: StudentIn) -> Dict[str, Any]:
	id_ = db.create_student(payload.name, payload.email)
	return {"id": id_}


@app.get("/students")
def list_students(limit: int = 100) -> Dict[str, Any]:
	return {"items": db.list_students(limit)}


@app.get("/students/{student_id}")
def get_student(student_id: int) -> Dict[str, Any]:
	obj = db.get_student(student_id)
	if not obj:
		return {"error": "not found"}
	return obj


@app.put("/students/{student_id}")
def update_student(student_id: int, payload: StudentIn) -> Dict[str, Any]:
	success = db.update_student(student_id, payload.name, payload.email)
	return {"success": bool(success)}


@app.delete("/students/{student_id}")
def delete_student(student_id: int) -> Dict[str, Any]:
	success = db.delete_student(student_id)
	return {"success": bool(success)}


# --- CRUD: Events ---
class EventIn(BaseModel):
	title: str
	description: Optional[str] = None
	date: Optional[str] = None


@app.post("/events")
def create_event(payload: EventIn) -> Dict[str, Any]:
	id_ = db.create_event(payload.title, payload.description, payload.date)
	return {"id": id_}


@app.get("/events")
def list_events(limit: int = 100) -> Dict[str, Any]:
	return {"items": db.list_events(limit)}


@app.get("/events/{event_id}")
def get_event(event_id: int) -> Dict[str, Any]:
	obj = db.get_event(event_id)
	if not obj:
		return {"error": "not found"}
	return obj


@app.put("/events/{event_id}")
def update_event(event_id: int, payload: EventIn) -> Dict[str, Any]:
	success = db.update_event(event_id, payload.title, payload.description, payload.date)
	return {"success": bool(success)}


@app.delete("/events/{event_id}")
def delete_event(event_id: int) -> Dict[str, Any]:
	success = db.delete_event(event_id)
	return {"success": bool(success)}


# --- Requirements JSON storage ---
class RequirementIn(BaseModel):
	title: str
	content: Dict[str, Any]


@app.post("/requirements")
def create_requirement(payload: RequirementIn) -> Dict[str, Any]:
	id_ = db.create_requirement(payload.title, payload.content)
	return {"id": id_}


@app.get("/requirements")
def list_requirements(limit: int = 100) -> Dict[str, Any]:
	return {"items": db.list_requirements(limit)}


@app.get("/requirements/{req_id}")
def get_requirement(req_id: int) -> Dict[str, Any]:
	obj = db.get_requirement(req_id)
	if not obj:
		return {"error": "not found"}
	return obj


@app.put("/requirements/{req_id}")
def update_requirement(req_id: int, payload: RequirementIn) -> Dict[str, Any]:
	success = db.update_requirement(req_id, payload.title, payload.content)
	return {"success": bool(success)}


@app.delete("/requirements/{req_id}")
def delete_requirement(req_id: int) -> Dict[str, Any]:
	success = db.delete_requirement(req_id)
	return {"success": bool(success)}


