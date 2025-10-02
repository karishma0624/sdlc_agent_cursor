from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import io
from PIL import Image
import base64
import os
from datetime import datetime
import logging

from .services.adapters import InferenceRouter
from .services.storage import DatabaseService
from .services.rag import RagService
from .services.agent import AgentOrchestrator

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


@app.get("/health")
def health() -> Dict[str, str]:
	logger.info("health check")
	return {"status": "ok"}


@app.get("/logs")
def list_logs(limit: int = 50) -> Dict[str, Any]:
	items = db.list_logs(limit)
	return {"items": items}


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
	response = router.generate_text(req.prompt)
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


@app.on_event("startup")
def _on_startup() -> None:
	logging.basicConfig(level=logging.INFO)
	logger.info("API startup: DB=%s, RAG optional=%s", os.getenv("PRIMARY_DB_URL", "sqlite:///app.db"), "enabled")


