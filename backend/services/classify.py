from __future__ import annotations
from typing import Literal
import os

TaskKind = Literal[
	"BUILD_FULLSTACK",
	"FRONTEND_ONLY",
	"BACKEND_ONLY",
	"CODE_SNIPPET",
	"REQUIREMENTS",
	"PREDICT",
	"OTHER_TEXT",
]


def classify_prompt(prompt: str) -> TaskKind:
	text = (prompt or "").lower()
	if any(k in text for k in ["fullstack", "frontend and backend", "end-to-end", "full stack"]):
		return "BUILD_FULLSTACK"
	if any(k in text for k in ["frontend only", "ui only", "react", "tailwind", "html", "css"]):
		return "FRONTEND_ONLY"
	if any(k in text for k in ["backend only", "api only", "fastapi", "flask", "node", "express"]):
		return "BACKEND_ONLY"
	if any(k in text for k in ["code snippet", "snippet", "example class", "function only", "method only"]):
		return "CODE_SNIPPET"
	if any(k in text for k in ["requirements", "spec", "acceptance criteria", "design doc"]):
		return "REQUIREMENTS"
	if any(k in text for k in ["predict", "classify", "inference", "train", "dataset"]):
		return "PREDICT"
	return "OTHER_TEXT"


def pick_tool(kind: TaskKind, free_only: bool = True) -> str:
	"""Pick external tool. Prefer v0.dev for frontend/fullstack when key present."""
	v0_available = bool(os.getenv("V0_API_KEY") or os.getenv("V0_DEV_API_KEY"))
	if kind in ("BUILD_FULLSTACK", "FRONTEND_ONLY"):
		if v0_available:
			return "v0"
		# fallbacks
		return "stitch" if free_only else "lovable"
	# other categories use local codegen/LLMs path
	return "local_codegen"


