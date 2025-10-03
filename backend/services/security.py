from __future__ import annotations
from typing import Dict, Any
import os
import re
from datetime import datetime


SECRET_PATTERNS = [
	re.compile(r"sk-[a-zA-Z0-9\-_]{20,}"),
	re.compile(r"(?:AIza)[A-Za-z0-9_\-]{10,}"),
	re.compile(r"hf_[A-Za-z0-9]{10,}"),
	re.compile(r"pplx-[A-Za-z0-9_\-]{10,}"),
	re.compile(r"gsk_[A-Za-z0-9_\-]{10,}"),
]


def scrub_secrets_in_text(content: str) -> str:
	redacted = content
	for pat in SECRET_PATTERNS:
		redacted = pat.sub("${REDACTED_ENV_VAR}", redacted)
	return redacted


def scrub_files(files: Dict[str, str]) -> Dict[str, str]:
	return {path: scrub_secrets_in_text(text) for path, text in files.items()}


def append_audit(action: str, metadata: Dict[str, Any]) -> None:
	line = {
		"timestamp": datetime.utcnow().isoformat(),
		"action": action,
		"metadata": metadata,
	}
	log_path = os.path.join(os.getcwd(), "audit.log")
	with open(log_path, "a", encoding="utf-8") as f:
		f.write(str(line) + "\n")


