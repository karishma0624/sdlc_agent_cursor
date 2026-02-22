# backend/services/supabase_client.py
"""
Supabase REST API client.
Persists real user prompts and AI responses to Supabase tables automatically.
"""

import os
import requests
from typing import Optional
from datetime import datetime

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")


def _headers() -> dict:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _post(table: str, payload: dict) -> Optional[dict]:
    """POST a row to a Supabase table. Returns the created row or None on failure."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=_headers(),
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return data[0] if isinstance(data, list) and data else data
    except Exception as e:
        print(f"[Supabase] POST /{table} failed: {e}")
        return None


def _patch(table: str, row_id: str, payload: dict) -> Optional[dict]:
    """PATCH (update) a row by id in a Supabase table."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        resp = requests.patch(
            f"{SUPABASE_URL}/rest/v1/{table}?id=eq.{row_id}",
            headers=_headers(),
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return data[0] if isinstance(data, list) and data else data
    except Exception as e:
        print(f"[Supabase] PATCH /{table}/{row_id} failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Public helpers — called by main.py and simple_builder.py
# ---------------------------------------------------------------------------

def create_project(name: str, description: str = "") -> Optional[str]:
    """Insert a row into `projects`. Returns the new project UUID."""
    row = _post("projects", {"name": name, "description": description})
    if row:
        print(f"[Supabase] Project created: {row.get('id')}")
        return row.get("id")
    return None


def create_session(
    project_id: str,
    status: str = "idle",
    current_phase: str = "planning",
    backend_model: Optional[str] = None,
    frontend_model: Optional[str] = None,
    design_model: Optional[str] = None,
) -> Optional[str]:
    """Insert a row into `sessions`. Returns the new session UUID."""
    row = _post(
        "sessions",
        {
            "project_id": project_id,
            "status": status,
            "current_phase": current_phase,
            "iteration_count": 0,
            "selected_backend_model": backend_model,
            "selected_frontend_model": frontend_model,
            "selected_design_model": design_model,
        },
    )
    if row:
        print(f"[Supabase] Session created: {row.get('id')}")
        return row.get("id")
    return None


def update_session(
    session_id: str,
    status: Optional[str] = None,
    current_phase: Optional[str] = None,
    iteration_count: Optional[int] = None,
) -> None:
    """PATCH the session row (status / phase / iteration)."""
    payload = {"updated_at": datetime.utcnow().isoformat()}
    if status is not None:
        payload["status"] = status
    if current_phase is not None:
        payload["current_phase"] = current_phase
    if iteration_count is not None:
        payload["iteration_count"] = iteration_count
    _patch("sessions", session_id, payload)


def save_message(session_id: str, role: str, content: str) -> Optional[str]:
    """Insert a row into `messages` (role = 'user' | 'assistant' | 'system')."""
    row = _post(
        "messages",
        {
            "session_id": session_id,
            "role": role,
            "content": content,
            "attachments": [],
        },
    )
    if row:
        return row.get("id")
    return None


def save_execution_log(
    session_id: str,
    phase: str,
    provider: Optional[str] = None,
    success: bool = True,
    error_output: Optional[str] = None,
    iteration: int = 1,
) -> None:
    """Insert a row into `execution_logs`."""
    _post(
        "execution_logs",
        {
            "session_id": session_id,
            "phase": phase,
            "provider": provider,
            "success": success,
            "error_output": error_output,
            "iteration": iteration,
        },
    )


def save_provider_usage(
    session_id: str,
    provider_name: str,
    model_name: Optional[str] = None,
    tokens_used: Optional[int] = None,
    phase: Optional[str] = None,
    success: bool = True,
    retries: int = 0,
) -> None:
    """Insert a row into `provider_usage_logs`."""
    _post(
        "provider_usage_logs",
        {
            "session_id": session_id,
            "provider_name": provider_name,
            "model_name": model_name,
            "tokens_used": tokens_used,
            "phase": phase,
            "success": success,
            "retries": retries,
        },
    )


def save_rag_summary(
    project_id: str,
    session_id: str,
    prompt: str,
    summary: str,
    status: str = "completed",
    iteration: int = 1,
) -> Optional[str]:
    """Insert a row into `rag_summaries`."""
    row = _post(
        "rag_summaries",
        {
            "project_id": project_id,
            "session_id": session_id,
            "prompt": prompt,
            "summary": summary,
            "status": status,
            "iteration": iteration,
        },
    )
    if row:
        return row.get("id")
    return None
