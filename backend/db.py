import os
import requests
import json
from datetime import datetime
from typing import Dict, Any, Optional

class SupabaseDB:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") # Prefer Service Role for backend ops
        
        if not self.url or not self.key:
             # Fallback to Anon key if Service Role missing, but warn
             self.key = os.getenv("SUPABASE_ANON_KEY")
             
        if not self.url or not self.key:
            print("[SupabaseDB] Warning: SUPABASE_URL or Keys missing. DB Disabled.")
            self.enabled = False
        else:
            self.enabled = True
            self.headers = {
                "apikey": self.key,
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation"
            }

    def _post(self, table: str, data: Dict[str, Any]) -> Optional[Dict]:
        if not self.enabled: return None
        try:
            resp = requests.post(f"{self.url}/rest/v1/{table}", headers=self.headers, json=data, timeout=5)
            resp.raise_for_status()
            return resp.json()[0] if resp.json() else None
        except Exception as e:
            print(f"[SupabaseDB] Insert {table} failed: {e}")
            return None

    def _update(self, table: str, record_id: str, data: Dict[str, Any]) -> Optional[Dict]:
        if not self.enabled: return None
        try:
            url = f"{self.url}/rest/v1/{table}?id=eq.{record_id}"
            resp = requests.patch(url, headers=self.headers, json=data, timeout=5)
            resp.raise_for_status()
            return resp.json()[0] if resp.json() else None
        except Exception as e:
            print(f"[SupabaseDB] Update {table} {record_id} failed: {e}")
            return None

    # --- SPECIFIC OPERATIONS ---

    def create_project(self, name: str, description: str = "") -> Optional[str]:
        # Assuming user_id is handled/mocked or we use a default if auth not strict yet
        data = {"name": name, "description": description}
        res = self._post("projects", data)
        return res.get("id") if res else None

    def create_session(self, session_id: str, project_id: str, backend_model: str = "mistral", frontend_model: str = "gemini") -> Optional[str]:
        data = {
            "id": session_id,  # Use provided job_id as session ID
            "project_id": project_id,
            "status": "idle",
            "selected_backend_model": backend_model,
            "selected_frontend_model": frontend_model,
            "selected_design_model": "mermaid"
        }
        res = self._post("sessions", data)
        return res.get("id") if res else None

    def update_session_status(self, session_id: str, phase: str, status: str):
        self._update("sessions", session_id, {"current_phase": phase, "status": status, "updated_at": datetime.now().isoformat()})

    def log_message(self, session_id: str, role: str, content: str):
        data = {"session_id": session_id, "role": role, "content": content}
        self._post("messages", data)

    def log_execution(self, session_id: str, phase: str, provider: str, success: bool, output: str = ""):
        data = {
            "session_id": session_id,
            "phase": phase,
            "provider": provider,
            "success": success,
            "error_output": output if not success else None
        }
        self._post("execution_logs", data)
        
    def save_artifact(self, project_id: str, file_path: str, content: str):
         data = {"project_id": project_id, "file_path": file_path, "content": content}
         self._post("artifacts", data)

# Global Instance
db = SupabaseDB()
