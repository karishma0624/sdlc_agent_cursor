import os
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional

# DB Integration
try:
    from backend.db import db
except ImportError:
    try:
        from db import db
    except ImportError:
        from .db import db

# IMPORT NEW PHASES
try:
    from backend.phases.requirements import execute_requirements_phase
    from backend.phases.planning import execute_planning_phase
    from backend.phases.design import execute_design_phase
    from backend.phases.frontend import execute_frontend_phase
    from backend.phases.backend import execute_backend_phase
except ImportError:
    # Fallback if running from backend dir directly
    try:
        from phases.requirements import execute_requirements_phase
        from phases.planning import execute_planning_phase
        from phases.design import execute_design_phase
        from phases.frontend import execute_frontend_phase
        from phases.backend import execute_backend_phase
    except ImportError:
         # Relative import attempt
        from .phases.requirements import execute_requirements_phase
        from .phases.planning import execute_planning_phase
        from .phases.design import execute_design_phase
        from .phases.frontend import execute_frontend_phase
        from .phases.backend import execute_backend_phase

class SDLCBuilder:
    """
    STRICT Deterministic SDLC Orchestrator.
    Phases:
    1. Requirements (Mistral)
    2. Planning (Mistral)
    3. Design (Mermaid)
    4. Frontend (Gemini -> v0)
    4. Frontend (Gemini -> v0)
    5. Backend (Gemini)
    STOP.
    """

    def __init__(self, runs_dir: str = "runs"):
        self.runs_dir = runs_dir
        os.makedirs(self.runs_dir, exist_ok=True)

    def init_run(self, prompt: str, job_id: Optional[str] = None) -> str:
        if not job_id:
            import uuid
            job_id = uuid.uuid4().hex
            
        # Initialize Supabase Session
        if db.enabled:
            try:
                # Create project first (using prompt as name)
                project_id = db.create_project(name=prompt[:50], description=prompt)
                if not project_id:
                    project_id = "00000000-0000-0000-0000-000000000000"  # Fallback
                
                # Create session with job_id as the session ID
                db.create_session(session_id=job_id, project_id=project_id)
            except Exception as e:
                print(f"[Supabase] Session init failed: {e}")
        
        run_dir = os.path.join(self.runs_dir, job_id)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def run_build(self, run_dir: str, prompt: str) -> Dict[str, Any]:
        """
        Main entry point. Orchestrates the 4-phase strictly.
        """
        job_id = os.path.basename(run_dir)
        status = self._load_status(run_dir)
        
        # Save prompt if new
        if prompt and prompt.strip():
            status["prompt"] = prompt
            self._save_json(run_dir, "status.json", status)
        else:
            prompt = status.get("prompt", "")

        if not prompt:
            return {"status": "failed", "error": "No prompt provided"}

        # Define Strict Phase Order
        phases = ["requirements", "planning", "design", "frontend", "backend"]
        
        self._log_event(run_dir, "system", f"Build triggered: {prompt}")
        if db.enabled:
             # Try to map job_id to session_id strictly or just log event
             # Assuming job_id is UUID compatible
             try:
                db.log_message(job_id, "user", prompt)
             except: pass

        for phase in phases:
            status = self._load_status(run_dir)
            current_status = status.get("phases", {}).get(phase, "waiting")
            
            if current_status == "completed":
                continue
                
            try:
                self._update_status(run_dir, job_id, phase, "running", f"Starting {phase} phase...")
                
                # --- EXECUTE PHASE ---
                if phase == "requirements":
                    execute_requirements_phase(prompt, run_dir)
                    
                elif phase == "planning":
                    # Load requirement artifact to pass as context
                    req_path = os.path.join(run_dir, "requirements.md")
                    if not os.path.exists(req_path): raise ValueError("Requirements missing.")
                    with open(req_path, "r") as f: req_content = f.read()
                    
                    execute_planning_phase(req_content, run_dir)
                    
                elif phase == "design":
                    # Load planning artifact
                    plan_path = os.path.join(run_dir, "planning.md")
                    if not os.path.exists(plan_path): raise ValueError("Planning missing.")
                    with open(plan_path, "r", encoding="utf-8") as f: plan_content = f.read()
                    
                    execute_design_phase(plan_content, run_dir)
                    
                elif phase == "frontend":
                    # Load planning artifact
                    plan_path = os.path.join(run_dir, "planning.md")
                    if not os.path.exists(plan_path): raise ValueError("Planning missing.")
                    with open(plan_path, "r", encoding="utf-8") as f: plan_content = f.read()
                    
                    success, msg = execute_frontend_phase(plan_content, run_dir)
                    if not success:
                        raise RuntimeError(f"Frontend Phase Failed: {msg}")

                elif phase == "backend":
                    # Load planning artifact
                    plan_path = os.path.join(run_dir, "planning.md")
                    if not os.path.exists(plan_path): raise ValueError("Planning missing.")
                    with open(plan_path, "r", encoding="utf-8") as f: plan_content = f.read()
                    
                    success, msg = execute_backend_phase(plan_content, run_dir)
                    if not success:
                        raise RuntimeError(f"Backend Phase Failed: {msg}")

                # --- PHASE COMPLETE ---
                self._update_status(run_dir, job_id, phase, "completed", f"{phase} completed.")
                self._log_event(run_dir, phase, "Success.")
                self._log_event(run_dir, phase, "Success.")
                output_path = os.path.join(run_dir, "frontend")
                success_msg = f"✅ **{phase.title()} Phase Completed Successfully!**\n\n**Output Location:** `{output_path}`"
                self._log_chat_message(run_dir, "agent", success_msg)
                print(f"[{job_id}] {phase} Output stored at: {output_path}")
                
                if db.enabled:
                    try:
                        db.update_session_status(job_id, phase, "completed")
                        db.log_execution(job_id, phase, "auto", True)
                    except: pass
                
            except Exception as e:
                err = f"{phase} failed: {str(e)}"
                traceback.print_exc()
                self._update_status(run_dir, job_id, phase, "failed", err)
                self._log_event(run_dir, "error", f"{phase} CRASH: {traceback.format_exc()}")
                self._log_chat_message(run_dir, "agent", f"❌ **{phase.title()} Phase Failed:** {str(e)}")
                
                if db.enabled:
                    try:
                        db.update_session_status(job_id, phase, "failed")
                        db.log_execution(job_id, phase, "auto", False, str(e))
                    except: pass
                    
                return {"status": "failed", "error": err}

        # STOP Condition
        self._update_status(run_dir, job_id, "complete", "completed", "All 5 phases completed.")
        return {"status": "completed", "summary": "Full Stack Build Successful"}

    # --- HELPERS ---

    def _load_status(self, run_dir: str) -> Dict[str, Any]:
        path = os.path.join(run_dir, "status.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f: return json.load(f)
            except: pass
        return {}

    def _update_status(self, run_dir: str, job_id: str, phase: str, status: str, msg: str):
        path = os.path.join(run_dir, "status.json")
        data = self._load_status(run_dir)
        
        data["job_id"] = job_id
        data["last_updated"] = datetime.now().isoformat()
        if phase != "complete":
            data["current_phase"] = phase
        
        if "phases" not in data:
            data["phases"] = {}
        data["phases"][phase] = status
        
        data["message"] = msg
        data["status"] = "running" if phase != "complete" else "completed"
        if status == "failed": data["status"] = "failed"
        
        if "execution_log" not in data:
            data["execution_log"] = []
        data["execution_log"].append({
            "time": datetime.now().isoformat(),
            "phase": phase,
            "status": status,
            "message": msg
        })
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _log_event(self, run_dir, category, content):
        p = os.path.join(run_dir, "audit.log")
        with open(p, "a") as f:
            f.write(f"[{datetime.now().isoformat()}] [{category}] {content}\n")
    
    def _log_chat_message(self, run_dir: str, role: str, content: str):
        """Log a message to the chat interface"""
        status_path = os.path.join(run_dir, "status.json")
        data = self._load_status(run_dir)
        
        if "messages" not in data:
            data["messages"] = []
        
        data["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        with open(status_path, "w") as f:
            json.dump(data, f, indent=2)


    def _save_json(self, run_dir: str, filename: str, data: Any):
        path = os.path.join(run_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
