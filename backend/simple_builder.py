import os
import json
import traceback
import subprocess
import threading
import time
import re
import requests
from datetime import datetime
from datetime import datetime
from typing import Dict, Any, Optional

# Robust Import for Services
try:
    from backend.services.adapters import call_gemini_text, _parse_json_garbage
except ImportError:
    try:
        from services.adapters import call_gemini_text, _parse_json_garbage
    except ImportError:
        try:
            from .services.adapters import call_gemini_text, _parse_json_garbage
        except ImportError:
            # Last ditch: sys path hack
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
            from services.adapters import call_gemini_text, _parse_json_garbage

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
        self.preview_processes = {}  # job_id -> subprocess.Popen

    def analyze_intent(self, prompt: str) -> Dict[str, Any]:
        """
        Classifies intent using Mistral/Groq.
        Extracts requirements if web_app.
        """
        system_prompt = (
            "You are an Intent Classifier for an SDLC Agent.\n"
            "Classify the user prompt into exactly one of: web_app, content, debug, ml_task, design, general.\n"
            "Rules:\n"
            "- 'web_app': Requests for a website, app, dashboard, tool with frontend/backend, or software system.\n"
            "- 'content': Requests for emails, essays, letters, poems, or text generation ONLY.\n"
            "- 'debug': Requests to fix specific code snippets or errors provided in the prompt.\n"
            "- 'other': Anything else.\n\n"
            "Return JSON ONLY:\n"
            "{\n"
            "  \"intent\": \"web_app|content|debug|other\",\n"
            "  \"reasoning\": \"brief explanation\",\n"
            "  \"requirements_summary\": \"Extracted requirements if web_app, else null\"\n"
            "}\n"
            f"User Prompt: {prompt}"
        )
        
        try:
            from services.adapters import call_mistral
            resp = call_mistral(system_prompt)
            return _parse_json_garbage(resp)
        except Exception as e:
            try:
                # Fallback to Gemini text
                resp = call_gemini_text(system_prompt, model="gemini-1.5-flash")
                return _parse_json_garbage(resp)
            except:
                print(f"Intent Analysis Failed: {e}")
                # Fallback to web_app if ambiguous but looks like a build request, else general
                if "build" in prompt.lower() or "create" in prompt.lower() or "app" in prompt.lower():
                    return {"intent": "web_app", "requirements_summary": prompt}
                return {"intent": "content", "reasoning": "Fallback to content due to error"}

    def preview_frontend(self, job_id: str) -> Dict[str, Any]:
        """
        1. Validate env. 2. npm install. 3. npm run dev. 4. Return URL.
        """
        run_dir = os.path.join(self.runs_dir, job_id)
        frontend_dir = os.path.join(run_dir, "frontend")
        
        if not os.path.exists(frontend_dir):
            return {"status": "failed", "error": "Frontend directory not found."}
            
        # Check if package.json exists
        if not os.path.exists(os.path.join(frontend_dir, "package.json")):
             return {"status": "failed", "error": "No package.json found in frontend."}

        # Kill existing if any (by our tracking map)
        if job_id in self.preview_processes:
            try:
                import psutil
                proc = self.preview_processes[job_id]
                parent = psutil.Process(proc.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
                proc.wait(timeout=2) 
            except: pass
            del self.preview_processes[job_id]
            
        import socket
        try:
             sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
             sock.bind(('127.0.0.1', 0))
             free_port = sock.getsockname()[1]
             sock.close()
        except:
             free_port = 5173

        try:
            # NPM INSTALL (Only run if node_modules is missing for speed)
            if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
                print(f"[{job_id}] Running npm install...")
                subprocess.run("npm install", shell=True, cwd=frontend_dir, check=True, capture_output=True)
            
            # RUN VITE on a dynamically found free port to completely avoid collisions with old projects
            # Using npx vite to avoid NPM argument parsing bugs
            proc = subprocess.Popen(
                f"npx vite --host 127.0.0.1 --port {free_port} --strictPort", 
                shell=True, 
                cwd=frontend_dir, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            self.preview_processes[job_id] = proc
            
            url = f"http://127.0.0.1:{free_port}"
            
            # Wait up to 5 seconds to ensure Vite has started, then return the URL
            start = time.time()
            import threading, queue
            
            q = queue.Queue()
            def read_output():
                for line in iter(proc.stdout.readline, ''):
                    q.put(line)
                    
            t = threading.Thread(target=read_output, daemon=True)
            t.start()
            
            # We don't even strictly need to regex match the port since we forced it via --port.
            # But we poll queue so it drains and gives it time to bind.
            while time.time() - start < 3:
                try:
                    line = q.get(timeout=0.5)
                    if "ready in" in line or "Local:" in line:
                         break
                except queue.Empty:
                    if proc.poll() is not None:
                        # Process died early (e.g. syntax error in generated code)
                        return {"status": "failed", "error": f"Vite dev server failed to start: {proc.stdout.read()}"}
                
            return {"status": "running", "url": url}

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def init_run(self, prompt: str, job_id: Optional[str] = None, mode: str = "auto") -> str:
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

        # --- MODE / INTENT HANDLING ---
        # The user wanted a smart agent model to decide, so we force Analyze Intent unless strictly overridden.
        mode = status.get("mode", "auto")
        
        # If mode is explicitly content, we bypass. Otherwise, analyze intelligently!
        if mode == "content":
            intent = "content"
        else:
            # Auto-detect intent using LLM (Mistral/Gemini)
            if "intent_data" not in status:
                intent_data = self.analyze_intent(prompt)
                status["intent_data"] = intent_data
                self._save_json(run_dir, "status.json", status)
            else:
                intent_data = status["intent_data"]
                
            intent = intent_data.get("intent", "web_app")
            # If the user explicitly chose web_app but the LLM strongly thinks it's content (e.g., leave letter)
            # we trust the LLM, as requested by the user.

        self._log_event(run_dir, "system", f"Mode: {mode}, Computed Intent: {intent}")
        
        if intent == "content":
            # Direct content generation
            self._update_status(run_dir, job_id, "content", "running", "Generating content...")
            from services.adapters import call_mistral
            content = call_mistral(f"You are a professional content generator. Provide ONLY the detailed text for the user's request. Request: {prompt}")
            
            # Save artifact
            with open(os.path.join(run_dir, "output.md"), "w") as f: f.write(content)
            
            self._update_status(run_dir, job_id, "content", "completed", "Content generated.")
            self._log_chat_message(run_dir, "agent", f"✅ Content Generated:\n\n{content}")
            return {"status": "completed", "summary": "Content Generated"}
            
        elif intent == "debug":
             # Debug mode (Simplification: just analyze files if attached, or prompt)
            self._update_status(run_dir, job_id, "debug", "running", "Debugging...")
            # For now, just a placeholder acknowledgement.
            self._log_chat_message(run_dir, "agent", "Debug mode initiated. Please check local files.")
            self._update_status(run_dir, job_id, "debug", "completed", "Debug info logged.")
            return {"status": "completed", "summary": "Debug Completed"}

        elif intent != "web_app":
             self._log_chat_message(run_dir, "agent", f"Intent '{intent}' not fully supported yet.")
             return {"status": "completed", "summary": "Skipped"}
             
        # --- WEB APP SDLC ---

        # As requested: after frontend generation the backend should be automatically done.
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
                output_path = os.path.join(run_dir, "frontend")
                
                if phase == "frontend":
                     success_msg = f"✅ **Frontend Generated!**\n\nPreview available via the 'Preview App' button."
                elif phase == "backend":
                     success_msg = f"✅ **Backend Generated!**\n\nAPI available via `uvicorn backend.main:app`."
                else:
                     success_msg = f"✅ **{phase.title()} Phase Completed Successfully!**"

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
