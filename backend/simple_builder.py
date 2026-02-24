import os
import json
import time
import subprocess
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime

from services.adapters import InferenceRouter
import services.supabase_client as supa
from rag.retriever import retrieve_and_merge_context
from rag.rag_service import index_phase_output

class SDLCBuilder:
    """
    The Autonomous SDLC Orchestrator (Upgraded).
    Executes a strict 6-phase pipeline with state persistence and resumability.
    
    Phases:
    1. Planning: Requirements Analysis -> requirements.md
    2. Design: System Architecture -> design.json, architecture.md, flowchart.mmd
    3. Backend: Code Generation -> FastAPI/Node
    4. Frontend: Code Generation -> React/Vite
    5. Testing: Real execution -> pytest, test_report.json
    6. Deployment: Configuration -> Dockerfile, deployment.md
    """

    def __init__(self, runs_dir: str = "runs"):
        self.runs_dir = runs_dir
        self.router = InferenceRouter()
        os.makedirs(self.runs_dir, exist_ok=True)

    def init_run(self, prompt: str, job_id: Optional[str] = None) -> str:
        if not job_id:
            import uuid
            job_id = uuid.uuid4().hex
        run_dir = os.path.join(self.runs_dir, job_id)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def run_build(
        self,
        run_dir: str,
        prompt: str,
        supabase_session_id: Optional[str] = None,
        supabase_project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point. Resumes from the last successful state.
        Automatically persists events to Supabase if session/project IDs are provided.
        """
        job_id = os.path.basename(run_dir)
        status = self._load_status(run_dir)

        # PERSIST PROMPT for Session Names
        if prompt and prompt.strip():
            status["prompt"] = prompt
            self._save_json(run_dir, "status.json", status)

        # Define Phase Order
        phases = ["planning", "design", "backend", "frontend", "tests", "deployment"]
        self._log_event(run_dir, "system", f"Build triggered: {prompt}")

        iteration = 1
        for phase in phases:
            current_phase_status = status.get("phases", {}).get(phase, "waiting")

            # Skip if already completed
            if current_phase_status == "completed":
                iteration += 1
                continue

            try:
                self._update_status(run_dir, job_id, phase, "running", f"Starting {phase} phase...")

                # Retrieve RAG Context
                rag_context = ""
                if supabase_project_id or supabase_session_id:
                    rag_context = retrieve_and_merge_context(
                        query_text=f"Find context for phase: {phase}. Original prompt: {prompt}",
                        project_id=supabase_project_id,
                        session_id=supabase_session_id,
                        iteration=iteration
                    )

                # Execute Phase
                phase_output_str = ""
                if phase == "planning":
                    phase_output_str = self._phase_planning(prompt, run_dir, rag_context)
                elif phase == "design":
                    phase_output_str = self._phase_design(run_dir, prompt, rag_context)
                elif phase == "backend":
                    phase_output_str = self._phase_backend(run_dir, prompt, rag_context)
                elif phase == "frontend":
                    phase_output_str = self._phase_frontend(run_dir, prompt, rag_context)
                elif phase == "tests":
                    phase_output_str = self._phase_tests(run_dir, prompt, rag_context)
                elif phase == "deployment":
                    phase_output_str = self._phase_deployment(run_dir, prompt, rag_context)

                self._update_status(run_dir, job_id, phase, "completed", f"{phase} completed successfully.")

                # --- RAG Index Phase Output ---
                if (supabase_project_id or supabase_session_id) and phase_output_str:
                    index_phase_output(
                        project_id=supabase_project_id,
                        session_id=supabase_session_id,
                        phase_name=phase,
                        content=phase_output_str,
                        original_prompt=prompt,
                        iteration=iteration
                    )

                # --- Supabase: log successful phase ---
                if supabase_session_id:
                    supa.save_execution_log(
                        session_id=supabase_session_id,
                        phase=phase,
                        success=True,
                        iteration=iteration,
                    )
                    supa.update_session(
                        session_id=supabase_session_id,
                        current_phase=phase,
                        iteration_count=iteration,
                    )

                # Refresh status for next loop iteration
                status = self._load_status(run_dir)
                iteration += 1

            except Exception as e:
                err_msg = f"{phase} failed: {str(e)}"
                traceback.print_exc()
                self._update_status(run_dir, job_id, phase, "failed", err_msg)
                self._log_event(run_dir, "error", traceback.format_exc())

                # --- Supabase: log failed phase ---
                if supabase_session_id:
                    supa.save_execution_log(
                        session_id=supabase_session_id,
                        phase=phase,
                        success=False,
                        error_output=err_msg,
                        iteration=iteration,
                    )
                    supa.update_session(
                        session_id=supabase_session_id,
                        status="failed",
                        current_phase=phase,
                    )

                return {"status": "failed", "error": err_msg}

        self._update_status(run_dir, job_id, "complete", "completed", "All phases executed successfully.")

        # --- Supabase: build done — save assistant message + RAG summary ---
        summary_text = (
            f"Build completed successfully for prompt: '{prompt}'. "
            f"All 6 SDLC phases executed: planning, design, backend, frontend, tests, deployment."
        )
        if supabase_session_id:
            supa.save_message(
                session_id=supabase_session_id,
                role="assistant",
                content=summary_text,
            )
            supa.update_session(
                session_id=supabase_session_id,
                status="completed",
                current_phase="deployment",
                iteration_count=iteration,
            )
            if supabase_project_id:
                supa.save_rag_summary(
                    project_id=supabase_project_id,
                    session_id=supabase_session_id,
                    prompt=prompt,
                    summary=summary_text,
                    status="completed",
                    iteration=iteration,
                )

        return {"status": "completed", "summary": summary_text}

    # ------------------------------------------------------------------
    # PHASES
    # ------------------------------------------------------------------

    def _phase_planning(self, prompt: str, run_dir: str, rag_context: str = "") -> str:
        """Phase 1: Requirements"""
        sys_prompt = (
            "You are an expert Software Architect.\n"
            "You are continuing a deterministic SDLC workflow.\n"
        )
        if rag_context:
            sys_prompt += f"\nRelevant Historical Context:\n{rag_context}\n\nFollow prior architectural decisions unless explicitly overridden.\n"
            
        sys_prompt += (
            "\nAnalyze the following user request and produce a detailed requirement analysis. "
            "You are an expert Software Architect. "
            "Analyze the following user request and produce a detailed requirement analysis. "
            "Return a JSON object with: "
            "'functional_requirements' (list provided as markdown strings), "
            "'non_functional_requirements' (list), "
            "'user_roles' (list), "
            "'tech_stack' (detailed dict), "
            "'assumptions' (list). "
            "Also provide a 'markdown_content' field which contains the full nicely formatted content for a requirements.md file."
        )
        
        # User defined: Planning -> OpenAI / Gemini
        res = self.router.generate_text(f"{sys_prompt}\n\nRequest: {prompt}", preference=["openai", "azure_openai", "gemini"])
        data = self._extract_json(res.get("output", ""))
        
        # Save JSON data
        self._save_json(run_dir, "planning/planning.json", data)
        
        # Save Markdown Artifact
        md_content = data.get("markdown_content")
        if not md_content:
            md_content = f"# Requirements\n\n## Functional\n{json.dumps(data.get('functional_requirements', []), indent=2)}\n"
            
        self._save_file(run_dir, "planning/requirements.md", md_content)
        self._log_usage(run_dir, "planning", res)
        return json.dumps(data)

    def _phase_design(self, run_dir: str, prompt: str, rag_context: str = "") -> str:
        """Phase 2: System Design"""
        # Load planning data
        planning = self._load_json(run_dir, "planning/planning.json")
        if not planning:
            raise ValueError("Planning data missing. Cannot proceed to design.")

        sys_prompt = (
            "You are a System Designer.\n"
            "You are continuing a deterministic SDLC workflow.\n"
        )
        if rag_context:
            sys_prompt += f"\nRelevant Historical Context:\n{rag_context}\n\nFollow prior decisions from the relevant context unless explicitly overridden.\n"
            
        sys_prompt += (
            "Based on the requirements, generate a System Design. "
            "Return a JSON object with keys: "
            "'architecture_description', 'api_endpoints' (list of {method, path, desc}), "
            "'database_schema' (text description), "
            "'component_flow' (text), "
            "and 'markdown_content' (full content for architecture.md)."
        )
        
        # User defined: Design -> Gemini
        res = self.router.generate_text(f"{sys_prompt}\n\nRequirements: {json.dumps(planning)}", preference=["gemini", "openai"])
        data = self._extract_json(res.get("output", ""))
        self._save_json(run_dir, "design/design.json", data)
        self._save_file(run_dir, "design/architecture.md", data.get("markdown_content", "# Architecture\n"))
        self._log_usage(run_dir, "design", res)

        # Generate Flowchart (Mermaid) separately
        mermaid_prompt = (
            "Create a Mermaid.js flowchart (graph TD) representing the system architecture and data flow for the following application. "
            "Return ONLY the mermaid code block."
        )
        res_mmd = self.router.generate_text(f"{mermaid_prompt}\n\nContext: {json.dumps(data)}", preference=["gemini", "openai"])
        mmd_content = self._clean_mermaid(res_mmd.get("output", ""))
        self._save_file(run_dir, "design/flowchart.mmd", mmd_content)
        return json.dumps(data) + "\nMermaid: " + mmd_content

    def _phase_backend(self, run_dir: str, original_prompt: str, rag_context: str = "") -> str:
        """Phase 3: Backend Generation"""
        design = self._load_json(run_dir, "design/design.json")
        
        prompt = (
            "You are continuing a deterministic SDLC workflow.\n"
        )
        if rag_context:
            prompt += f"\nRelevant Historical Context:\n{rag_context}\n\nFollow prior architectural decisions unless explicitly overridden.\n"
            
        prompt += (
            "\nGenerate a production-ready FastAPI (Python) backend using the provided design. "
            "Strictly follow the project structure. "
            "Return a JSON mapping of filenames to their full string content. "
            "Include: 'main.py' (FastAPI entrypoint), 'models.py' (Pydantic/SQLAlchemy), 'database.py', 'requirements.txt'. "
            "Ensure 'main.py' uses absolute imports if needed, or simple local imports. "
            "Ensure uvicorn start string is standard. "
            f"Design Context: {json.dumps(design)}"
        )
        
        # User defined: Code -> OpenAI / Mistral
        res = self.router.generate_code(prompt, preference=["openai", "mistral", "groq"])
        files = res.get("files", {})
        if not files:
            raise ValueError("No backend files generated.")
            
        self._write_files(run_dir, files, prefix="backend")
        self._log_usage(run_dir, "backend", res)
        return json.dumps(list(files.keys()))

    def _phase_frontend(self, run_dir: str, original_prompt: str, rag_context: str = "") -> str:
        """Phase 4: Frontend Generation"""
        design = self._load_json(run_dir, "design/design.json")
        
        prompt = (
            "You are continuing a deterministic SDLC workflow.\n"
        )
        if rag_context:
            prompt += f"\nRelevant Historical Context:\n{rag_context}\n\nFollow prior decisions from the context unless explicitly overridden.\n"
            
        prompt += (
            "\nGenerate a complete React + Vite + Tailwind frontend application. "
            "Return a JSON mapping of filenames to content. "
            "Files REQUIRED: 'package.json', 'vite.config.js', 'index.html', "
            "'src/main.jsx', 'src/App.jsx', 'src/index.css', 'src/components/Layout.jsx'. "
            "Ensure the App components link to the backend at http://localhost:8000. "
            "Do not truncate codes. "
            f"Design Context: {json.dumps(design)}"
        )
        
        # User defined: Frontend -> OpenAI
        res = self.router.generate_code(prompt, preference=["openai", "azure_openai", "gemini"])
        files = res.get("files", {})
        if not files:
            raise ValueError("No frontend files generated.")
            
        self._write_files(run_dir, files, prefix="frontend")
        self._log_usage(run_dir, "frontend", res)
        return json.dumps(list(files.keys()))

    def _phase_tests(self, run_dir: str, original_prompt: str, rag_context: str = "") -> str:
        """Phase 5: Test Generation & Execution"""
        design = self._load_json(run_dir, "design/design.json")
        
        # 1. Generate Tests
        prompt = (
            "You are continuing a deterministic SDLC workflow.\n"
        )
        if rag_context:
            prompt += f"\nRelevant Historical Context:\n{rag_context}\n\nFollow prior decisions from the context unless explicitly overridden.\n"
            
        prompt += (
            "\nGenerate a 'test_main.py' using pytest and TestClient for the FastAPI backend. "
            "Cover the endpoints defined in the design."
        )
        # User defined: Tests -> Groq / OpenAI
        res = self.router.generate_code(prompt, preference=["groq", "openai"])
        files = res.get("files", {})
        self._write_files(run_dir, files, prefix="backend/tests") 
        self._log_usage(run_dir, "tests_gen", res)
        
        # 2. Execute Tests
        backend_dir = os.path.join(run_dir, "backend")
        
        report = {"passed": 0, "failed": 0, "log": ""}
        try:
            cmd = ["pytest", "tests", "--disable-warnings", "-v"]
            
            proc = subprocess.run(
                cmd, 
                cwd=backend_dir, 
                capture_output=True, 
                text=True,
                timeout=45
            )
            
            report["log"] = proc.stdout + "\n" + proc.stderr
            report["return_code"] = proc.returncode
            report["passed"] = int(report["log"].count("PASSED"))
            report["failed"] = int(report["log"].count("FAILED"))
            
            if proc.returncode != 0 and report["failed"] == 0:
                 report["error"] = "Test execution failed (syntax or env error)."
        
        except Exception as e:
            report["error"] = str(e)
            
        self._save_json(run_dir, "tests/test_report.json", report)
        # Also save to run root for easier access if needed, but primary is in tests/
        self._save_json(run_dir, "run_report.json", report)
        return json.dumps(report)

    def _phase_deployment(self, run_dir: str, original_prompt: str, rag_context: str = "") -> str:
        """Phase 6: Deployment Artifacts"""
        prompt = "You are continuing a deterministic SDLC workflow.\n"
        if rag_context:
            prompt += f"\nRelevant Historical Context:\n{rag_context}\n\nFollow prior context strictly.\n"
            
        prompt += "\nGenerate 'Dockerfile' for a FastAPI app and a 'docker-compose.yml' that serves backend and a generic frontend service."
        res = self.router.generate_code(prompt, preference=["openai", "mistral"])
        files = res.get("files", {})
        self._write_files(run_dir, files, prefix="deployment")
        self._save_file(run_dir, "deployment/deployment.md", "# Deployment Guide\n\nRun `docker-compose up --build`.")
        self._log_usage(run_dir, "deployment", res)
        return json.dumps(list(files.keys()))

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

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
        # A simple append-only log specific to the build
        pass # covered by status.json execution_log mostly

    def _log_usage(self, run_dir: str, phase: str, res: Dict, supabase_session_id: Optional[str] = None):
        path = os.path.join(run_dir, "providers_used.json")
        history = []
        if os.path.exists(path):
            try:
                with open(path, "r") as f: history = json.load(f).get("history", [])
            except: pass

        history.append({
            "phase": phase,
            "provider": res.get("provider", "unknown"),
            "model": res.get("model", "unknown"),
            "timestamp": datetime.now().isoformat()
        })

        with open(path, "w") as f:
            json.dump({"history": history}, f, indent=2)

        # --- Supabase: provider usage ---
        if supabase_session_id:
            supa.save_provider_usage(
                session_id=supabase_session_id,
                provider_name=res.get("provider", "unknown"),
                model_name=res.get("model"),
                phase=phase,
                success=True,
            )

    def _save_json(self, run_dir: str, filename: str, data: Any):
        path = os.path.join(run_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_json(self, run_dir: str, filename: str) -> Optional[Dict]:
        path = os.path.join(run_dir, filename)
        if os.path.exists(path):
            with open(path, "r") as f: return json.load(f)
        return None

    def _save_file(self, run_dir: str, filename: str, content: str):
        path = os.path.join(run_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _write_files(self, run_dir: str, files: Dict[str, str], prefix: str = ""):
        for rel_path, content in files.items():
            clean_path = rel_path
            if prefix and not clean_path.startswith(prefix):
                clean_path = os.path.join(prefix, clean_path)
            full_path = os.path.join(run_dir, clean_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        try: return json.loads(text)
        except: pass
        import re
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except: pass
        return {"error": "Failed to parse JSON", "raw": text}

    def _clean_mermaid(self, text: str) -> str:
        """Cleans up mermaid code from markdown blocks and ensures valid syntax."""
        import re
        content = text
        
        # 1. Remove markdown code blocks
        content = re.sub(r"```mermaid\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"```\s*", "", content)
        
        # 2. Trim whitespace
        content = content.strip()
        
        # 3. Fallback check
        if not any(content.startswith(k) for k in ["graph", "sequenceDiagram", "classDiagram", "erDiagram", "flowchart"]):
            # If AI didn't give a chart, provide a simple placeholder to avoid syntax errors
            return "graph TD;\n    A[Architecture] --> B[Generated];"
            
        return content

