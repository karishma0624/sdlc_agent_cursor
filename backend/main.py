# backend/main.py

import os
import sys
# Add parent dir to allow 'backend.research' imports even if running from inside backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Add current dir (backend) to allow 'services' imports from phases
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import threading
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from dotenv import load_dotenv

# -------------------------------------------------
# ENV
# -------------------------------------------------
load_dotenv()
if not os.getenv("CheckEnv"):
    # Fallback to parent .env if valid keys are missing or file prevents it
    # But simpler: explicit path check
    if not os.path.exists(".env") and os.path.exists("../.env"):
        load_dotenv("../.env")

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# -------------------------------------------------
# APP INIT
# -------------------------------------------------
app = FastAPI(
    title="Autonomous SDLC Builder API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# AUTH
# -------------------------------------------------
# Use pbkdf2_sha256 for maximum compatibility on Windows (no C-extensions required like bcrypt/argon2)
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# -------------------------------------------------
# MODELS (IN-MEMORY FOR STABILITY)
# -------------------------------------------------
class User(BaseModel):
    username: str
    password_hash: str
    is_active: bool = True


USERS: Dict[str, User] = {}


class Token(BaseModel):
    access_token: str
    token_type: str


class UserCreate(BaseModel):
    username: str
    password: str


# -------------------------------------------------
# AUTH HELPERS
# -------------------------------------------------
def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def authenticate_user(username: str, password: str) -> Optional[User]:
    user = USERS.get(username)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = USERS.get(username)
    if not user or not user.is_active:
        raise credentials_exception
    return user


# -------------------------------------------------
# ROUTES – AUTH
# -------------------------------------------------
@app.post("/register")
def register(user: UserCreate):
    if user.username in USERS:
        raise HTTPException(status_code=400, detail="User exists")

    USERS[user.username] = User(
        username=user.username,
        password_hash=get_password_hash(user.password),
    )
    return {"message": "User registered"}

# --- DEFAULT ADMIN USER ---
if "admin" not in USERS:
    USERS["admin"] = User(
        username="admin", 
        password_hash=get_password_hash("admin")
    )
# --------------------------


@app.post("/token", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid login")

    token = create_access_token(
        {"sub": user.username},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": token, "token_type": "bearer"}


@app.get("/users/me")
def me(user: User = Depends(get_current_user)):
    return {"username": user.username, "active": user.is_active}


# -------------------------------------------------
# HEALTH & PROVIDERS
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}




# -------------------------------------------------
# SDLC BUILD (REAL DETERMINISTIC)
# -------------------------------------------------
try:
    from backend.simple_builder import SDLCBuilder
    from backend.db import db
except ImportError:
    from simple_builder import SDLCBuilder
    from db import db

# Consolidate RUNS directory to be consistently inside 'PROJECT_ROOT/runs'
# regardless of where the script is run from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # .../backend
ROOT_DIR = os.path.dirname(BASE_DIR) # .../
RUNS_DIR_ABS = os.path.join(ROOT_DIR, "runs")

JOBS_FILE = "jobs.json"
_LOCK = threading.Lock()
# Initialize SDLCBuilder
builder = SDLCBuilder(runs_dir=RUNS_DIR_ABS)

# Initialize Research Wrapper (Metrics & Protection)
try:
    from backend.research.wrapper import ResearchWrapper
    # Wrap standard builder with research features
    builder = ResearchWrapper(builder)
    print(f"[Research] Wrapper Enabled | Logging to: {RUNS_DIR_ABS}")
except ImportError as e:
    # This should be rare now with sys.path fix
    print(f"[Research] ⚠️ Wrapper Failed to Load: {e}")
    print("[Research] Running in Standard Mode (No Metrics)")
except Exception as e:
    print(f"[Research] ⚠️ Wrapper Initialization Error: {e}")

def rebuild_jobs_from_disk() -> Dict[str, Dict[str, Any]]:
    """SCANS runs/ directory and rebuilds memory state from status.json files."""
    jobs = {}
    if not os.path.exists(RUNS_DIR_ABS):
        return jobs
        
    for dirname in os.listdir(RUNS_DIR_ABS):
        run_dir = os.path.join(RUNS_DIR_ABS, dirname)
        status_path = os.path.join(run_dir, "status.json")
        if os.path.isdir(run_dir) and os.path.exists(status_path):
            try:
                with open(status_path, "r") as f:
                    job_data = json.load(f)
                    # Backfill missing keys for old jobs
                    if "job_id" not in job_data:
                         job_data["job_id"] = dirname # Fallback
                    
                    # Ensure path is absolute/correct
                    job_data["run_dir"] = run_dir

                    # SELF-HEALING: If job was 'running' when server died, mark it failed so UI unlocks.
                    if job_data.get("status") == "running":
                        job_data["status"] = "failed"
                        job_data["error"] = "Process interrupted (Server Restart)"
                        # Optionally write back to disk to sync state
                        try:
                            with open(status_path, "w") as fw:
                                json.dump(job_data, fw, indent=2)
                        except: pass
                    
                    jobs[job_data["job_id"]] = job_data
            except Exception as e:
                print(f"Skipping corrupt job {dirname}: {e}")
    return jobs

# ... (skipped unchanged lines)

# -------------------------------------------------
# OPEN OUTPUT FOLDER
# -------------------------------------------------
class OpenFolderRequest(BaseModel):
    path: str

@app.post("/open-folder")
def open_folder(req: OpenFolderRequest):
    """Opens the specified folder in Windows Explorer"""
    import subprocess
    import platform
    
    try:
        # 1. Try exact match
        target_path = req.path
        
        # 2. If relative, try relative to RUNS directory
        if not os.path.isabs(target_path):
             potential = os.path.join(RUNS_DIR_ABS, os.path.basename(target_path))
             if os.path.exists(potential):
                 target_path = potential
             else:
                 # Try relative to CWD
                 potential = os.path.abspath(target_path)
                 if os.path.exists(potential):
                     target_path = potential

        if not os.path.exists(target_path):
             # 3. Last ditch: check if it's a job ID
             potential = os.path.join(RUNS_DIR_ABS, req.path)
             if os.path.exists(potential):
                 target_path = potential
             else:
                raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")
        
        print(f"Opening folder: {target_path}")

        # Open folder based on OS
        if platform.system() == "Windows":
            os.startfile(target_path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", target_path])
        else:  # Linux
            subprocess.Popen(["xdg-open", target_path])
            
        return {"success": True, "path": target_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def load_jobs() -> Dict[str, Dict[str, Any]]:
    # Always rebuild from disk to be authoritative
    return rebuild_jobs_from_disk()

def save_jobs(jobs: Dict[str, Dict[str, Any]]):
    # We no longer rely on a monolithic jobs.json. 
    # Each job updates its own status.json.
    # This function is kept for compatibility but does nothing or could update cache.
    pass

# Initial load
_BUILD_JOBS: Dict[str, Dict[str, Any]] = rebuild_jobs_from_disk()

class NewSessionRequest(BaseModel):
    mode: Optional[str] = "auto" # 'web_app', 'content', 'auto'

@app.post("/sessions/new")
def new_session(req: NewSessionRequest = NewSessionRequest()):
    job_id = uuid.uuid4().hex
    started_at = datetime.utcnow().isoformat()
    mode = req.mode
    
    # Create an 'idle' job entry. We don't create a run_dir yet until the first build.
    # OR better: create a placeholder dir so chat.json can exist immediately.
    # Let's create the dir to strictly follow "One Job = One Session" rule.
    run_dir = builder.init_run("New Session", job_id=job_id, mode=mode)
    
    with _LOCK:
        _BUILD_JOBS[job_id] = {
            "job_id": job_id,
            "status": "idle",
            "mode": mode,
            "prompt": "New Session",
            "started_at": started_at,
            "finished_at": None,
            "run_dir": run_dir,
            "phases": {
                "planning": "waiting",
                "design": "waiting",
                "backend": "waiting",
                "frontend": "waiting",
                "tests": "waiting",
                "deployment": "waiting"
            },
            "summary": None,
            "error": None
        }
        # CRITICAL FIX: Persist initial state to disk immediately so restarts don't lose the job
        try:
            builder._save_json(run_dir, "status.json", _BUILD_JOBS[job_id])
            # Initialize empty chat log
            builder._save_json(run_dir, "chat.json", [])
        except Exception as e:
            print(f"Failed to persist new session: {e}")
            
        save_jobs(_BUILD_JOBS)
        
    return {"job_id": job_id, "created_at": started_at, "status": "idle", "mode": mode}

class BuildRequest(BaseModel):
    prompt: Optional[str] = None
    job_id: str  # STRICT REQUIREMENT

class ChatRequest(BaseModel):
    job_id: str
    message: str
    attachments: Optional[list] = [] # List of {name, type, content(base64)}

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    """
    Handles follow-up chat messages.
    1. Appends to chat.json
    2. Triggers builder analysis (continuation)
    """
    global _BUILD_JOBS
    
    # Reload logic (Self-Healing)
    if req.job_id not in _BUILD_JOBS:
        _BUILD_JOBS = rebuild_jobs_from_disk()
    
    job = _BUILD_JOBS.get(req.job_id)
    if not job:
         raise HTTPException(status_code=404, detail="Job not found")
         
    run_dir = job["run_dir"]
    
    # 1. Persist User Message
    chat_path = os.path.join(run_dir, "chat.json")
    chat_log = []
    if os.path.exists(chat_path):
        try: 
            with open(chat_path, "r") as f: chat_log = json.load(f)
        except: pass
        
    chat_log.append({
        "role": "user", 
        "content": req.message, 
        "attachments": req.attachments,
        "timestamp": datetime.utcnow().isoformat()
    })
    with open(chat_path, "w") as f:
        json.dump(chat_log, f, indent=2)
        
    # 2. Trigger Builder (Continuation)
    # We update status to 'running' and let the builder decide if it needs to code or just reply.
    # For now, we assume every message is a 'Prompt' that might need code changes.
    
    job["status"] = "running"
    job["prompt"] = req.message # Update active prompt context
    job["status_message"] = "Analyzing request..."
    
    # Sync to disk
    p = os.path.join(run_dir, "status.json")
    if os.path.exists(p):
        with open(p, "r") as f: d = json.load(f)
        d.update({"status": "running", "status_message": "Analyzing request..."})
        with open(p, "w") as f: json.dump(d, f)

    def _run_continuation(job_id, run_dir, msg):
        try:
            builder.run_build(run_dir, msg)
        except Exception as e:
            # Error handling handled inside run_build usually, but safety net
            pass

    t = threading.Thread(target=_run_continuation, args=(req.job_id, run_dir, req.message), daemon=True)
    t.start()
    
    return {"status": "processing", "job_id": req.job_id}

@app.post("/sdlc/build")
def sdlc_build(req: BuildRequest):
    # ... (existing logic)
    job_id = req.job_id
    
    with _LOCK:
        current_job = _BUILD_JOBS.get(job_id)
        if not current_job:
            # If not in memory, try to load from disk or valid run_dir if we had logic for that.
            # For now, strict: must exist. But we just created it in /sessions/new so it should exist.
            # Handle restart case where _BUILD_JOBS is empty: we rely on load_jobs() at startup.
            # If really missing, fail or recreate. Recreating is safer for user but weird.
            # Let's recreate if missing to be robust against restarts.
             _BUILD_JOBS[job_id] = {
                "job_id": job_id,
                "status": "idle",
                "prompt": req.prompt or "Restored",
                "started_at": datetime.utcnow().isoformat(),
                "finished_at": None,
                "run_dir": builder.init_run(req.prompt or "Restored", job_id=job_id),
                "summary": None,
                "error": None
            }
             save_jobs(_BUILD_JOBS)
             current_job = _BUILD_JOBS[job_id]

        run_dir = current_job.get("run_dir")
        
        # PERSIST CHAT
        if req.prompt:
            chat_path = os.path.join(run_dir, "chat.json")
            chat_log = []
            if os.path.exists(chat_path):
                try: 
                    with open(chat_path, "r") as f: chat_log = json.load(f)
                except: pass
            
            chat_log.append({"role": "user", "content": req.prompt, "timestamp": datetime.utcnow().isoformat()})
            # We don't have the "agent" message yet, that comes from execution events.
            # frontend poll will pick up status updates.
            
            with open(chat_path, "w") as f:
                json.dump(chat_log, f, indent=2)

        # Update Job Status to Running
        current_job["status"] = "running"
        current_job["finished_at"] = None
        current_job["error"] = None
        if req.prompt:
            current_job["prompt"] = req.prompt # Update context
        save_jobs(_BUILD_JOBS)

    # 3. Start Background Build
    def _run_build(job_id: str, run_dir: str, prompt: str):
        try:
            # Execute actual build
            prompt_to_use = prompt or "Continue build"
            result = builder.run_build(run_dir, prompt_to_use)
            
            with _LOCK:
                if job_id in _BUILD_JOBS:
                    job = _BUILD_JOBS[job_id]
                    # Update status based on result
                    final_status = result.get("status", "completed")
                    job["status"] = final_status
                    job["finished_at"] = datetime.utcnow().isoformat()
                    job["summary"] = result.get("summary")
                    if final_status == "failed":
                        job["error"] = result.get("error")
                    save_jobs(_BUILD_JOBS)
                
        except Exception as e:
            with _LOCK:
                if job_id in _BUILD_JOBS:
                    job = _BUILD_JOBS[job_id]
                    job["status"] = "failed"
                    job["finished_at"] = datetime.utcnow().isoformat()
                    job["error"] = str(e)
                    save_jobs(_BUILD_JOBS)
                
                try:
                    p = os.path.join(run_dir, "status.json")
                    if os.path.exists(p):
                        with open(p, "r") as f: d = json.load(f)
                        d.update({"status": "failed", "error": str(e)})
                        with open(p, "w") as f: json.dump(d, f)
                except:
                    pass

    t = threading.Thread(target=_run_build, args=(job_id, run_dir, req.prompt), daemon=True)
    t.start()

    return {
        "job_id": job_id,
        "status": "running",
        "started_at": current_job["started_at"],
        "run_dir": run_dir
    }

# --- ALIAS FOR FRONTEND COMPATIBILITY ---
@app.post("/api/sdlc/build")
def api_sdlc_build(req: BuildRequest):
    return sdlc_build(req)

class PreviewRequest(BaseModel):
    job_id: str

@app.post("/sdlc/preview")
def preview_endpoint(req: PreviewRequest):
    """
    Triggers npm run dev for the given job.
    Returns the URL (e.g., http://localhost:5173).
    """
    result = builder.preview_frontend(req.job_id)
    if result.get("status") == "failed":
        raise HTTPException(status_code=500, detail=result.get("error"))
        
    return result

@app.get("/runs")
def list_runs():
    """Return historical runs for the sidebar."""
    current_jobs = load_jobs()
    # Sort by started_at desc, handle None by using a very old date
    
    def get_sort_key(job):
        s = job.get("started_at")
        if not s: 
            return "1970-01-01"
        return s

    sorted_jobs = sorted(current_jobs.values(), key=get_sort_key, reverse=True)
    return {"runs": sorted_jobs}

@app.get("/sdlc/status")
def sdlc_status(job_id: Optional[str] = None):
    global _BUILD_JOBS
    # 1. Check Memory
    job = _BUILD_JOBS.get(job_id)
    
    # 2. If missing, FORCE DISK RESCAN (Self-Healing)
    if not job:
        print(f"Job {job_id} missing from memory, scanning disk...")
        _BUILD_JOBS = rebuild_jobs_from_disk()
        job = _BUILD_JOBS.get(job_id)

    if not job:
            return {
            "job_id": job_id,
            "status": "not_found",
            "error": "Job not found on disk", 
            "run_dir": None
        }
        
    # 3. Always refresh from status.json to get latest BUILDER updates
    # The builder runs in a thread and writes to disk. Memory might be stale.
    if job.get("run_dir"):
        try:
            p = os.path.join(job["run_dir"], "status.json")
            if os.path.exists(p):
                with open(p, "r") as f:
                    live_status = json.load(f)
                    _BUILD_JOBS[job_id].update(live_status) # Sync memory
                    job = _BUILD_JOBS[job_id]
        except:
            pass

        try:
            # Load flowchart
            fc_path = os.path.join(job["run_dir"], "design", "flowchart.mmd")
            if not os.path.exists(fc_path):
                    # Fallback for old runs
                    fc_path = os.path.join(job["run_dir"], "flowchart.mmd")
            
            if os.path.exists(fc_path):
                with open(fc_path, "r", encoding="utf-8") as f:
                    job["flowchart"] = f.read()
        except:
            pass
            
        try:
            # Load providers history
            prov_path = os.path.join(job["run_dir"], "providers_used.json")
            if os.path.exists(prov_path):
                    with open(prov_path, "r") as f:
                        job["providers_history"] = json.load(f).get("history", [])
        except:
            pass

        try:
            # Load test report
            test_path = os.path.join(job["run_dir"], "tests", "test_report.json")
            if not os.path.exists(test_path):
                    # Fallback
                    test_path = os.path.join(job["run_dir"], "test_report.json")

            if os.path.exists(test_path):
                    with open(test_path, "r") as f:
                        job["test_report"] = json.load(f)
        except:
            pass
            
    # 4. LOAD CHAT HISTORY
    run_dir = job.get("run_dir")
    if run_dir:
        chat_path = os.path.join(run_dir, "chat.json")
        if os.path.exists(chat_path):
            try:
                with open(chat_path, "r") as f:
                    job["messages"] = json.load(f)
            except:
                job["messages"] = []
        else:
            job["messages"] = []

    return {
        "run_id": job.get("job_id"),
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "current_phase": job.get("current_phase", "planning"),
        "phases": job.get("phases", {}),
        "messages": job.get("messages", []),
        "flowchart": job.get("flowchart"),
        "providers_history": job.get("providers_history", []),
        "test_report": job.get("test_report"),
        "run_dir": job.get("run_dir")
    }
        





@app.get("/sdlc/report")
def get_run_report(run_dir: str = ""):
    """
    Returns structured artifacts for the run.
    """
    if not run_dir or not os.path.exists(run_dir):
        # Maybe job_id was passed?
        # Try to find run_dir from job_id logic if empty?
        # But frontend sends strict run_dir from status.
        return {"artifacts": {}, "commands": {}}
        
    artifacts = {}
    
    # Check for basic artifacts
    for f in ["requirements.md", "planning.md", "design/flowchart.mmd"]:
        fp = os.path.join(run_dir, f)
        if os.path.exists(fp):
            artifacts[f] = fp

    # Frontend
    fe_path = os.path.join(run_dir, "frontend")
    if os.path.exists(fe_path):
        artifacts["frontend"] = [f for f in os.listdir(fe_path) if not f.startswith(".")]

    # Backend 
    be_path = os.path.join(run_dir, "backend")
    if os.path.exists(be_path):
        artifacts["backend"] = [f for f in os.listdir(be_path) if not f.startswith(".")]

    return {
        "summary": "Run Artifacts",
        "artifacts": artifacts,
        "commands": {
            "frontend": ["npm install", "npm run dev"],
            "backend": ["uvicorn backend.main:app --reload"]
        }
    }

@app.delete("/runs/{job_id}")
def delete_run(job_id: str):
    with _LOCK:
        job = _BUILD_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        run_dir = job.get("run_dir")
        if run_dir and os.path.exists(run_dir):
            import shutil
            try:
                shutil.rmtree(run_dir)
            except Exception as e:
                print(f"Error deleting dir {run_dir}: {e}")
        
        del _BUILD_JOBS[job_id]
        save_jobs(_BUILD_JOBS)
        
    return {"status": "deleted", "job_id": job_id}


@app.get("/providers")
def get_providers():
    # builder.router is removed. Check Env directly.
    return {
        "providers": {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "gemini": bool(os.getenv("GEMINI_API_KEY")),
            "mistral": bool(os.getenv("MISTRAL_API_KEY")),
            "groq": bool(os.getenv("GROQ_API_KEY")),
            "v0": bool(os.getenv("V0_API_KEY") or os.getenv("V0_DEV_API_KEY")),
            "mermaid": True # Always available via Gemini/Adapter
        }
    }



# -------------------------------------------------
# ROOT
# -------------------------------------------------
@app.get("/")
def root():
    return {"message": "Autonomous SDLC Builder API running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

