# backend/main.py

import os
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
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# AUTH
# -------------------------------------------------
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
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
# -------------------------------------------------
# SDLC BUILD (REAL DETERMINISTIC)
# -------------------------------------------------
from .simple_builder import SimpleBuilder

JOBS_FILE = "jobs.json"
_LOCK = threading.Lock()
builder = SimpleBuilder(runs_dir="runs")

def load_jobs() -> Dict[str, Dict[str, Any]]:
    if os.path.exists(JOBS_FILE):
        try:
            with open(JOBS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_jobs(jobs: Dict[str, Dict[str, Any]]):
    try:
        with open(JOBS_FILE, "w") as f:
            json.dump(jobs, f, indent=2)
    except:
        pass

_BUILD_JOBS: Dict[str, Dict[str, Any]] = load_jobs()

class BuildRequest(BaseModel):
    prompt: str

@app.post("/sdlc/build")
def sdlc_build(req: BuildRequest):
    job_id = uuid.uuid4().hex
    started_at = datetime.utcnow().isoformat()
    
    # 1. Initialize Run (Create Dir & status.json) synchronously
    run_dir = builder.init_run(req.prompt)
    
    # 2. Update Status Storage (Persist immediately)
    with _LOCK:
        _BUILD_JOBS[job_id] = {
            "job_id": job_id,
            "status": "running",
            "prompt": req.prompt,
            "started_at": started_at,
            "finished_at": None,
            "run_dir": run_dir,
            "summary": None,
            "error": None
        }
        save_jobs(_BUILD_JOBS)

    # 3. Start Background Build
    def _run_build(job_id: str, run_dir: str, prompt: str):
        try:
            # Execute actual build in the created dir
            result = builder.run_build(run_dir, prompt)
            
            with _LOCK:
                if job_id in _BUILD_JOBS:
                    job = _BUILD_JOBS[job_id]
                    job["status"] = "completed"
                    job["finished_at"] = datetime.utcnow().isoformat()
                    job["summary"] = result.get("summary")
                    save_jobs(_BUILD_JOBS)
                
        except Exception as e:
            with _LOCK:
                if job_id in _BUILD_JOBS:
                    job = _BUILD_JOBS[job_id]
                    job["status"] = "failed"
                    job["finished_at"] = datetime.utcnow().isoformat()
                    job["error"] = str(e)
                    save_jobs(_BUILD_JOBS)
                
                # Also update status.json to failed
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

    # 4. Return IMMEDIATE Response with Run Dir
    return {
        "job_id": job_id,
        "status": "running",
        "started_at": started_at,
        "run_dir": run_dir
    }

@app.get("/sdlc/status")
def sdlc_status(job_id: Optional[str] = None):
    # Reload jobs to ensure we have latest state if modified by other workers (though Uvicorn reload restarts global state)
    # Ideally for multi-worker we'd reload from file, but with single worker in-memory + write-through is fine.
    # We will reload just in case.
    # Note: Frequent reads might be slow but safety first.
    current_jobs = load_jobs()
    
    if job_id:
        job = current_jobs.get(job_id)
        if not job:
            return {
                "job_id": job_id,
                "status": "not_found",
                "error": "Job not found",
                "started_at": None,
                "finished_at": None,
                "run_dir": None
            }
        return job
    # List all
    return {"jobs": list(current_jobs.values())}

@app.get("/providers")
def get_providers():
    builder.router.refresh()
    # CONTRACT: Nested "providers" object with lowercase keys
    p = builder.router.providers
    return {
        "providers": {
            "openai": p.get("openai", False),
            "gemini": p.get("gemini", False),
            "mistral": p.get("mistral", False),
            "groq": p.get("groq", False),
            "hf": p.get("hf", False),
            "ollama": p.get("ollama", False),
            "v0": p.get("v0", False),
            "perplexity": p.get("perplexity", False),
            "lovable": p.get("lovable", False),
            "stitch": p.get("stitch", False)
        }
    }

@app.get("/sdlc/report")
def sdlc_report(job_id: Optional[str] = None):
    target_dir = None
    current_jobs = load_jobs()
    if job_id:
        job = current_jobs.get(job_id)
        if job:
            target_dir = job.get("run_dir")
    
    if not target_dir:
        return {"error": "job not found or no run_dir"}
        
    report_path = os.path.join(target_dir, "run_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            return json.load(f)
    return {"error": "report missing"}


# -------------------------------------------------
# ROOT
# -------------------------------------------------
@app.get("/")
def root():
    return {"message": "Autonomous SDLC Builder API running"}
