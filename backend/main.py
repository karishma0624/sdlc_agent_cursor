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


@app.get("/providers")
def providers():
    return {
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY")),
        "MISTRAL_API_KEY": bool(os.getenv("MISTRAL_API_KEY")),
        "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
        "HUGGINGFACE_API_KEY": bool(os.getenv("HUGGINGFACE_API_KEY")),
        "V0_API_KEY": bool(os.getenv("V0_API_KEY")),
        "OLLAMA_BASE_URL": bool(os.getenv("OLLAMA_BASE_URL")),
    }


# -------------------------------------------------
# SDLC BUILD (STUB – STABLE)
# -------------------------------------------------
_BUILD_JOBS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


class BuildRequest(BaseModel):
    prompt: str


@app.post("/sdlc/build")
def sdlc_build(req: BuildRequest):
    job_id = uuid.uuid4().hex

    with _LOCK:
        _BUILD_JOBS[job_id] = {
            "job_id": job_id,
            "status": "completed",
            "prompt": req.prompt,
            "run_dir": f"runs/{job_id}",
            "timestamp": datetime.utcnow().isoformat(),
        }

    return {"job_id": job_id, "status": "completed"}


@app.get("/sdlc/status")
def sdlc_status(job_id: Optional[str] = None):
    with _LOCK:
        if job_id:
            return _BUILD_JOBS.get(job_id, {"error": "not found"})
        return {"jobs": list(_BUILD_JOBS.values())}


# -------------------------------------------------
# ROOT
# -------------------------------------------------
@app.get("/")
def root():
    return {"message": "Autonomous SDLC Builder API running"}
