import os
import json
from services.adapters import call_gemini, call_v0
from .templates import (
    _TEMPLATE_BACKEND_REQUIREMENTS,
    _TEMPLATE_BACKEND_MAIN,
    _TEMPLATE_BACKEND_DATABASE
)

def execute_backend_phase(planning_content: str, run_dir: str) -> tuple[bool, str]:
    """
    PHASE 5: BACKEND GENERATION (Gemini)
    
    Logic:
    1. Try Gemini.
    2. If fails -> Fallback to basic FastAPI scaffold.
    
    Returns: (True, "Success Message") or (False, "Error Message")
    """
    
    backend_root = os.path.join(run_dir, "backend")
    os.makedirs(backend_root, exist_ok=True)
    
    # --- PROMPT CONSTRUCTION ---
    prompt = (
        "You are an expert backend developer. Generate a COMPLETE, PRODUCTION-READY FastAPI (Python) backend application.\n\n"
        f"IMPLEMENTATION PLAN:\n{planning_content[:3000]}\n\n"
        "STRICT REQUIREMENTS:\n"
        "1. **Output Format**: ONLY valid JSON with this structure:\n"
        "   {\"files\": {\"path/to/file.ext\": \"file content as string\", ...}}\n"
        "2. **Tech Stack**: FastAPI, SQLAlchemy (SQLite), Pydantic\n"
        "3. **Required Files** (MUST include ALL of these):\n"
        "   - requirements.txt (fastapi, uvicorn, sqlalchemy, etc.)\n"
        "   - app/main.py (FastAPI app, CORS, routes)\n"
        "   - app/models.py (SQLAlchemy models based on plan)\n"
        "   - app/schemas.py (Pydantic models)\n"
        "   - app/database.py (SQLAlchemy setup)\n"
        "   - app/crud.py (CRUD operations)\n"
        "4. **Code Quality**:\n"
        "   - Type hints everywhere\n"
        "   - Docstrings for all endpoints\n"
        "   - Correct relationships in models\n"
        "   - No placeholders\n"
        "5. **Output**: ONLY the JSON object. NO markdown code fences. NO explanations.\n\n"
        "Generate the complete backend NOW:"
    )
    
    # Strict Order: Gemini only (v0 is frontend focused usually)
    # But we can use the same pattern
    providers = [
        ("Gemini", call_gemini)
    ]
    
    last_error = ""
    
    for provider_name, provider_func in providers:
        print(f"[{run_dir}] Trying {provider_name} for Backend...")
        try:
            files = provider_func(prompt)
            
            # Validate Output
            valid, msg = _validate_backend_files(files)
            if not valid:
                err_msg = f"{provider_name} output validation failed: {msg}"
                print(f"[{run_dir}] {err_msg}")
                last_error = err_msg
                continue
                
            print(f"[{run_dir}] {provider_name} Success! Writing files.")
            _write_files(backend_root, files)
            return True, f"Success using {provider_name}"
            
        except Exception as e:
            err_msg = f"{provider_name} Failed: {str(e)}"
            print(f"[{run_dir}] {err_msg}")
            last_error = err_msg
            continue
    
    print(f"[{run_dir}] JSON content generation failed. Generating fallback scaffold.")
    _generate_fallback(backend_root)
    return True, "API limits reached. Generated local fallback scaffold instead."


def _generate_fallback(root: str):
    """Writes a minimal working FastAPI app"""
    files = {
        "requirements.txt": _TEMPLATE_BACKEND_REQUIREMENTS,
        "app/main.py": _TEMPLATE_BACKEND_MAIN,
        "app/database.py": _TEMPLATE_BACKEND_DATABASE,
        "app/models.py": "# Add your SQLAlchemy models here\n"
    }
    _write_files(root, files)


def _validate_backend_files(files: dict) -> tuple[bool, str]:
    if not files or not isinstance(files, dict):
        return False, "Empty or invalid format"
        
    required = ["requirements.txt", "app/main.py"]
    missing = []
    
    # Normalize keys
    normalized_keys = []
    for k in files.keys():
        clean = k.replace("backend/", "").replace("backend\\", "")
        normalized_keys.append(clean)
        
    for r in required:
        if r not in normalized_keys:
            missing.append(r)
            
    if len(missing) > 0:
         return False, f"Missing required files: {missing}"
            
    return True, "Valid"

def _write_files(root_dir: str, files: dict):
    for filename, content in files.items():
        # Clean path
        clean_name = filename
        if clean_name.startswith("backend/") or clean_name.startswith("backend\\"):
            clean_name = clean_name[8:]
        
        # Security
        if ".." in clean_name:
            continue
            
        clean_name = clean_name.lstrip("/\\")
        full_path = os.path.join(root_dir, clean_name)
        
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            if isinstance(content, str):
                f.write(content)
            else:
                f.write(str(content))
