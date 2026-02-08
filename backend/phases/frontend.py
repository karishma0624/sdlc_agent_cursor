import os
import json
from services.adapters import call_gemini, call_v0

def execute_frontend_phase(planning_content: str, run_dir: str) -> bool:
    """
    PHASE 4: FRONTEND GENERATION (Strict Gemini -> v0)
    
    Logic:
    1. Try Gemini.
    2. If fails validation/execution -> Try v0.
    3. If v0 fails -> FAIL PHASE.
    
    Returns: True if success, False if failed.
    """
    
    frontend_root = os.path.join(run_dir, "frontend")
    os.makedirs(frontend_root, exist_ok=True)
    
    # --- PROMPT CONSTRUCTION ---
    prompt = (
        "Generate a COMPLETE, WORKING React + Vite + Tailwind frontend based on this plan.\n\n"
        f"PLAN: {planning_content[:2000]}...\n\n"
        "REQUIREMENTS:\n"
        "1. Output valid JSON: {\"files\": {\"filename\": \"content\"}}\n"
        "2. Tech Stack: React, Vite, TailwindCSS, Lucide-React\n"
        "3. Include 'package.json', 'vite.config.js', 'src/main.jsx', 'src/App.jsx', 'index.html', 'src/index.css'\n"
        "4. DO NOT use placeholders. Write full component code.\n"
        "5. NO Markdown. JUST JSON.\n"
    )
    
    # --- ATTEMPT 1: GEMINI ---
    print(f"[{run_dir}] Trying Gemini for Frontend...")
    files = {}
    try:
        files = call_gemini(prompt)
        
        # Validate Gemini Output
        if not _validate_frontend_files(files):
            print(f"[{run_dir}] Gemini output validation failed. Switching to v0.")
            raise ValueError("Invalid Gemini Output")
            
        print(f"[{run_dir}] Gemini Success. Writing files.")
        _write_files(frontend_root, files)
        return True
        
    except Exception as e:
        print(f"[{run_dir}] Gemini Failed: {e}")
        
    # --- ATTEMPT 2: v0 ---
    print(f"[{run_dir}] Trying v0 for Frontend...")
    try:
        files = call_v0(prompt)
        
        if not _validate_frontend_files(files):
            print(f"[{run_dir}] v0 output validation failed.")
            raise ValueError("Invalid v0 Output")
            
        print(f"[{run_dir}] v0 Success. Writing files.")
        _write_files(frontend_root, files)
        return True
        
    except Exception as e:
        print(f"[{run_dir}] v0 Failed: {e}")
        
    # --- FAILURE ---
    print(f"[{run_dir}] FRONTEND PHASE FAILED.")
    return False

def _validate_frontend_files(files: dict) -> bool:
    if not files or not isinstance(files, dict):
        return False
        
    required = ["package.json", "index.html", "src/App.jsx", "src/main.jsx"]
    missing = [f for f in required if f not in files]
    
    if missing:
        # Check if keys have prefix e.g. "frontend/package.json"
        # We might need to normalize keys, but for strict validation we rely on pure filenames mostly
        # Let's try to be smart scan
        found = 0
        for r in required:
            for k in files.keys():
                if k.endswith(r):
                    found += 1
                    break
        if found < len(required):
            return False
            
    # Check for empty content
    for k, v in files.items():
        if not v or len(v.strip()) == 0:
            return False
            
    return True

def _write_files(root_dir: str, files: dict):
    for filename, content in files.items():
        # Clean path (remove 'frontend/' prefix if LLM added it)
        clean_name = filename
        if clean_name.startswith("frontend/") or clean_name.startswith("frontend\\"):
            clean_name = clean_name[9:]
        
        clean_name = clean_name.lstrip("/\\")
        full_path = os.path.join(root_dir, clean_name)
        
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
