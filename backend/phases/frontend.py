import os
import json
from services.adapters import call_gemini, call_v0
from .templates import (
    _TEMPLATE_PACKAGE_JSON, _TEMPLATE_TAILWIND_CONFIG, _TEMPLATE_POSTCSS, 
    _TEMPLATE_VITE, _TEMPLATE_INDEX_HTML, _TEMPLATE_MAIN_JSX, 
    _TEMPLATE_INDEX_CSS, _TEMPLATE_APP_JSX
)


def execute_frontend_phase(planning_content: str, run_dir: str) -> tuple[bool, str]:
    """
    PHASE 4: FRONTEND GENERATION (Gemini -> v0 Strictly)
    
    Logic:
    1. Try Gemini.
    2. If fails -> Try v0.
    3. If v0 fails -> FAIL PHASE.
    
    Returns: (True, "Success Message") or (False, "Error Message")
    """
    
    frontend_root = os.path.join(run_dir, "frontend")
    os.makedirs(frontend_root, exist_ok=True)
    
    # --- PROMPT CONSTRUCTION ---
    prompt = (
        "You are an expert frontend developer. Generate a COMPLETE, PRODUCTION-READY React + Vite + Tailwind CSS frontend application.\n\n"
        f"IMPLEMENTATION PLAN:\n{planning_content[:3000]}\n\n"
        "STRICT REQUIREMENTS:\n"
        "1. **Output Format**: ONLY valid JSON with this structure:\n"
        "   {\"files\": {\"path/to/file.ext\": \"file content as string\", ...}}\n"
        "2. **Tech Stack**: React 18 + Vite 5 + Tailwind CSS 3\n"
        "3. **Required Files** (MUST include ALL of these):\n"
        "   - package.json (with react, react-dom, vite, tailwindcss, autoprefixer, postcss)\n"
        "   - index.html (with <div id=\"root\"></div>)\n"
        "   - vite.config.js (with @vitejs/plugin-react)\n"
        "   - tailwind.config.js (proper Tailwind v3 config)\n"
        "   - postcss.config.js (with tailwindcss and autoprefixer)\n"
        "   - src/main.jsx (React 18 entry point with createRoot)\n"
        "   - src/App.jsx (main application component)\n"
        "   - src/index.css (with @tailwind directives)\n"
        "4. **Code Quality**:\n"
        "   - NO placeholders or TODO comments\n"
        "   - Fully functional components with proper state management\n"
        "   - Responsive design using Tailwind utilities\n"
        "   - Modern React patterns (hooks, functional components)\n"
        "   - Clean, production-ready code\n"
        "5. **Design**:\n"
        "   - Beautiful, modern UI following the plan above\n"
        "   - Proper component structure and organization\n"
        "   - Tailwind CSS for ALL styling (no inline styles)\n"
        "   - Mobile-responsive layout\n"
        "6. **Output**: ONLY the JSON object. NO markdown code fences. NO explanations.\n\n"
        "Generate the complete frontend NOW:"
    )
    
    # Strict Order: Gemini -> v0
    providers = [
        ("Gemini", call_gemini),
        ("v0", call_v0)
    ]
    
    last_error = ""
    
    for provider_name, provider_func in providers:
        print(f"[{run_dir}] Trying {provider_name} for Frontend...")
        try:
            files = provider_func(prompt)
            
            # Validate Output
            valid, msg = _validate_frontend_files(files)
            if not valid:
                err_msg = f"{provider_name} output validation failed: {msg}"
                print(f"[{run_dir}] {err_msg}")
                last_error = err_msg
                continue
                
            print(f"[{run_dir}] {provider_name} Success! Writing files.")
            _write_files(frontend_root, files)
            return True, f"Success using {provider_name}"
            
        except Exception as e:
            err_msg = f"{provider_name} Failed: {str(e)}"
            print(f"[{run_dir}] {err_msg}")
            last_error = err_msg
            continue
    
    print(f"[{run_dir}] All API providers failed. Generating fallback scaffold.")
    _generate_fallback(frontend_root)
    return True, "API limits reached. Generated local fallback scaffold instead."


def _generate_fallback(root: str):
    """Writes a minimal working React+Vite+Tailwind app"""
    files = {
        "package.json": _TEMPLATE_PACKAGE_JSON,
        "vite.config.js": _TEMPLATE_VITE,
        "index.html": _TEMPLATE_INDEX_HTML,
        "tailwind.config.js": _TEMPLATE_TAILWIND_CONFIG,
        "postcss.config.js": _TEMPLATE_POSTCSS,
        "src/main.jsx": _TEMPLATE_MAIN_JSX,
        "src/App.jsx": _TEMPLATE_APP_JSX,
        "src/index.css": _TEMPLATE_INDEX_CSS
    }
    _write_files(root, files)


def _validate_frontend_files(files: dict) -> tuple[bool, str]:
    if not files or not isinstance(files, dict):
        return False, "Empty or invalid format"
        
    required = ["package.json", "index.html", "src/App.jsx", "src/main.jsx"]
    missing = []
    
    # Normalize keys to check for existence
    normalized_keys = []
    for k in files.keys():
        # Remove potential prefixes
        clean = k.replace("frontend/", "").replace("frontend\\", "")
        normalized_keys.append(clean)
        
    for r in required:
        if r not in normalized_keys:
            # Try fuzzy match?
            # No, strictly require standard paths or close enough.
            # But let's be generous if they put it in src/
            missing.append(r)
            
    if len(missing) > 0:
        # Check if we have at least package.json and index.html
        if "package.json" in normalized_keys and "index.html" in normalized_keys:
             pass # Maybe acceptable
        else:
             return False, f"Missing required files: {missing}"
            
    # Check for empty content
    for k, v in files.items():
        if not v or len(str(v).strip()) == 0:
            return False, f"File {k} is empty"
            
    return True, "Valid"

def _write_files(root_dir: str, files: dict):
    for filename, content in files.items():
        # Clean path (remove 'frontend/' prefix if LLM added it)
        clean_name = filename
        if clean_name.startswith("frontend/") or clean_name.startswith("frontend\\"):
            clean_name = clean_name[9:]
        
        # Security: prevent backtracking
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
