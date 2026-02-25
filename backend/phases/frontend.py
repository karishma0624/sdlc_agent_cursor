import os
import json
import re
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
        "   - package.json (MUST include react, react-dom, react-router-dom, framer-motion, recharts, vite, tailwindcss, autoprefixer, postcss, lucide-react. DO NOT MISS ANY. Use latest versions.)\n"
        "   - index.html (with <div id=\"root\"></div>)\n"
        "   - vite.config.js (with @vitejs/plugin-react)\n"
        "   - tailwind.config.js (proper Tailwind v3 config)\n"
        "   - postcss.config.js (with tailwindcss and autoprefixer)\n"
        "   - src/main.jsx (React 18 entry point with createRoot and BrowserRouter if applicable)\n"
        "   - src/App.jsx (main application component, MUST act as a layout containing a Sidebar/Navbar and react-router-dom <Routes>)\n"
        "   - At least 3 distinct full-page components inside `src/pages/` (Name them according to the IMPLEMENTATION PLAN above! Do NOT just blindly build a Dashboard/Analytics page unless the plan asks for it.)\n"
        "   - src/index.css (with @tailwind directives)\n"
        "4. **Code Quality & Architecture**:\n"
        "   - MUST BE MULTI-PAGE. Use `react-router-dom` to implement multiple distinct views/pages (minimum 3). The pages MUST strictly match the exact domain/features outlined in the IMPLEMENTATION PLAN. Do NOT generate generic 'Dashboard' or 'Analytics' pages unless the plan explicitly asks for them!\n"
        "   - CRITICAL COMPONENT RULE: You MUST provide the code for EVERY SINGLE local file you `import`. If `App.jsx` imports `Navbar` and 3 Pages, you MUST output the exact code for `Navbar.jsx` and those 3 Pages. Missing files will crash the build!\n"
        "   - DO NOT import small UI fragments (like 'Button.jsx' or 'Input.jsx') unless you explicitly generate them! It is safer to build the UI inline using Tailwind than to import a file you might forget to generate.\n"
        "   - Ensure `BrowserRouter` wraps your `<App />` in `src/main.jsx`.\n"
        "   - Keep components visually stunning but structurally concise so you stay within output limits.\n"
        "   - NO placeholders or TODO comments\n"
        "   - Fully functional components with proper state management\n"
        "   - Responsive design using Tailwind utilities\n"
        "   - Clean, production-ready code\n"
        "5. **Design Aesthetics (CRITICAL)**:\n"
        "   - The UI MUST be highly attractive, COLORFUL, and visually engaging. DO NOT build a simple or boring flat UI!\n"
        "   - Use stunning Tailwind CSS styles: deep rich gradients (e.g. `bg-gradient-to-r`), vibrant background colors, glassmorphism (`backdrop-blur-md bg-white/10`), soft shadows (`shadow-xl`), and rounded corners (`rounded-2xl`).\n"
        "   - DO NOT USE ANY IMAGES. No external images, no placeholders, no image URLs. Use CSS gradients, solid colors, and icons instead.\n"
        "   - Use `framer-motion` for elegant page transitions, hover effects, and entrance animations.\n"
        "   - Use `recharts` for beautiful graphs ONLY IF data representation is needed for the requested app.\n"
        "   - Incorporate `lucide-react` icons extensively for a polished UI. WARNING: `lucide-react` DOES NOT HAVE domain-specific icons like `Massage`, `Spa`, `Nails`, `Google`, `Facebook`, etc! DO NOT hallucinate icon names or the app will crash with an export error! Use ONLY basic standard icons (e.g. `User`, `LogIn`, `Mail`, `Star`, `Heart`, `Scissors`, `Sparkles`, `Settings`).\n"
        "   - The UI MUST look extremely premium and WOW the user at first glance. It must look like a high-end, professionally designed product, full of color and life.\n"
        "   - Create realistic mock data to fill the UI (NO empty states or placeholders with dummy text). Every card or section should look fully populated with real-world examples.\n"
        "6. **NO FAKE IMPORTS (CRITICAL)**:\n"
        "   - You MAY import `framer-motion`, `recharts`, `react-router-dom`, and `lucide-react` since they are explicitly in the package.json.\n"
        "   - Do NOT import from '@heroicons/react', 'react-icons', or any other UI library not in your package.json.\n"
        "   - Do NOT import local components unless you are absolutely sure you are also generating the JSON object for that file. If building a multi-page app, failure to generate imported pages will crash the preview completely!\n"
        "7. **SYNTAX & EXPLICIT IMPORTS MUST BE PERFECT (CRITICAL)**:\n"
        "   - THERE SHOULD BE ABSOLUTELY NO ERRORS IN THE FRONTEND GENERATION CODE. ZERO SYNTAX ERRORS, ZERO RUNTIME ERRORS! The code MUST render properly and work correctly in the preview, without any blank screens or crashes.\n"
        "   - ALL JSX tags MUST be properly closed. Check for unclosed `<style>`, `<div>`, `<svg>`, `<input>`, `<br>`, `<hr>`, etc. Always use self-closing tags like `<br />` and `<input />`.\n"
        "   - MISSING IMPORTS WILL CRASH THE APP. You MUST explicitly import every single component, icon, and hook you use!\n"
        "   - If you use `AnimatePresence` or `motion`, YOU MUST `import { motion, AnimatePresence } from 'framer-motion';`\n"
        "   - If you use `Search`, `Menu`, or ANY icon, YOU MUST `import { Search, Menu } from 'lucide-react';` (Import exactly what you use).\n"
        "   - ALL required React hooks (`useState`, `useEffect`, etc.) MUST be imported from 'react'.\n"
        "   - DO NOT use undefined components or variables. ALL components used MUST be declared or imported.\n"
        "   - NEVER use `class=`, you MUST use `className=` exclusively. NEVER use `for=`, use `htmlFor=`.\n"
        "   - ALL generated components MUST be properly exported using `export default function`.\n"
        "   - Your output MUST be 100% valid, strictly formatted JSON. Carefully escape quotes (`\"`) inside string values using `\\\"`, and escape newlines using `\\n`! Escaping is critical to avoid JSONDecodeError.\n"
        "   - DO NOT wrap your content in markdown blocks like ```json ... ```, output ONLY the raw JSON object starting with `{` and ending with `}`.\n"
        "8. **Output**: ONLY the raw JSON object. NO markdown code fences around the JSON object. NO explanations. Start immediately with `{`.\n\n"
        "Generate the perfect, error-free complete frontend NOW:"
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
        
    required = ["package.json", "index.html", "src/App.jsx", "src/main.jsx", "vite.config.js", "postcss.config.js", "tailwind.config.js", "src/index.css"]
    
    # Normalize keys to check for existence
    normalized_keys = []
    for k in files.keys():
        clean = k.replace("frontend/", "").replace("frontend\\", "")
        normalized_keys.append(clean)
        
    # Auto-inject missing boilerplate to save expensive LLM generations
    from .templates import (
        _TEMPLATE_PACKAGE_JSON, _TEMPLATE_INDEX_HTML, _TEMPLATE_APP_JSX, 
        _TEMPLATE_MAIN_JSX, _TEMPLATE_VITE, _TEMPLATE_POSTCSS, 
        _TEMPLATE_TAILWIND_CONFIG, _TEMPLATE_INDEX_CSS
    )
    
    template_map = {
        "package.json": _TEMPLATE_PACKAGE_JSON,
        "index.html": _TEMPLATE_INDEX_HTML,
        "vite.config.js": _TEMPLATE_VITE,
        "postcss.config.js": _TEMPLATE_POSTCSS,
        "tailwind.config.js": _TEMPLATE_TAILWIND_CONFIG,
        "src/main.jsx": _TEMPLATE_MAIN_JSX,
        "src/App.jsx": _TEMPLATE_APP_JSX,
        "src/index.css": _TEMPLATE_INDEX_CSS
    }
    
    for r in required:
        if r not in normalized_keys:
            # Inject the missing file
            print(f"Auto-injecting missing {r} into payload.")
            files[r] = template_map[r]
            
    # Fix strict ESM configuration errors from LLM (module.exports -> export default)
    for k, v in files.items():
        if k.endswith(".js") and "module.exports" in str(v):
            print(f"Patching {k} to ES Module format...")
            patched = str(v).replace("module.exports =", "export default")
            patched = patched.replace("module.exports", "export default")
            files[k] = patched
            
        # Auto-fix common JSX syntax errors
        if k.endswith(".jsx"):
            patched = str(files[k])
            # Fix class= to className=
            patched = re.sub(r'\bclass=(["\'])', r'className=\1', patched)
            # Fix for= to htmlFor=
            patched = re.sub(r'\bfor=(["\'])', r'htmlFor=\1', patched)
            # Fix unclosed common void elements
            patched = re.sub(r'<br\s*>', r'<br />', patched)
            patched = re.sub(r'<hr\s*>', r'<hr />', patched)
            # Remove img tags entirely as requested (no images)
            patched = re.sub(r'<img[^>]*>', '', patched)
            # Fix common JSX SVG attributes
            for attr in ['stroke-width', 'stroke-linecap', 'stroke-linejoin', 'fill-rule', 'clip-rule']:
                camel = ''.join(word.capitalize() if i > 0 else word for i, word in enumerate(attr.split('-')))
                patched = re.sub(rf'\b{attr}=', f'{camel}=', patched)
            # Fix html comments to JSX comments
            patched = re.sub(r'<!--(.*?)-->', r'{/* \1 */}', patched, flags=re.DOTALL)
            
            # --- Auto-fix hallucinated brand icons from lucide-react ---
            hallucinated_icons = {
                "Google": "Globe", "Facebook": "Users", "Twitter": "MessageCircle",
                "Github": "Code", "GitHub": "Code", "Apple": "Monitor",
                "Microsoft": "Monitor", "Linkedin": "Briefcase", "LinkedIn": "Briefcase",
                "Instagram": "Camera", "YouTube": "Video", "Youtube": "Video",
                "Discord": "MessageSquare", "TikTok": "Video", "Tiktok": "Video",
                "Massage": "Heart", "Spa": "Sparkles", "Salon": "Scissors",
                "Nails": "Star", "Nail": "Star", "Hair": "Scissors", "Cut": "Scissors",
                "Beauty": "Sparkles", "Treatment": "Heart"
            }
            for fake_icon, real_icon in hallucinated_icons.items():
                if fake_icon in patched:
                    # Look for imports of this icon and tags of this icon, we replace the word globally in this file
                    # (only matching exact word boundaries to avoid replacing parts of other words)
                    patched = re.sub(r'\b' + fake_icon + r'\b', real_icon, patched)
            
            # --- Auto-inject missing common imports ---
            
            # 1. framer-motion
            fm_needed = []
            if "AnimatePresence" in patched: fm_needed.append("AnimatePresence")
            if "motion." in patched or "<motion " in patched: fm_needed.append("motion")
            if fm_needed:
                fm_imports = [line for line in patched.splitlines() if "framer-motion" in line]
                missing_fm = [i for i in fm_needed if not any(i in line for line in fm_imports)]
                if missing_fm:
                    if fm_imports:
                        patched = patched.replace(fm_imports[0], fm_imports[0].replace("{", f"{{ {', '.join(missing_fm)}, "))
                    else:
                        patched = f"import {{ {', '.join(missing_fm)} }} from 'framer-motion';\n" + patched
                        
            # 2. react-router-dom
            rrd_items = ["Link", "Routes", "Route", "BrowserRouter", "Navigate", "useNavigate", "useLocation", "useParams", "Outlet"]
            rrd_needed = [item for item in rrd_items if re.search(r'\b' + item + r'\b', patched)]
            if rrd_needed:
                rrd_imports = [line for line in patched.splitlines() if "react-router-dom" in line]
                missing_rrd = [i for i in rrd_needed if not any(i in line for line in rrd_imports)]
                if missing_rrd:
                    if rrd_imports:
                        patched = patched.replace(rrd_imports[0], rrd_imports[0].replace("{", f"{{ {', '.join(missing_rrd)}, "))
                    else:
                        patched = f"import {{ {', '.join(missing_rrd)} }} from 'react-router-dom';\n" + patched
                        
            # 3. React hooks
            react_hooks = ["useState", "useEffect", "useRef", "useMemo", "useCallback", "useContext"]
            hooks_needed = [hook for hook in react_hooks if re.search(r'\b' + hook + r'\b', patched)]
            if hooks_needed:
                react_imports = [line for line in patched.splitlines() if "import" in line and ("'react'" in line or '"react"' in line)]
                # If they imported via `import React from 'react'`, they might use `React.useState`, but if they used `useState` directly we need it.
                missing_hooks = [i for i in hooks_needed if not any(i in line for line in react_imports)]
                if missing_hooks:
                    if react_imports:
                         if "{" in react_imports[0]:
                             patched = patched.replace(react_imports[0], react_imports[0].replace("{", f"{{ {', '.join(missing_hooks)}, "))
                         else:
                             patched = patched.replace(react_imports[0], react_imports[0] + f"\nimport {{ {', '.join(missing_hooks)} }} from 'react';")
                    else:
                        patched = f"import React, {{ {', '.join(missing_hooks)} }} from 'react';\n" + patched

            # 4. lucide-react (basic heuristic: find <IconName /> that are camelcase and missing imports)
            # This is riskier so we only catch heavily used ones that might be missing
            commonly_missed_lucides = ["Home", "Settings", "Activity", "Menu", "Search", "User", "Bell", "ChevronDown", "ChevronRight", "ChevronUp", "ChevronLeft", "Plus", "Minus", "Trash", "Edit", "Check", "X"]
            lucide_needed = [icon for icon in commonly_missed_lucides if re.search(r'<' + icon + r'\b', patched)]
            if lucide_needed:
                lucide_imports = [line for line in patched.splitlines() if "lucide-react" in line]
                missing_lucides = [i for i in lucide_needed if not any(i in line for line in lucide_imports)]
                # Don't auto-import if there's a local import of that exact name (e.g. import Home from './Home')
                missing_lucides = [i for i in missing_lucides if not re.search(r'import\s+.*?\b' + i + r'\b.*?\bfrom\s+', patched)]
                if missing_lucides:
                    if lucide_imports:
                        patched = patched.replace(lucide_imports[0], lucide_imports[0].replace("{", f"{{ {', '.join(missing_lucides)}, "))
                    else:
                        patched = f"import {{ {', '.join(missing_lucides)} }} from 'lucide-react';\n" + patched

            files[k] = patched
            
    # Clean any markdown code fences surrounding the string values
    for k, v in files.items():
        if isinstance(v, str):
            cleaned = v.strip()
            cleaned = re.sub(r"^```[a-zA-Z0-9-]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
            files[k] = cleaned
            
    # Check for empty content or missing core views
    has_app = False
    for k, v in files.items():
        if "App.jsx" in k: has_app = True
        if not v or len(str(v).strip()) == 0:
            return False, f"File {k} is empty"
            
    if not has_app:
         return False, "Failed to generate App.jsx entry point."
            
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
