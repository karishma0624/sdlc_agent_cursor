import os
import requests
import json
import re
from typing import Dict, Any, Optional

# ==============================================================================
# STRICT PROVIDER ROUTING ADAPTERS
# ==============================================================================
# Mistral API       -> Requirements, Planning
# Gemini API        -> Frontend (Primary)
# Mermaid (LLM)     -> Design (Diagrams - via Gemini)
# v0 API            -> Frontend (Fallback)
# ==============================================================================

def call_mistral(prompt: str, model: str = "mistral-small-latest") -> str:
    """
    Strictly calls Mistral API for text generation. 
    Used for Phase 1 (Requirements) and Phase 2 (Planning).
    Returns raw text content.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is missing.")
        
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}", 
        "Content-Type": "application/json"
    }
    payload = {
        "model": model, 
        "messages": [{"role": "user", "content": prompt}], 
        "temperature": 0.2
    }
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"Mistral API Call Failed: {str(e)}")


def call_mermaid(prompt: str, intent_context: Dict) -> str:
    """
    Calls Gemini to generate Mermaid syntax.
    """
    if not os.getenv("GEMINI_API_KEY"):
        return "graph TD;\n    Error[Missing GEMINI_KEY]-->Stop;"

    return call_gemini_text(f"{prompt}\n\nContext: {json.dumps(intent_context)}", model="gemini-1.5-flash")


def _get_gemini_models(api_key: str) -> list[str]:
    """Dynamically fetch available models from Gemini API"""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        resp = requests.get(url, timeout=10)
        if resp.ok:
            data = resp.json()
            # Filter for generateContent support
            models = [m['name'].replace('models/', '') for m in data.get('models', []) 
                     if 'generateContent' in m.get('supportedGenerationMethods', [])]
            # Prioritize flash models
            models.sort(key=lambda x: 'flash' not in x)
            return models
    except:
        pass
    return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]

def call_gemini(prompt: str, model: str = "gemini-1.5-flash") -> Dict[str, str]:
    """
    Strictly calls Gemini for JSON file generation (Frontend Phase).
    Tries multiple models in order of preference if the default fails.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing.")

    # Dynamically correct models + fallbacks
    models_to_try = _get_gemini_models(api_key)
    # Remove duplicates while preserving order
    unique_models = []
    [unique_models.append(m) for m in models_to_try if m not in unique_models]
    
    last_error = None

    for current_model in unique_models:
        # Use v1beta for newer models, v1 for older if needed, but v1beta is generally safer for all recently
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        
        final_prompt = (
            f"{prompt}\n\n"
            "IMPORTANT: Output ONLY valid JSON. No markdown fences. No explanations.\n"
            "The JSON must strictly follow this schema: {\"files\": {\"path/to/file.ext\": \"file content string\"}}\n"
        )
        
        payload = {
            "contents": [{"parts": [{"text": final_prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json"
            }
        }
        
        try:
            print(f"  > Attempting Gemini model: {current_model}")
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            
            # If 404 or 400, try next model
            if resp.status_code in [404, 400, 500, 503]:
                 error_msg = f"{resp.status_code} {resp.text}"
                 print(f"  > Gemini {current_model} failed: {error_msg}")
                 last_error = error_msg
                 continue
                 
            resp.raise_for_status()
            data = resp.json()
            
            cands = data.get("candidates", [])
            if not cands:
                # Check for prompt feedback block etc.
                if "promptFeedback" in data:
                     print(f"  > Prompt blocked: {data['promptFeedback']}")
                raise ValueError("Gemini returned no candidates.")
                
            text = cands[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
            parsed = _parse_json_garbage(text)
            if "files" not in parsed:
                # Fallback for structure mismatch
                if any(k.startswith("src/") for k in parsed.keys()):
                     return parsed
                # If parsed is a list or just string, fail.
                # But sometimes it returns just the file content? unlikely with this prompt.
                # Heuristic: check if keys look like file paths
                if isinstance(parsed, dict) and len(parsed) > 0:
                     # Check if keys have extensions
                     if any("." in k for k in parsed.keys()):
                         return parsed

                raise ValueError("Gemini response missing 'files' key.")
            return parsed["files"]
            
        except Exception as e:
            last_error = str(e)
            print(f"  > Gemini {current_model} Exception: {e}")
            continue

    raise RuntimeError(f"Gemini API Call Failed (All models): {last_error}")


def call_v0(prompt: str) -> Dict[str, str]:
    """
    Fallback for Frontend. Calls v0.dev API with robust error handling.
    Try 'generate' endpoint first, then 'chat/completions'.
    """
    api_key = os.getenv("V0_API_KEY") or os.getenv("V0_DEV_API_KEY")
    if not api_key:
        raise ValueError("V0_API_KEY is missing.")
        
    base = os.getenv("V0_API_BASE", "https://api.v0.dev") 
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    # Strategy 1: /generate (Custom/Unofficial common pattern)
    url = f"{base}/generate"
    payload = {
        "task": "frontend_only", 
        "stack": "react+tailwind", 
        "prompt": prompt
    }
    
    # Strategy 2: /chat/completions (OpenAI Compatible)
    url_chat = f"{base}/chat/completions"
    payload_chat = {
        "model": "v0-preview",
        "messages": [
            {"role": "system", "content": "You are a frontend generator. Output JSON with 'files' key."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    try:
        # Try Strategy 1
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 404:
             # Try Strategy 2
             resp = requests.post(url_chat, headers=headers, json=payload_chat, timeout=60)
        
        resp.raise_for_status()
        data = resp.json()
        
        # Handle response format variation
        if "files" in data:
            return data["files"]
        elif "choices" in data:
            text = data["choices"][0]["message"]["content"]
            parsed = _parse_json_garbage(text)
            return parsed.get("files", parsed)
        else:
            raise ValueError(f"Unknown v0 response format: {data.keys()}")

    except Exception as e:
        # Final attempt to parse if valid JSON even if error? No.
        raise RuntimeError(f"v0 API Call Failed: {str(e)}")


# ==============================================================================
# HELPERS
# ==============================================================================

def call_gemini_text(prompt: str, model: str = "gemini-1.5-flash") -> str:
    """
    Generic Gemini Text Generation.
    Used for Intent Classification and Mermaid.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return "Error: Missing GEMINI_API_KEY"
    
    # Simple fallback model logic
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if not resp.ok: 
            return f"Error: API_{resp.status_code} {resp.text}"
        data = resp.json()
        return data.get("candidates", [])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
    except Exception as e:
        return f"Error: {str(e)}"


def _parse_json_garbage(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except: pass
        # Last ditch: fix common errors?
        pass
    
    raise ValueError(f"Could not parse JSON. Preview: {text[:200]}...")
