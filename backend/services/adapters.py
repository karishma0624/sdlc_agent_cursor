import os
import requests
import json
from typing import Dict, Any, Optional

# ==============================================================================
# STRICT PROVIDER ROUTING ADAPTERS
# ==============================================================================
# Mistral API       -> Requirements, Planning
# Gemini API        -> Frontend (Primary)
# Mermaid (LLM)     -> Design (Diagrams)
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
        raise ValueError("MISTRAL_API_KEY is missing. Cannot proceed with Requirements/Planning.")
        
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
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"Mistral API Call Failed: {str(e)}")


def call_mermaid(prompt: str, intent_context: Dict) -> str:
    """
    Approximation: Calls a capable LLM (Gemini or OpenAI) specifically to generate Mermaid syntax.
    We route this to Gemini by default as 'Mermaid Provider'.
    """
    # Strict routing: Use Gemini for Mermaid generation as it's good at syntax
    # If standard Mermaid service exists, we would use it, but here we generate code.
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback to simple deterministic graph if no Keys
        return "graph TD;\n    Start-->End;"

    # We reuse the logic but strictly for mermaid
    # If Gemini available
    if os.getenv("GEMINI_API_KEY"):
        return _call_gemini_text(f"{prompt}\n\nContext: {json.dumps(intent_context)}", model="gemini-1.5-flash")
    
    return "graph TD;\n    Error[Missing Key]-->Stop;"


def call_gemini(prompt: str, model: str = "gemini-1.5-flash") -> Dict[str, str]:
    """
    Strictly calls Gemini for JSON file generation (Frontend Phase).
    Expects specific JSON structure in return.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Cannot proceed with Frontend Primary.")

    url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    # Enforce JSON structure via prompt engineering (Gemini supports responseSchema but we'll stick to prompt for now for compat)
    final_prompt = (
        f"{prompt}\n\n"
        "IMPORTANT: Output ONLY valid JSON. No markdown fences. No explanations.\n"
        "Format: {\"files\": {\"path/to/file\": \"content\"}}"
    )
    
    payload = {
        "contents": [{"parts": [{"text": final_prompt}]}],
        "generationConfig": {"temperature": 0.2}
    }
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        
        # Extract Text
        cands = data.get("candidates", [])
        if not cands:
            raise ValueError("Gemini returned no candidates.")
            
        text = cands[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        
        # Parse JSON
        return _parse_json_garbage(text)
        
    except Exception as e:
        raise RuntimeError(f"Gemini API Call Failed: {str(e)}")


def call_v0(prompt: str) -> Dict[str, str]:
    """
    Fallback for Frontend. Calls v0.dev API.
    """
    api_key = os.getenv("V0_API_KEY") or os.getenv("V0_DEV_API_KEY")
    if not api_key:
        raise ValueError("V0_API_KEY is missing. Cannot proceed with Frontend Fallback.")
        
    base = os.getenv("V0_API_BASE", "https://api.v0.dev")
    url = f"{base}/generate"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "task": "frontend_only", 
        "stack": "react+tailwind", 
        "prompt": prompt
    }
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("files", {})
    except Exception as e:
        raise RuntimeError(f"v0 API Call Failed: {str(e)}")


# ==============================================================================
# HELPERS (Internal)
# ==============================================================================

def _call_gemini_text(prompt: str, model: str) -> str:
    """Internal helper for text generation (used for Mermaid)"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return "graph TD; Error-->NoKey;"
    
    url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if not resp.ok: return "graph TD; Error-->API_Fail;"
        data = resp.json()
        text = data.get("candidates", [])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        return text
    except:
        return "graph TD; Error-->Exception;"

def _parse_json_garbage(text: str) -> Dict[str, Any]:
    """
    Cleans up LLM markdown garbage to extract JSON.
    """
    text = text.strip()
    # Remove markdown fences
    if text.startswith("```"):
        import re
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try finding { ... }
        import re
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        raise ValueError("Could not parse JSON from response.")
