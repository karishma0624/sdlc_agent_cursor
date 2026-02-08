from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import io
import os
import base64
import requests
from tenacity import retry, wait_exponential, stop_after_attempt, wait_fixed


class InferenceRouter:
    """Unified routing with graceful fallbacks. Uses a simple local baseline when no keys set."""

    def __init__(self) -> None:
        self.providers = self._detect_providers()
        # Active validation (optional but recommended for debugging)
        # self.validate_providers()

    def refresh(self) -> None:
        """Re-detect providers (e.g., after loading .env)."""
        self.providers = self._detect_providers()

    def _detect_providers(self):
        hf_key = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
        ollama_available = False
        # Prefer explicit env, otherwise probe localhost with short timeout
        base = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        try:
            # Very short timeout for probe
            resp = requests.get(base.rstrip("/") + "/api/tags", timeout=1.0)
            if resp.ok:
                ollama_available = True
        except Exception:
            ollama_available = False
            
        return {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "gemini": bool(os.getenv("GEMINI_API_KEY")),
            "mistral": bool(os.getenv("MISTRAL_API_KEY")),
            "groq": bool(os.getenv("GROQ_API_KEY")),
            "hf": bool(hf_key),
            "perplexity": bool(os.getenv("PERPLEXITY_API_KEY")),
            "ollama": bool(os.getenv("OLLAMA_BASE_URL")) or ollama_available,
            "lovable": True,
            "stich": True,
            "v0": bool(os.getenv("V0_API_KEY") or os.getenv("V0_DEV_API_KEY")),
        }

    def _get_hf_key(self) -> Optional[str]:
        return os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")

    @retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(1))
    def _v0_generate_frontend(self, prompt: str) -> Dict[str, Any]:
        """Call v0.dev to generate frontend assets."""
        api_key = os.getenv("V0_API_KEY") or os.getenv("V0_DEV_API_KEY")
        base = os.getenv("V0_API_BASE", "https://api.v0.dev")
        url = f"{base}/generate"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"task": "frontend_only", "stack": "react+tailwind", "prompt": prompt}
        resp = requests.post(url, headers=headers, json=payload, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        files = data.get("files") or {}
        return {"files": files, "provider": "v0", "model": data.get("model", "free"), "notes": data.get("instructions", "")}

    def classify_image(self, image: Image.Image) -> Dict[str, Any]:
        if self.providers.get("hf"):
            try:
                label, score = self._hf_classify(image)
                return {"label": label, "confidence": float(score), "provider": "hf", "model": "vit", "fallback": False}
            except: pass
        
        # Fallback
        pixels = image.resize((32, 32))
        avg = sum(p[0] + p[1] + p[2] for p in pixels.getdata()) / (32 * 32 * 3)
        label = "cow" if avg < 128 else "cat"
        return {"label": label, "confidence": 0.65, "provider": "local", "model": "baseline", "fallback": True}


    def generate_text(self, prompt: str, preference: Optional[List[str]] = None) -> Dict[str, Any]:
        """General text generation with token accounting."""
        strategies: List[str] = []
        if preference:
            for name in preference:
                if self.providers.get(name):
                    strategies.append(name)
        
        # Default fallback priority
        for name in ["gemini", "perplexity", "mistral", "groq", "openai", "hf", "ollama"]:
            if self.providers.get(name) and name not in strategies:
                strategies.append(name)

        errors = []
        for name in strategies:
            try:
                # STRICT timeouts enforced in individual methods
                if name == "gemini":
                    text, tokens = self._gemini_chat(prompt)
                    if text: return {"output": text, "provider": "gemini", "model": "gemini-1.5-flash", "tokens": tokens}
                elif name == "perplexity":
                    text, tokens = self._perplexity_chat(prompt)
                    if text: return {"output": text, "provider": "perplexity", "model": "sonar-small-chat", "tokens": tokens}
                elif name == "mistral":
                    text, tokens = self._mistral_chat(prompt)
                    if text: return {"output": text, "provider": "mistral", "model": "mistral-small-latest", "tokens": tokens}
                elif name == "groq":
                    text, tokens = self._groq_chat(prompt)
                    if text: return {"output": text, "provider": "groq", "model": "llama-3.1-8b-instant", "tokens": tokens}
                elif name == "openai":
                    text, tokens = self._openai_chat(prompt)
                    if text: return {"output": text, "provider": "openai", "model": "gpt-4o-mini", "tokens": tokens}
                elif name == "hf":
                    text, tokens = self._hf_generate_text(prompt)
                    if text: return {"output": text, "provider": "hf", "model": "Qwen2.5-7B-Instruct", "tokens": tokens}
                elif name == "ollama":
                    text, tokens = self._ollama_chat(prompt)
                    if text: return {"output": text, "provider": "ollama", "model": "codellama", "tokens": tokens}
            except Exception as e:
                errors.append(f"{name}: {str(e)}")
                continue

        # If we get here, all failed
        return {"output": "", "provider": "none", "model": "none", "error": "; ".join(errors), "fallback": True}

    def generate_code(self, instruction: str, preference: Optional[List[str]] = None) -> Dict[str, Any]:
        """Ask providers to return a JSON object mapping file paths to contents."""
        prompt = (
            "Return ONLY a JSON object where keys are relative file paths and values are file contents. "
            "Do not include explanations. "
            f"Instruction: {instruction}"
        )
        # Force a preference list if not provided
        default_pref = ["gemini", "openai", "mistral", "groq", "ollama"]
        result = self.generate_text(prompt, preference=preference or default_pref)
        
        text = result.get("output", "{}")
        if not text:
             # Explicitly return empty so caller knows to use fallback
             return {"files": {}, "provider": "failed", "model": "none"}

        files: Dict[str, str] = {}
        try:
            files = self._parse_files_json(text)
        except Exception:
            import re, json as _json
            # 1. Try finding markdown code blocks
            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if m:
                try: files = _json.loads(m.group(1))
                except: pass
            
            if not files:
                 # 2. Try to find the outer-most JSON object
                 try:
                    start = text.find("{")
                    end = text.rfind("}")
                    if start != -1 and end != -1:
                        files = _json.loads(text[start:end+1])
                 except: pass

        # POST-PROCESSING: Ensure files is a dict of strings
        if not isinstance(files, dict):
            files = {}
        
        clean_files = {}
        for k, v in files.items():
            if isinstance(k, str) and isinstance(v, str):
                clean_files[k] = v
        
        result["files"] = clean_files
        return result

    def _parse_files_json(self, text: str) -> Dict[str, str]:
        import json as _json
        data = _json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("files payload is not a dict")
        return {str(k): str(v) for k, v in data.items()}

    def _pick_provider_for_vision(self):
        order = ["openai", "gemini", "groq", "hf", "ollama"]
        for name in order:
            if self.providers.get(name):
                return name, "auto"
        return "local", "baseline"

    def _pick_provider_for_codegen(self):
        order = ["openai", "mistral", "groq", "hf", "ollama"]
        for name in order:
            if self.providers.get(name):
                return name, "auto"
        return "local", "baseline"

    @retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(1))
    def _hf_classify(self, image: Image.Image):
        api_key = self._get_hf_key()
        headers = {"Authorization": f"Bearer {api_key}"}
        # Use a general image classification model
        url = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        resp = requests.post(url, headers=headers, data=buf.getvalue(), timeout=12)
        resp.raise_for_status()
        data = resp.json()
        # HF may return nested lists
        preds = data[0] if isinstance(data, list) and data and isinstance(data[0], list) else data
        best = max(preds, key=lambda x: x.get("score", 0))
        return best.get("label", "unknown"), best.get("score", 0.0)

    @retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(1))
    def _hf_generate_text(self, prompt: str) -> Tuple[str, int]:
        api_key = os.getenv("HF_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        # Use instruct-tuned model for text generation
        url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256, "temperature": 0.3}}
        resp = requests.post(url, headers=headers, json=payload, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        text = (
            data[0]["generated_text"]
            if isinstance(data, list) and data and "generated_text" in data[0]
            else data.get("generated_text", str(data))
        )
        return text, len(text) // 4

    @retry(wait=wait_fixed(1), stop=stop_after_attempt(1))
    def _openai_chat(self, prompt: str) -> Tuple[str, int]:
        api_key = os.getenv("OPENAI_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        tokens = int(data.get("usage", {}).get("total_tokens", 0))
        if not tokens:
            tokens = len(text) // 4
        return text, tokens

    @retry(wait=wait_fixed(1), stop=stop_after_attempt(1))
    def _mistral_chat(self, prompt: str) -> Tuple[str, int]:
        api_key = os.getenv("MISTRAL_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        url = "https://api.mistral.ai/v1/chat/completions"
        payload = {"model": "mistral-small-latest", "messages": [{"role":"user","content": prompt}], "temperature": 0.2}
        resp = requests.post(url, headers=headers, json=payload, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        tokens = int(data.get("usage", {}).get("total_tokens", 0)) if isinstance(data.get("usage"), dict) else 0
        if not tokens:
            tokens = len(text) // 4
        return text, tokens

    @retry(wait=wait_fixed(1), stop=stop_after_attempt(1))
    def _groq_chat(self, prompt: str) -> Tuple[str, int]:
        api_key = os.getenv("GROQ_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        # Groq is OpenAI-compatible endpoint
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {"model": "llama-3.1-8b-instant", "messages": [{"role":"user","content": prompt}], "temperature": 0.2}
        resp = requests.post(url, headers=headers, json=payload, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        tokens = int(data.get("usage", {}).get("total_tokens", 0)) if isinstance(data.get("usage"), dict) else 0
        if not tokens:
            tokens = len(text) // 4
        return text, tokens

    @retry(wait=wait_fixed(1), stop=stop_after_attempt(1))
    def _gemini_chat(self, prompt: str) -> Tuple[str, int]:
        api_key = os.getenv("GEMINI_API_KEY")
        headers = {"Content-Type": "application/json"}
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}}
        resp = requests.post(url, headers=headers, json=payload, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        # Parse text from candidates
        cands = data.get("candidates", [])
        if not cands:
            return ""
        parts = cands[0].get("content", {}).get("parts", [])
        text = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict))
        tokens = int(data.get("usageMetadata", {}).get("totalTokenCount", 0)) if isinstance(data.get("usageMetadata"), dict) else 0
        if not tokens:
            tokens = len(text) // 4
        return text, tokens

    @retry(wait=wait_fixed(1), stop=stop_after_attempt(1))
    def _perplexity_chat(self, prompt: str) -> Tuple[str, int]:
        api_key = os.getenv("PERPLEXITY_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        url = "https://api.perplexity.ai/chat/completions"
        payload = {"model": "sonar-small-chat", "messages": [{"role":"user","content": prompt}], "temperature": 0.2}
        resp = requests.post(url, headers=headers, json=payload, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        tokens = int(data.get("usage", {}).get("total_tokens", 0)) if isinstance(data.get("usage"), dict) else 0
        if not tokens:
            tokens = len(text) // 4
        return text, tokens

    @retry(wait=wait_fixed(1), stop=stop_after_attempt(1))
    def _ollama_chat(self, prompt: str) -> Tuple[str, int]:
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        url = f"{base}/api/generate"
        payload = {"model": "codellama", "prompt": prompt, "stream": False}
        resp = requests.post(url, json=payload, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "")
        return text, len(text) // 4


