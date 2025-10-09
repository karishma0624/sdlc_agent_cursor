from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import io
import os
import base64
import requests
from tenacity import retry, wait_exponential, stop_after_attempt


class InferenceRouter:
	"""Unified routing with graceful fallbacks. Uses a simple local baseline when no keys set."""

	def __init__(self) -> None:
		self.providers = self._detect_providers()

	def refresh(self) -> None:
		"""Re-detect providers (e.g., after loading .env)."""
		self.providers = self._detect_providers()

	def _detect_providers(self):
		hf_key = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
		ollama_available = False
		# Prefer explicit env, otherwise probe localhost
		base = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
		try:
			resp = requests.get(base.rstrip("/") + "/api/tags", timeout=1.5)
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
			"stitch": True,
			"v0": bool(os.getenv("V0_API_KEY") or os.getenv("V0_DEV_API_KEY")),
		}

	def _get_hf_key(self) -> Optional[str]:
		return os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
	def run_tool(self, tool: str, prompt: str) -> Dict[str, Any]:
		if tool == "v0":
			try:
				return self._v0_generate_frontend(prompt)
			except Exception as e:
				# Fall through to stitch/local
				pass
		if tool == "lovable":
			# Placeholder: return a scaffold JSON
			return {"files": {"README.md": "# Lovable scaffold\n"}, "provider": "lovable", "model": "free-tier"}
		if tool == "stitch":
			return {"files": {"frontend/App.jsx": "export default function App(){return <div>Stitch UI</div>}"}, "provider": "stitch", "model": "free-tier"}
		# local codegen via LLMs
		text = self.generate_text(prompt)
		return {"text": text.get("output", ""), "provider": text.get("provider"), "model": text.get("model")}

	@retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(3))
	def _v0_generate_frontend(self, prompt: str) -> Dict[str, Any]:
		"""Call v0.dev to generate frontend assets. Requires V0_API_KEY and optional V0_API_BASE."""
		api_key = os.getenv("V0_API_KEY") or os.getenv("V0_DEV_API_KEY")
		base = os.getenv("V0_API_BASE", "https://api.v0.dev")
		url = f"{base}/generate"
		headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
		payload = {"task": "frontend_only", "stack": "react+tailwind", "prompt": prompt}
		resp = requests.post(url, headers=headers, json=payload, timeout=90)
		resp.raise_for_status()
		data = resp.json()
		# Expect {files: {path: content, ...}, instructions?: str}
		files = data.get("files") or {}
		return {"files": files, "provider": "v0", "model": data.get("model", "free"), "notes": data.get("instructions", "")}

	def _available(self, *names):
		return [n for n in names if self.providers.get(n)]

	def classify_image(self, image: Image.Image) -> Dict[str, Any]:
		# Prefer HF Inference API for a ready image classifier if key available
		if self.providers.get("hf"):
			try:
				label, score = self._hf_classify(image)
				return {"label": label, "confidence": float(score), "provider": "hf", "model": "vit-base-patch16-224", "fallback": False}
			except Exception:
				pass
		# Fallback to simple baseline heuristic
		pixels = image.resize((32, 32))
		avg = sum(p[0] + p[1] + p[2] for p in pixels.getdata()) / (32 * 32 * 3)
		label = "cow" if avg < 128 else "cat"
		confidence = 0.65
		provider, model = self._pick_provider_for_vision()
		return {"label": label, "confidence": confidence, "provider": provider, "model": model, "fallback": True}

	def generate_text(self, prompt: str, preference: Optional[List[str]] = None) -> Dict[str, Any]:
		"""General text generation with token accounting.

		Provider priority defaults to: gemini → perplexity → hf → mistral → groq → openai → ollama (unless overridden).
		"""
		strategies: List[str] = []
		if preference:
			for name in preference:
				if self.providers.get(name):
					strategies.append(name)
		# Default priority per requirement
		for name in ["gemini", "perplexity", "hf", "mistral", "groq", "openai", "ollama"]:
			if self.providers.get(name) and name not in strategies:
				strategies.append(name)

		for name in strategies:
			try:
				if name == "gemini":
					text, tokens = self._gemini_chat(prompt)
					return {"output": text, "provider": "gemini", "model": "gemini-1.5-flash", "tokens": tokens, "fallback": False}
				if name == "perplexity":
					text, tokens = self._perplexity_chat(prompt)
					return {"output": text, "provider": "perplexity", "model": "sonar-small-chat", "tokens": tokens, "fallback": False}
				if name == "hf":
					text, tokens = self._hf_generate_text(prompt)
					return {"output": text, "provider": "hf", "model": "Qwen2.5-7B-Instruct (inference)", "tokens": tokens, "fallback": False}
				if name == "mistral":
					text, tokens = self._mistral_chat(prompt)
					return {"output": text, "provider": "mistral", "model": "mistral-small-latest", "tokens": tokens, "fallback": False}
				if name == "groq":
					text, tokens = self._groq_chat(prompt)
					return {"output": text, "provider": "groq", "model": "llama-3.1-8b-instant", "tokens": tokens, "fallback": False}
				if name == "openai":
					text, tokens = self._openai_chat(prompt)
					return {"output": text, "provider": "openai", "model": "gpt-4o-mini", "tokens": tokens, "fallback": False}
				if name == "ollama":
					text, tokens = self._ollama_chat(prompt)
					return {"output": text, "provider": "ollama", "model": "codellama", "tokens": tokens, "fallback": False}
			except Exception:
				continue

		provider, model = self._pick_provider_for_codegen()
		output = f"[baseline] You asked: {prompt}"
		return {"output": output, "provider": provider, "model": model, "tokens": len(output) // 4, "fallback": True}

	def generate_code(self, instruction: str) -> Dict[str, Any]:
		"""Ask providers (in required priority) to return a JSON object mapping file paths to contents.

		The response is expected to be JSON. If not, we'll try to extract a JSON code block.
		"""
		prompt = (
			"Return ONLY a JSON object where keys are relative file paths and values are file contents. "
			"Do not include explanations. "
			f"Instruction: {instruction}"
		)
		result = self.generate_text(prompt, preference=["gemini", "perplexity", "hf", "mistral", "groq", "openai"])  # priority enforced
		text = result.get("output", "{}")
		files: Dict[str, str] = {}
		try:
			files = self._parse_files_json(text)
		except Exception:
			# attempt to find JSON in code fences
			import re, json as _json
			m = re.search(r"```(?:json)?\n([\s\S]*?)```", text)
			if m:
				files = _json.loads(m.group(1))
		result["files"] = files
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

	@retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(3))
	def _hf_classify(self, image: Image.Image):
		api_key = self._get_hf_key()
		headers = {"Authorization": f"Bearer {api_key}"}
		# Use a general image classification model
		url = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
		buf = io.BytesIO()
		image.save(buf, format="PNG")
		resp = requests.post(url, headers=headers, data=buf.getvalue(), timeout=60)
		resp.raise_for_status()
		data = resp.json()
		# HF may return nested lists
		preds = data[0] if isinstance(data, list) and data and isinstance(data[0], list) else data
		best = max(preds, key=lambda x: x.get("score", 0))
		return best.get("label", "unknown"), best.get("score", 0.0)

	@retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(3))
	def _hf_generate_text(self, prompt: str) -> Tuple[str, int]:
		api_key = os.getenv("HF_API_KEY")
		headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
		# Use instruct-tuned model for text generation
		url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"
		payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256, "temperature": 0.3}}
		resp = requests.post(url, headers=headers, json=payload, timeout=60)
		resp.raise_for_status()
		data = resp.json()
		text = (
			data[0]["generated_text"]
			if isinstance(data, list) and data and "generated_text" in data[0]
			else data.get("generated_text", str(data))
		)
		return text, len(text) // 4

	@retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(3))
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
		resp = requests.post(url, headers=headers, json=payload, timeout=60)
		resp.raise_for_status()
		data = resp.json()
		text = data["choices"][0]["message"]["content"].strip()
		tokens = int(data.get("usage", {}).get("total_tokens", 0))
		if not tokens:
			tokens = len(text) // 4
		return text, tokens

	@retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(3))
	def _mistral_chat(self, prompt: str) -> Tuple[str, int]:
		api_key = os.getenv("MISTRAL_API_KEY")
		headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
		url = "https://api.mistral.ai/v1/chat/completions"
		payload = {"model": "mistral-small-latest", "messages": [{"role":"user","content": prompt}], "temperature": 0.2}
		resp = requests.post(url, headers=headers, json=payload, timeout=60)
		resp.raise_for_status()
		data = resp.json()
		text = data["choices"][0]["message"]["content"].strip()
		tokens = int(data.get("usage", {}).get("total_tokens", 0)) if isinstance(data.get("usage"), dict) else 0
		if not tokens:
			tokens = len(text) // 4
		return text, tokens

	@retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(3))
	def _groq_chat(self, prompt: str) -> Tuple[str, int]:
		api_key = os.getenv("GROQ_API_KEY")
		headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
		# Groq is OpenAI-compatible endpoint
		url = "https://api.groq.com/openai/v1/chat/completions"
		payload = {"model": "llama-3.1-8b-instant", "messages": [{"role":"user","content": prompt}], "temperature": 0.2}
		resp = requests.post(url, headers=headers, json=payload, timeout=60)
		resp.raise_for_status()
		data = resp.json()
		text = data["choices"][0]["message"]["content"].strip()
		tokens = int(data.get("usage", {}).get("total_tokens", 0)) if isinstance(data.get("usage"), dict) else 0
		if not tokens:
			tokens = len(text) // 4
		return text, tokens

	@retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(3))
	def _gemini_chat(self, prompt: str) -> Tuple[str, int]:
		api_key = os.getenv("GEMINI_API_KEY")
		headers = {"Content-Type": "application/json"}
		url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
		payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}}
		resp = requests.post(url, headers=headers, json=payload, timeout=60)
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

	@retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(3))
	def _perplexity_chat(self, prompt: str) -> Tuple[str, int]:
		api_key = os.getenv("PERPLEXITY_API_KEY")
		headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
		url = "https://api.perplexity.ai/chat/completions"
		payload = {"model": "sonar-small-chat", "messages": [{"role":"user","content": prompt}], "temperature": 0.2}
		resp = requests.post(url, headers=headers, json=payload, timeout=60)
		resp.raise_for_status()
		data = resp.json()
		text = data["choices"][0]["message"]["content"].strip()
		tokens = int(data.get("usage", {}).get("total_tokens", 0)) if isinstance(data.get("usage"), dict) else 0
		if not tokens:
			tokens = len(text) // 4
		return text, tokens

	@retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(3))
	def _ollama_chat(self, prompt: str) -> Tuple[str, int]:
		base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
		url = f"{base}/api/generate"
		payload = {"model": "codellama", "prompt": prompt, "stream": False}
		resp = requests.post(url, json=payload, timeout=60)
		resp.raise_for_status()
		data = resp.json()
		text = data.get("response", "")
		return text, len(text) // 4


