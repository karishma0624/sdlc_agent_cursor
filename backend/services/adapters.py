from typing import Dict, Any, List
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

	def _detect_providers(self):
		return {
			"openai": bool(os.getenv("OPENAI_API_KEY")),
			"gemini": bool(os.getenv("GEMINI_API_KEY")),
			"mistral": bool(os.getenv("MISTRAL_API_KEY")),
			"groq": bool(os.getenv("GROQ_API_KEY")),
			"hf": bool(os.getenv("HF_API_KEY")),
			"perplexity": bool(os.getenv("PERPLEXITY_API_KEY")),
			"ollama": bool(os.getenv("OLLAMA_BASE_URL")),
		}

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

	def generate_text(self, prompt: str) -> Dict[str, Any]:
		# Try providers in priority order, return first success
		strategies: List[str] = []
		if self.providers.get("openai"): strategies.append("openai")
		if self.providers.get("mistral"): strategies.append("mistral")
		if self.providers.get("groq"): strategies.append("groq")
		if self.providers.get("gemini"): strategies.append("gemini")
		if self.providers.get("perplexity"): strategies.append("perplexity")
		if self.providers.get("hf"): strategies.append("hf")
		if self.providers.get("ollama"): strategies.append("ollama")

		for name in strategies:
			try:
				if name == "openai":
					text = self._openai_chat(prompt)
					return {"output": text, "provider": "openai", "model": "gpt-4o-mini", "fallback": False}
				if name == "mistral":
					text = self._mistral_chat(prompt)
					return {"output": text, "provider": "mistral", "model": "mistral-small-latest", "fallback": False}
				if name == "groq":
					text = self._groq_chat(prompt)
					return {"output": text, "provider": "groq", "model": "llama-3.1-8b-instant", "fallback": False}
				if name == "gemini":
					text = self._gemini_chat(prompt)
					return {"output": text, "provider": "gemini", "model": "gemini-1.5-flash", "fallback": False}
				if name == "perplexity":
					text = self._perplexity_chat(prompt)
					return {"output": text, "provider": "perplexity", "model": "sonar-small-chat", "fallback": False}
				if name == "hf":
					text = self._hf_generate_text(prompt)
					return {"output": text, "provider": "hf", "model": "Qwen2.5-7B-Instruct (inference)", "fallback": False}
				if name == "ollama":
					text = self._ollama_chat(prompt)
					return {"output": text, "provider": "ollama", "model": "codellama", "fallback": False}
			except Exception:
				continue

		# Minimal offline baseline
		provider, model = self._pick_provider_for_codegen()
		output = f"[baseline] You asked: {prompt}"
		return {"output": output, "provider": provider, "model": model, "fallback": True}

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
		api_key = os.getenv("HF_API_KEY")
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
	def _hf_generate_text(self, prompt: str) -> str:
		api_key = os.getenv("HF_API_KEY")
		headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
		# Use instruct-tuned model for text generation
		url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"
		payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256, "temperature": 0.3}}
		resp = requests.post(url, headers=headers, json=payload, timeout=60)
		resp.raise_for_status()
		data = resp.json()
		if isinstance(data, list) and data and "generated_text" in data[0]:
			return data[0]["generated_text"]
		# Some servers return {generated_text: str}
		return data.get("generated_text", str(data))

	@retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(3))
	def _openai_chat(self, prompt: str) -> str:
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
		return data["choices"][0]["message"]["content"].strip()

	@retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(3))
	def _mistral_chat(self, prompt: str) -> str:
		api_key = os.getenv("MISTRAL_API_KEY")
		headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
		url = "https://api.mistral.ai/v1/chat/completions"
		payload = {"model": "mistral-small-latest", "messages": [{"role":"user","content": prompt}], "temperature": 0.2}
		resp = requests.post(url, headers=headers, json=payload, timeout=60)
		resp.raise_for_status()
		data = resp.json()
		return data["choices"][0]["message"]["content"].strip()

	@retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(3))
	def _groq_chat(self, prompt: str) -> str:
		api_key = os.getenv("GROQ_API_KEY")
		headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
		# Groq is OpenAI-compatible endpoint
		url = "https://api.groq.com/openai/v1/chat/completions"
		payload = {"model": "llama-3.1-8b-instant", "messages": [{"role":"user","content": prompt}], "temperature": 0.2}
		resp = requests.post(url, headers=headers, json=payload, timeout=60)
		resp.raise_for_status()
		data = resp.json()
		return data["choices"][0]["message"]["content"].strip()

	@retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(3))
	def _gemini_chat(self, prompt: str) -> str:
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
		return "\n".join(p.get("text", "") for p in parts if isinstance(p, dict))

	@retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(3))
	def _perplexity_chat(self, prompt: str) -> str:
		api_key = os.getenv("PERPLEXITY_API_KEY")
		headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
		url = "https://api.perplexity.ai/chat/completions"
		payload = {"model": "sonar-small-chat", "messages": [{"role":"user","content": prompt}], "temperature": 0.2}
		resp = requests.post(url, headers=headers, json=payload, timeout=60)
		resp.raise_for_status()
		data = resp.json()
		return data["choices"][0]["message"]["content"].strip()

	@retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(3))
	def _ollama_chat(self, prompt: str) -> str:
		base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
		url = f"{base}/api/generate"
		payload = {"model": "codellama", "prompt": prompt, "stream": False}
		resp = requests.post(url, json=payload, timeout=60)
		resp.raise_for_status()
		data = resp.json()
		return data.get("response", "")


