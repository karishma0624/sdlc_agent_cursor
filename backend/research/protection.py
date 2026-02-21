import os
import json
import time
from typing import Dict, Any, Optional

class ProtectionManager:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.daily_limit = 1000000  # Example: 1M tokens
        self.session_limit = 50000  # Example: 50k tokens
        self.cache_dir = os.path.join(os.path.dirname(run_dir), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.usage_file = os.path.join(os.path.dirname(run_dir), "token_usage.json")
        self.current_usage = self._load_usage()
        
    def _load_usage(self) -> Dict[str, Any]:
        if os.path.exists(self.usage_file):
            try:
                with open(self.usage_file, "r") as f:
                    return json.load(f)
            except: pass
        return {"daily_total": 0, "last_reset": time.time()}

    def check_allowance(self, model: str, estimated_tokens: int) -> bool:
        # Reset daily
        if time.time() - self.current_usage.get("last_reset", 0) > 86400:
            self.current_usage["daily_total"] = 0
            self.current_usage["last_reset"] = time.time()
            
        current = self.current_usage["daily_total"]
        if current + estimated_tokens > self.daily_limit:
            return False
        return True

    def log_usage(self, model: str, tokens: int, cost: float):
        self.current_usage["daily_total"] += tokens
        self.current_usage["last_updated"] = time.time()
        
        # Save
        with open(self.usage_file, "w") as f:
            json.dump(self.current_usage, f)

    def get_cached_response(self, prompt: str, model: str) -> Optional[str]:
        # Simple hash based cache
        import hashlib
        h = hashlib.md5((prompt + model).encode()).hexdigest()
        path = os.path.join(self.cache_dir, h + ".json")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                return data.get("response")
        return None

    def cache_response(self, prompt: str, model: str, response: str):
        import hashlib
        h = hashlib.md5((prompt + model).encode()).hexdigest()
        path = os.path.join(self.cache_dir, h + ".json")
        with open(path, "w") as f:
            json.dump({"prompt": prompt, "model": model, "response": response}, f)
