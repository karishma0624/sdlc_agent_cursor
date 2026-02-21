import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import platform
import psutil

class MetricsLogger:
    def __init__(self, run_dir: str, job_id: str, repo_name: str = "unknown"):
        self.run_dir = run_dir
        self.job_id = job_id
        self.repo_name = repo_name
        self.start_time = time.time()
        self.metrics: Dict[str, Any] = {
            "task_id": job_id,
            "repo_name": repo_name,
            "success": False,
            "failure_category": None,
            "iterations": 0,
            "recovery_attempts": 0,
            "execution_time": 0.0,
            "commands_executed": [],
            "tests_total": 0,
            "tests_passed": 0,
            "static_errors_before": 0,
            "static_errors_after": 0,
            "token_usage": {
                "mistral": 0,
                "gemini": 0,
                "v0": 0,
                "total_estimated_cost": 0.0
            },
            "model_used": [],
            "memory_used_mb": 0.0,
            "sandbox_violations": 0,
            "blocked_commands": 0,
            "human_intervention": False,
            "project_size_files": 0,
            "lines_of_code": 0,
            "timestamp": datetime.now().isoformat(),
            "phases": {}
        }
        self._ensure_log_dir()

    def _ensure_log_dir(self):
        log_dir = os.path.join(self.run_dir, "evaluation")
        os.makedirs(log_dir, exist_ok=True)

    def log_phase_start(self, phase: str):
        self.metrics["phases"][phase] = {
            "start_time": time.time(),
            "status": "running"
        }

    def log_phase_end(self, phase: str, success: bool, error: Optional[str] = None):
        if phase in self.metrics["phases"]:
            p = self.metrics["phases"][phase]
            p["end_time"] = time.time()
            p["duration"] = p["end_time"] - p["start_time"]
            p["status"] = "completed" if success else "failed"
            p["error"] = error

    def log_token_usage(self, model: str, tokens: int, cost: float):
        # Normalize model names to keys
        key = "other"
        if "mistral" in model.lower(): key = "mistral"
        elif "gemini" in model.lower(): key = "gemini"
        elif "v0" in model.lower(): key = "v0"
        
        if key not in self.metrics["token_usage"]:
            self.metrics["token_usage"][key] = 0
            
        self.metrics["token_usage"][key] += tokens
        self.metrics["token_usage"]["total_estimated_cost"] += cost
        
        if model not in self.metrics["model_used"]:
            self.metrics["model_used"].append(model)

    def log_failure(self, category: str, error_msg: str):
        self.metrics["success"] = False
        self.metrics["failure_category"] = category
        self.metrics["error_details"] = error_msg

    def log_success(self):
        self.metrics["success"] = True
        self.metrics["failure_category"] = None

    def capture_system_stats(self):
        try:
            process = psutil.Process()
            self.metrics["memory_used_mb"] = process.memory_info().rss / (1024 * 1024)
        except:
            self.metrics["memory_used_mb"] = 0.0

    def capture_project_stats(self, project_root: str):
        file_count = 0
        loc = 0
        for root, dirs, files in os.walk(project_root):
            if "research_logs" in root or "__pycache__" in root or "venv" in root or ".git" in root:
                continue
            for f in files:
                file_count += 1
                try:
                    with open(os.path.join(root, f), "r", encoding="utf-8", errors="ignore") as fp:
                        loc += len(fp.readlines())
                except: pass
        self.metrics["project_size_files"] = file_count
        self.metrics["lines_of_code"] = loc

    def finalize(self):
        self.metrics["execution_time"] = time.time() - self.start_time
        self.capture_system_stats()
        # Project stats should be captured via capture_project_stats externally if needed
        
        # Write to JSON
        path = os.path.join(self.run_dir, "evaluation", "metrics.json")
        try:
             with open(path, "w") as f:
                json.dump(self.metrics, f, indent=2)
             print(f"[MetricsLogger] Metrics saved to: {path}")
        except Exception as e:
             print(f"[MetricsLogger] Failed to save metrics: {e}")
            
        # Append to master log (if exists)
        master_log = os.path.join(os.path.dirname(self.run_dir), "master_metrics.jsonl")
        try:
            with open(master_log, "a") as f:
                f.write(json.dumps(self.metrics) + "\n")
            print(f"[MetricsLogger] Appended to master log: {master_log}")
        except Exception as e:
             print(f"[MetricsLogger] Failed to append master log: {e}")

        # --- Additional Artifacts ---
        try:
            # 1. Token Report
            token_report = {
                "usage": self.metrics.get("token_usage", {}),
                "breakdown_by_model": self.metrics.get("model_used", []),
                "cost_analysis": f"${self.metrics.get('token_usage', {}).get('total_estimated_cost', 0):.4f}"
            }
            with open(os.path.join(self.run_dir, "evaluation", "token_report.json"), "w") as f:
                json.dump(token_report, f, indent=2)

            # 2. Failure Summary
            failure_summary = {
                "success": self.metrics.get("success"),
                "category": self.metrics.get("failure_category"),
                "details": self.metrics.get("error_details"),
                "recovery_attempts": self.metrics.get("recovery_attempts")
            }
            with open(os.path.join(self.run_dir, "evaluation", "failure_summary.json"), "w") as f:
                json.dump(failure_summary, f, indent=2)

            # 3. SQLite DB
            import sqlite3
            db_path = os.path.join(self.run_dir, "evaluation", "metrics.db")
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS runs (id TEXT, success INTEGER, time REAL, cost REAL)''')
            c.execute("INSERT INTO runs VALUES (?, ?, ?, ?)", (
                self.metrics.get("task_id"),
                1 if self.metrics.get("success") else 0,
                self.metrics.get("execution_time"),
                self.metrics.get("token_usage", {}).get("total_estimated_cost", 0)
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[MetricsLogger] Failed to save extra artifacts: {e}")
