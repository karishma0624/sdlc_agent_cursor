from __future__ import annotations
from typing import Any, Dict, Optional
import os
import json
from datetime import datetime
import sqlite3


class MetricsLogger:
    """Writes structured JSON logs and persists audit metrics to SQLite.

    - logs/build.log: append one JSON object per stage
    - logs/errors.log: append error entries
    - logs/audit.sqlite: sqlite db with audit table
    """

    def __init__(self, logs_dir: str = "logs") -> None:
        self.logs_dir = logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)
        self.build_log_path = os.path.join(self.logs_dir, "build.log")
        self.error_log_path = os.path.join(self.logs_dir, "errors.log")
        self.audit_db_path = os.path.join(self.logs_dir, "audit.sqlite")
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        self._conn = sqlite3.connect(self.audit_db_path, check_same_thread=False)
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS audit (
              id INTEGER PRIMARY KEY,
              timestamp TEXT NOT NULL,
              task_name TEXT,
              stage TEXT,
              models_used TEXT,
              tokens_spent INTEGER,
              files_generated INTEGER,
              errors TEXT,
              human_interventions INTEGER,
              success_rate REAL,
              metadata TEXT
            )
            """
        )
        self._conn.commit()

    def log_stage(
        self,
        task_name: str,
        stage: str,
        success: bool,
        models_used: Optional[Dict[str, Any]] = None,
        tokens_spent: Optional[int] = None,
        files_generated: Optional[int] = None,
        errors: Optional[str] = None,
        human_interventions: int = 0,
        success_rate: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "task_name": task_name,
            "stage": stage,
            "success": success,
            "models_used": models_used or {},
            "tokens_spent": tokens_spent,
            "files_generated": files_generated,
            "errors": errors,
            "human_interventions": human_interventions,
            "success_rate": success_rate,
            "metadata": metadata or {},
        }

        with open(self.build_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        if not success or errors:
            with open(self.error_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO audit (timestamp, task_name, stage, models_used, tokens_spent, files_generated, errors, human_interventions, success_rate, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry["timestamp"],
                task_name,
                stage,
                json.dumps(models_used or {}),
                tokens_spent or 0,
                files_generated or 0,
                errors or "",
                human_interventions,
                success_rate if success_rate is not None else (1.0 if success else 0.0),
                json.dumps(metadata or {}),
            ),
        )
        self._conn.commit()


