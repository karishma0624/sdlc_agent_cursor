from __future__ import annotations
from typing import List
import os
import sqlite3
from datetime import datetime
from .storage import DatabaseService


class RagService:
	"""Optional, graceful RAG. Falls back to list-only if embeddings unavailable."""

	def __init__(self, db: DatabaseService) -> None:
		self.db = db
		self._texts: List[str] = []
		self._has_embeddings = False
		self._init_embeddings()
		self._init_sqlite()

	def _init_embeddings(self) -> None:
		try:
			from sentence_transformers import SentenceTransformer  # type: ignore
			import numpy as np  # type: ignore
			import faiss  # type: ignore
			self._np = np
			self._faiss = faiss
			model_name = os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
			self.embedder = SentenceTransformer(model_name)
			self.index = faiss.IndexFlatIP(384)
			self._has_embeddings = True
		except Exception:
			self._has_embeddings = False

	def _init_sqlite(self) -> None:
		"""Initialize optional SQLite persistence using RAG_DB_URL or rag.db."""
		url = os.getenv("RAG_DB_URL", "sqlite:///rag.db")
		# crude parse for sqlite path
		path = url.replace("sqlite:///", "").strip()
		self._rag_db_path = path if path else "rag.db"
		try:
			self._rag_conn = sqlite3.connect(self._rag_db_path, check_same_thread=False)
			cur = self._rag_conn.cursor()
			cur.execute(
				"CREATE TABLE IF NOT EXISTS rag_entries (id INTEGER PRIMARY KEY, text TEXT NOT NULL, created_at TEXT NOT NULL)"
			)
			self._rag_conn.commit()
		except Exception:
			self._rag_conn = None

	def index_text(self, text: str) -> None:
		self._texts.append(text)
		if self._has_embeddings:
			vec = self.embedder.encode([text])
			self.index.add(vec.astype(self._np.float32))
		# persist to sqlite (append-only)
		try:
			if getattr(self, "_rag_conn", None):
				cur = self._rag_conn.cursor()
				cur.execute("INSERT INTO rag_entries(text, created_at) VALUES (?, ?)", (text, datetime.utcnow().isoformat()))
				self._rag_conn.commit()
		except Exception:
			pass

	def retrieve(self, query: str, k: int = 3) -> List[str]:
		if self._has_embeddings and self._texts:
			vec = self.embedder.encode([query]).astype(self._np.float32)
			_, idx = self.index.search(vec, k)
			return [self._texts[i] for i in idx[0] if 0 <= i < len(self._texts)]
		# Fallback: return last k items from sqlite if available, else memory
		try:
			if getattr(self, "_rag_conn", None):
				cur = self._rag_conn.cursor()
				cur.execute("SELECT text FROM rag_entries ORDER BY id DESC LIMIT ?", (k,))
				rows = cur.fetchall()
				if rows:
					return [r[0] for r in rows]
		except Exception:
			pass
		return list(self._texts[-k:])


