from __future__ import annotations
from typing import List
import os
from .storage import DatabaseService


class RagService:
	"""Optional, graceful RAG. Falls back to list-only if embeddings unavailable."""

	def __init__(self, db: DatabaseService) -> None:
		self.db = db
		self._texts: List[str] = []
		self._has_embeddings = False
		self._init_embeddings()

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

	def index_text(self, text: str) -> None:
		self._texts.append(text)
		if self._has_embeddings:
			vec = self.embedder.encode([text])
			self.index.add(vec.astype(self._np.float32))

	def retrieve(self, query: str, k: int = 3) -> List[str]:
		if self._has_embeddings and self._texts:
			vec = self.embedder.encode([query]).astype(self._np.float32)
			_, idx = self.index.search(vec, k)
			return [self._texts[i] for i in idx[0] if 0 <= i < len(self._texts)]
		# Fallback: return last k items
		return list(self._texts[-k:])


