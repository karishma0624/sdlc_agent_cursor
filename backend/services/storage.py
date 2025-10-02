from __future__ import annotations
from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import os


DATABASE_URL = os.getenv("PRIMARY_DB_URL", "sqlite:///app.db")
engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


class ImageBlob(Base):
	__tablename__ = "images"
	id = Column(Integer, primary_key=True)
	filename = Column(String, nullable=False)
	content = Column(LargeBinary, nullable=False)
	created_at = Column(DateTime, default=datetime.utcnow)
	predictions = relationship("Prediction", back_populates="image")


class Prediction(Base):
	__tablename__ = "predictions"
	id = Column(Integer, primary_key=True)
	image_id = Column(Integer, ForeignKey("images.id"))
	label = Column(String, nullable=False)
	confidence = Column(Float, nullable=False)
	created_at = Column(DateTime, default=datetime.utcnow)
	image = relationship("ImageBlob", back_populates="predictions")



class LogEntry(Base):
	__tablename__ = "logs"
	id = Column(Integer, primary_key=True)
	stage = Column(String, nullable=False)
	provider = Column(String)
	model = Column(String)
	success = Column(Integer, default=1)
	message = Column(String)
	meta = Column(JSON)
	created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(engine)


class DatabaseService:
	def __init__(self) -> None:
		self._Session = SessionLocal

	def save_image(self, content: bytes, filename: str) -> int:
		session = self._Session()
		try:
			img = ImageBlob(filename=filename, content=content)
			session.add(img)
			session.commit()
			session.refresh(img)
			return img.id
		finally:
			session.close()

	def save_prediction(self, image_id: int, label: str, confidence: float) -> int:
		session = self._Session()
		try:
			p = Prediction(image_id=image_id, label=label, confidence=confidence)
			session.add(p)
			session.commit()
			session.refresh(p)
			return p.id
		finally:
			session.close()

	def save_log(self, stage: str, provider: Optional[str], model: Optional[str], success: bool, message: str, metadata: Optional[Dict[str, Any]]) -> int:
		session = self._Session()
		try:
			entry = LogEntry(stage=stage, provider=provider, model=model, success=1 if success else 0, message=message, meta=metadata)
			session.add(entry)
			session.commit()
			session.refresh(entry)
			return entry.id
		finally:
			session.close()

	def list_logs(self, limit: int = 50):
		session = self._Session()
		try:
			q = session.query(LogEntry).order_by(LogEntry.created_at.desc()).limit(limit)
			return [
				{
					"id": r.id,
					"stage": r.stage,
					"provider": r.provider,
					"model": r.model,
					"success": bool(r.success),
					"message": r.message,
					"metadata": r.meta,
					"created_at": r.created_at.isoformat(),
				}
				for r in q
			]
		finally:
			session.close()


