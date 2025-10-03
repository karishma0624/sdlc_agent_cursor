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


class Student(Base):
	__tablename__ = "students"
	id = Column(Integer, primary_key=True)
	name = Column(String, nullable=False)
	email = Column(String, nullable=True)
	created_at = Column(DateTime, default=datetime.utcnow)


class Event(Base):
	__tablename__ = "events"
	id = Column(Integer, primary_key=True)
	title = Column(String, nullable=False)
	description = Column(String, nullable=True)
	date = Column(String, nullable=True)
	created_at = Column(DateTime, default=datetime.utcnow)


class RequirementDoc(Base):
	__tablename__ = "requirements"
	id = Column(Integer, primary_key=True)
	title = Column(String, nullable=False)
	content = Column(JSON, nullable=False)
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

	# Students CRUD
	def create_student(self, name: str, email: Optional[str]) -> int:
		session = self._Session()
		try:
			obj = Student(name=name, email=email)
			session.add(obj)
			session.commit()
			session.refresh(obj)
			return obj.id
		finally:
			session.close()

	def get_student(self, student_id: int) -> Optional[dict]:
		session = self._Session()
		try:
			obj = session.get(Student, student_id)
			if not obj:
				return None
			return {"id": obj.id, "name": obj.name, "email": obj.email, "created_at": obj.created_at.isoformat()}
		finally:
			session.close()

	def list_students(self, limit: int = 100) -> list[dict]:
		session = self._Session()
		try:
			rows = session.query(Student).order_by(Student.created_at.desc()).limit(limit)
			return [{"id": r.id, "name": r.name, "email": r.email, "created_at": r.created_at.isoformat()} for r in rows]
		finally:
			session.close()

	def update_student(self, student_id: int, name: Optional[str], email: Optional[str]) -> bool:
		session = self._Session()
		try:
			obj = session.get(Student, student_id)
			if not obj:
				return False
			if name is not None:
				obj.name = name
			if email is not None:
				obj.email = email
			session.commit()
			return True
		finally:
			session.close()

	def delete_student(self, student_id: int) -> bool:
		session = self._Session()
		try:
			obj = session.get(Student, student_id)
			if not obj:
				return False
			session.delete(obj)
			session.commit()
			return True
		finally:
			session.close()

	# Events CRUD
	def create_event(self, title: str, description: Optional[str], date: Optional[str]) -> int:
		session = self._Session()
		try:
			obj = Event(title=title, description=description, date=date)
			session.add(obj)
			session.commit()
			session.refresh(obj)
			return obj.id
		finally:
			session.close()

	def get_event(self, event_id: int) -> Optional[dict]:
		session = self._Session()
		try:
			obj = session.get(Event, event_id)
			if not obj:
				return None
			return {"id": obj.id, "title": obj.title, "description": obj.description, "date": obj.date, "created_at": obj.created_at.isoformat()}
		finally:
			session.close()

	def list_events(self, limit: int = 100) -> list[dict]:
		session = self._Session()
		try:
			rows = session.query(Event).order_by(Event.created_at.desc()).limit(limit)
			return [{"id": r.id, "title": r.title, "description": r.description, "date": r.date, "created_at": r.created_at.isoformat()} for r in rows]
		finally:
			session.close()

	def update_event(self, event_id: int, title: Optional[str], description: Optional[str], date: Optional[str]) -> bool:
		session = self._Session()
		try:
			obj = session.get(Event, event_id)
			if not obj:
				return False
			if title is not None:
				obj.title = title
			if description is not None:
				obj.description = description
			if date is not None:
				obj.date = date
			session.commit()
			return True
		finally:
			session.close()

	def delete_event(self, event_id: int) -> bool:
		session = self._Session()
		try:
			obj = session.get(Event, event_id)
			if not obj:
				return False
			session.delete(obj)
			session.commit()
			return True
		finally:
			session.close()

	# Requirements storage
	def create_requirement(self, title: str, content: Dict[str, Any]) -> int:
		session = self._Session()
		try:
			obj = RequirementDoc(title=title, content=content)
			session.add(obj)
			session.commit()
			session.refresh(obj)
			return obj.id
		finally:
			session.close()

	def get_requirement(self, req_id: int) -> Optional[dict]:
		session = self._Session()
		try:
			obj = session.get(RequirementDoc, req_id)
			if not obj:
				return None
			return {"id": obj.id, "title": obj.title, "content": obj.content, "created_at": obj.created_at.isoformat()}
		finally:
			session.close()

	def list_requirements(self, limit: int = 100) -> list[dict]:
		session = self._Session()
		try:
			rows = session.query(RequirementDoc).order_by(RequirementDoc.created_at.desc()).limit(limit)
			return [{"id": r.id, "title": r.title, "content": r.content, "created_at": r.created_at.isoformat()} for r in rows]
		finally:
			session.close()

	def update_requirement(self, req_id: int, title: Optional[str], content: Optional[Dict[str, Any]]) -> bool:
		session = self._Session()
		try:
			obj = session.get(RequirementDoc, req_id)
			if not obj:
				return False
			if title is not None:
				obj.title = title
			if content is not None:
				obj.content = content
			session.commit()
			return True
		finally:
			session.close()

	def delete_requirement(self, req_id: int) -> bool:
		session = self._Session()
		try:
			obj = session.get(RequirementDoc, req_id)
			if not obj:
				return False
			session.delete(obj)
			session.commit()
			return True
		finally:
			session.close()


