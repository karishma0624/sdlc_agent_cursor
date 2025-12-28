"""
Database models for the SDLC Builder Agent
"""
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class BuildStatus(str, Enum):
    """Status of a build."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class BuildArtifact(BaseModel):
    """An artifact produced by a build."""
    type: str  # e.g., 'frontend', 'backend', 'logs', 'docker', 'docs'
    path: str  # relative path to the artifact
    size: int = 0  # size in bytes
    mime_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Build(BaseModel):
    """A build represents a single generation request."""
    id: str
    prompt: str
    provider: str
    status: BuildStatus = BuildStatus.QUEUED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    artifacts: List[BuildArtifact] = Field(default_factory=list)
    logs: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_log(self, message: str, level: str = "info", **kwargs):
        """Add a log entry to the build."""
        self.logs.append({
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            **kwargs
        })
        self.updated_at = datetime.utcnow()

class User(BaseModel):
    """A user of the system."""
    id: str
    username: str
    email: str
    hashed_password: str
    is_active: bool = True
    is_superuser: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "username": "johndoe",
                "email": "john@example.com",
                "is_active": True,
                "is_superuser": False,
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00"
            }
        }

class Token(BaseModel):
    """Authentication token."""
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    """Token payload."""
    username: Optional[str] = None
    scopes: List[str] = []

class BuildCreate(BaseModel):
    """Schema for creating a new build."""
    prompt: str
    provider: str = "openai"
    options: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Create a task management app with user authentication",
                "provider": "openai",
                "options": {
                    "framework": "react",
                    "backend": "fastapi",
                    "database": "sqlite"
                }
            }
        }

class BuildUpdate(BaseModel):
    """Schema for updating a build."""
    status: Optional[BuildStatus] = None
    artifacts: Optional[List[BuildArtifact]] = None
    metadata: Optional[Dict[str, Any]] = None

class UserCreate(BaseModel):
    """Schema for creating a new user."""
    username: str
    email: str
    password: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "johndoe",
                "email": "john@example.com",
                "password": "securepassword123"
            }
        }

class UserUpdate(BaseModel):
    """Schema for updating a user."""
    email: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None
