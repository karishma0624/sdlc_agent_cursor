"""
Configuration management for the SDLC Builder Agent
"""
import os
from typing import Dict, Optional, Any
from pydantic import BaseSettings, validator, HttpUrl
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "SDLC Builder Agent"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # API settings
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours
    
    # CORS settings
    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
    ]
    
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./sdlc_agent.db")
    
    # Provider API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    MISTRAL_API_KEY: Optional[str] = os.getenv("MISTRAL_API_KEY")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    PERPLEXITY_API_KEY: Optional[str] = os.getenv("PERPLEXITY_API_KEY")
    
    # Provider configuration
    DEFAULT_PROVIDER: str = os.getenv("DEFAULT_PROVIDER", "openai")
    
    # File storage
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", "10485760"))  # 10MB
    
    # RAG Settings
    RAG_DISTANCE_THRESHOLD: float = float(os.getenv("RAG_DISTANCE_THRESHOLD", "0.25"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

# Global settings instance
settings = Settings()

def get_provider_api_key(provider: str) -> Optional[str]:
    """Get the API key for a specific provider."""
    provider_key = f"{provider.upper()}_API_KEY"
    return getattr(settings, provider_key, None)

def validate_provider_api_key(provider: str, api_key: Optional[str] = None) -> bool:
    """Validate if a provider's API key is set and valid."""
    if api_key is None:
        api_key = get_provider_api_key(provider)
    return api_key is not None and len(api_key.strip()) > 0

def get_available_providers() -> Dict[str, Dict[str, Any]]:
    """Get a list of available providers and their status."""
    providers = {
        "openai": {
            "name": "OpenAI",
            "enabled": validate_provider_api_key("openai"),
            "requires_key": True,
            "supports_streaming": True,
        },
        "gemini": {
            "name": "Google Gemini",
            "enabled": validate_provider_api_key("gemini"),
            "requires_key": True,
            "supports_streaming": True,
        },
        "mistral": {
            "name": "Mistral AI",
            "enabled": validate_provider_api_key("mistral"),
            "requires_key": True,
            "supports_streaming": True,
        },
        "groq": {
            "name": "Groq",
            "enabled": validate_provider_api_key("groq"),
            "requires_key": True,
            "supports_streaming": True,
        },
        "perplexity": {
            "name": "Perplexity",
            "enabled": validate_provider_api_key("perplexity"),
            "requires_key": True,
            "supports_streaming": True,
        },
        "ollama": {
            "name": "Ollama",
            "enabled": True,  # Local, no API key required
            "requires_key": False,
            "supports_streaming": True,
        },
        "mock": {
            "name": "Mock (Testing)",
            "enabled": True,  # Always enabled for testing
            "requires_key": False,
            "supports_streaming": False,
        },
    }
    
    return providers
