"""
Provider factory with API key management for SDLC Builder Agent
"""
from typing import Dict, Type, Optional, Any
import logging
from ..config import settings, validate_provider_api_key, get_provider_api_key
from .base import Provider, ProviderFactory, MockProvider

logger = logging.getLogger(__name__)

# Import all provider implementations
try:
    from .openai_provider import OpenAIProvider
except ImportError:
    logger.warning("OpenAI provider not available. Install with: pip install openai")
    OpenAIProvider = None

try:
    from .gemini_provider import GeminiProvider
except ImportError:
    logger.warning("Gemini provider not available. Install with: pip install google-generativeai")
    GeminiProvider = None

try:
    from .mistral_provider import MistralProvider
except ImportError:
    logger.warning("Mistral provider not available. Install with: pip install mistralai")
    MistralProvider = None

try:
    from .groq_provider import GroqProvider
except ImportError:
    logger.warning("Groq provider not available. Install with: pip install groq")
    GroqProvider = None

try:
    from .perplexity_provider import PerplexityProvider
except ImportError:
    logger.warning("Perplexity provider not available. Install with: pip install perplexity-ai")
    PerplexityProvider = None

class ProviderManager:
    """Manager for AI providers with API key handling."""
    
    def __init__(self):
        self._providers: Dict[str, Type[Provider]] = {}
        self._provider_instances: Dict[str, Provider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Register all available providers."""
        # Always register the mock provider for testing
        self.register_provider("mock", MockProvider)
        
        # Register other providers if available
        if OpenAIProvider:
            self.register_provider("openai", OpenAIProvider)
        if GeminiProvider:
            self.register_provider("gemini", GeminiProvider)
        if MistralProvider:
            self.register_provider("mistral", MistralProvider)
        if GroqProvider:
            self.register_provider("groq", GroqProvider)
        if PerplexityProvider:
            self.register_provider("perplexity", PerplexityProvider)
    
    def register_provider(self, name: str, provider_class: Type[Provider]) -> None:
        """Register a provider class."""
        self._providers[name.lower()] = provider_class
        logger.info(f"Registered provider: {name}")
    
    def get_provider(self, name: str, api_key: Optional[str] = None, **kwargs) -> Provider:
        """
        Get a provider instance with API key handling.
        
        Args:
            name: Name of the provider
            api_key: Optional API key (if not provided, will use from settings)
            **kwargs: Additional arguments to pass to the provider
            
        Returns:
            An instance of the requested provider
            
        Raises:
            ValueError: If the provider is not found or not configured
        """
        name = name.lower()
        
        # Check if provider exists
        if name not in self._providers:
            raise ValueError(f"Provider '{name}' not found. Available providers: {list(self._providers.keys())}")
        
        # Use cached instance if available
        cache_key = f"{name}:{api_key or 'default'}"
        if cache_key in self._provider_instances:
            return self._provider_instances[cache_key]
        
        # Get provider class and configuration
        provider_class = self._providers[name]
        
        # Handle API key
        if name != "mock":  # Mock provider doesn't need an API key
            if not api_key:
                api_key = get_provider_api_key(name)
            
            if not api_key and provider_class.requires_key:
                raise ValueError(f"API key required for provider: {name}")
            
            kwargs["api_key"] = api_key
        
        # Create and cache the provider instance
        try:
            provider = provider_class(**kwargs)
            self._provider_instances[cache_key] = provider
            return provider
        except Exception as e:
            logger.error(f"Failed to initialize provider {name}: {str(e)}")
            raise ValueError(f"Failed to initialize provider {name}: {str(e)}")
    
    def get_available_providers(self) -> Dict[str, Dict[str, Any]]:
        """Get a list of available providers and their status."""
        providers = {}
        
        for name, provider_class in self._providers.items():
            try:
                # Try to get a test instance to check if the provider is working
                test_instance = self.get_provider(name)
                status = {
                    "name": provider_class.name if hasattr(provider_class, "name") else name.capitalize(),
                    "enabled": True,
                    "requires_key": provider_class.requires_key,
                    "supports_streaming": provider_class.supports_streaming,
                    "capabilities": test_instance.get_capabilities() if hasattr(test_instance, "get_capabilities") else []
                }
            except Exception as e:
                logger.warning(f"Provider {name} is not available: {str(e)}")
                status = {
                    "name": name.capitalize(),
                    "enabled": False,
                    "error": str(e),
                    "requires_key": True,
                    "supports_streaming": False
                }
            
            providers[name] = status
        
        return providers

# Global provider manager instance
provider_manager = ProviderManager()

def get_provider(name: str, api_key: Optional[str] = None, **kwargs) -> Provider:
    """Get a provider instance with API key handling."""
    return provider_manager.get_provider(name, api_key, **kwargs)

def get_available_providers() -> Dict[str, Dict[str, Any]]:
    """Get a list of available providers and their status."""
    return provider_manager.get_available_providers()

# Register the provider manager with the base ProviderFactory
ProviderFactory.get_provider = get_provider
