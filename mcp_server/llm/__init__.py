"""
LLM Module - Provider Abstraction Layer

Supports LM Studio, OpenAI, and Azure OpenAI providers.
"""

from mcp_server.llm.base_provider import BaseLLMProvider
from mcp_server.llm.lm_studio_provider import LMStudioProvider
from mcp_server.llm.openai_provider import OpenAIProvider

__all__ = [
    "BaseLLMProvider",
    "LMStudioProvider",
    "OpenAIProvider",
]


def get_provider(settings=None):
    """Factory function to get configured LLM provider."""
    from mcp_server.config import get_settings
    settings = settings or get_settings()
    
    if settings.llm.provider == "lm_studio":
        return LMStudioProvider(settings)
    elif settings.llm.provider in ("openai", "azure"):
        return OpenAIProvider(settings)
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm.provider}")
