"""
LLM - Base Provider

Abstract base class for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class BaseLLMProvider(ABC):
    """Base class for LLM provider implementations."""
    
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Generate completion from prompt.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate chat completion.
        
        Args:
            messages: List of {"role": "...", "content": "..."} dicts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Assistant message content
        """
        pass
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings (optional, use fastembed by default).
        
        Args:
            texts: Texts to embed
            
        Returns:
            List of embedding vectors
        """
        raise NotImplementedError("Use fastembed for embeddings")
    
    def is_available(self) -> bool:
        """Check if provider is available."""
        return True
