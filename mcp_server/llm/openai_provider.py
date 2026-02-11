"""
LLM - OpenAI Provider

OpenAI and Azure OpenAI provider implementation.
"""

from typing import Optional, List, Dict, Any
import httpx

from mcp_server.llm.base_provider import BaseLLMProvider
from mcp_server.config import get_settings


class OpenAIProvider(BaseLLMProvider):
    """OpenAI and Azure OpenAI provider."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.llm.base_url.rstrip("/")
        self.api_key = self.settings.llm.api_key
        self.model = self.settings.llm.model
        self.timeout = self.settings.llm.timeout_ms / 1000
    
    async def complete(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate completion using OpenAI API."""
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, max_tokens, temperature)
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """Generate chat completion using OpenAI API."""
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            return data["choices"][0]["message"]["content"]
    
    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)
