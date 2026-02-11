"""
LLM - LM Studio Provider

Local LLM provider using LM Studio's OpenAI-compatible API.
"""

from typing import Optional, List, Dict, Any
import httpx

from mcp_server.llm.base_provider import BaseLLMProvider
from mcp_server.config import get_settings


class LMStudioProvider(BaseLLMProvider):
    """LM Studio local LLM provider."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.llm.base_url.rstrip("/")
        self.model = self.settings.llm.model
        self.timeout = self.settings.llm.timeout_ms / 1000
    
    async def complete(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate completion using LM Studio."""
        # Use chat completion with single message
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, max_tokens, temperature)
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """Generate chat completion using LM Studio."""
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            return data["choices"][0]["message"]["content"]
    
    def is_available(self) -> bool:
        """Check if LM Studio is running."""
        import httpx
        
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{self.base_url}/models")
                return response.status_code == 200
        except Exception:
            return False
