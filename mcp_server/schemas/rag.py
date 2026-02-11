"""
Schemas - RAG Models

Pydantic models for RAG responses and citations.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class Citation(BaseModel):
    """Source citation in answer."""
    page_id: str
    title: str
    url: str
    relevance: float = Field(ge=0.0, le=1.0)


class GroundingResult(BaseModel):
    """Grounding validation result."""
    score: float = Field(ge=0.0, le=1.0)
    valid_claims: int
    total_claims: int
    ungrounded_claims: List[str] = []


class RAGResponse(BaseModel):
    """RAG answer with citations."""
    answer: str
    sources: List[Citation]
    confidence: float = Field(ge=0.0, le=1.0)
    mode_used: str  # "extractive" | "ai" | "hybrid"
    grounding: Optional[GroundingResult] = None
    latency_ms: int = 0
