"""
Schemas - Search Models

Pydantic models for search queries and results.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class SearchQuery(BaseModel):
    """Search query input."""
    query: str
    limit: int = Field(default=10, ge=1, le=50)
    space_filter: Optional[List[str]] = None
    include_code_pages: bool = False


class SearchResult(BaseModel):
    """Single search result."""
    page_id: str
    title: str
    url: str
    snippet: str
    score: float = Field(ge=0.0, le=1.0)
    importance: str = "low"
    has_code: bool = False
    labels: List[str] = []
    space_key: str


class SearchResponse(BaseModel):
    """Full search response."""
    results: List[SearchResult]
    total_count: int
    query: str
    latency_ms: int = 0
