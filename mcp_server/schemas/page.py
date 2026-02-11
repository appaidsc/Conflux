"""
Schemas - Page Models

Pydantic models for Confluence page data.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class PageMetadata(BaseModel):
    """Lightweight page metadata from index."""
    page_id: str
    title: str
    url: str
    space_key: str
    version: int
    updated_at: datetime
    importance: str = "low"
    has_code: bool = False
    labels: List[str] = []


class PageContent(BaseModel):
    """Full page content from live fetch."""
    page_id: str
    title: str
    url: str
    space_key: str
    space_name: str
    markdown: str
    word_count: int
    headings: List[str] = []
    has_code: bool = False
    code_languages: List[str] = []
    has_images: bool = False
    author: str
    updated_at: datetime
    version: int
    breadcrumb: List[Dict[str, str]] = []
    children: Optional[List[Dict[str, str]]] = None


class PageTreeNode(BaseModel):
    """Page hierarchy tree node."""
    id: str
    title: str
    url: str
    children: List["PageTreeNode"] = []


# Allow recursive model
PageTreeNode.model_rebuild()
