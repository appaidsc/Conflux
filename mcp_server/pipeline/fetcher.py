"""
Pipeline - Content Fetcher

REST API content fetching with parallel requests and rate limiting.
"""

import asyncio
import httpx
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from mcp_server.config import get_settings


@dataclass
class PageContent:
    """Full page content from Confluence."""
    page_id: str
    title: str
    space_key: str
    space_name: str
    html_content: str
    url: str
    version: int
    created_at: datetime
    updated_at: datetime
    author: str
    author_id: str
    parent_id: Optional[str]
    children_ids: List[str]
    ancestors: List[Dict[str, str]]
    labels: List[str]


class ContentFetcher:
    """Fetches full page content from Confluence API."""
    
    def __init__(self, settings=None, concurrency: int = 5):
        self.settings = settings or get_settings()
        self.base_url = self.settings.confluence.base_url.rstrip("/")
        self.auth = (
            self.settings.confluence.username,
            self.settings.confluence.api_token,
        )
        self.concurrency = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)
    
    async def fetch_page(self, page_id: str) -> Optional[PageContent]:
        """
        Fetch a single page with full content and metadata.
        
        Args:
            page_id: Confluence page ID
            
        Returns:
            PageContent or None if not found
        """
        if not page_id.isalnum():
             raise ValueError(f"Invalid page_id: {page_id}. Must be alphanumeric.")

        async with self._semaphore:
            async with httpx.AsyncClient(auth=self.auth, timeout=30.0) as client:
                url = f"{self.base_url}/rest/api/content/{page_id}"
                params = {
                    "expand": "body.storage,ancestors,children.page,metadata.labels,space,version,history"
                }
                
                try:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        return None
                    raise
                
                # Extract children IDs
                children_ids = [
                    child["id"]
                    for child in data.get("children", {}).get("page", {}).get("results", [])
                ]
                
                # Extract ancestors (breadcrumb)
                ancestors = [
                    {"id": a["id"], "title": a["title"]}
                    for a in data.get("ancestors", [])
                ]
                
                # Extract labels
                labels = [
                    label["name"]
                    for label in data.get("metadata", {}).get("labels", {}).get("results", [])
                ]
                
                # Get parent ID (last ancestor)
                parent_id = ancestors[-1]["id"] if ancestors else None
                
                return PageContent(
                    page_id=data["id"],
                    title=data["title"],
                    space_key=data["space"]["key"],
                    space_name=data["space"]["name"],
                    html_content=data["body"]["storage"]["value"],
                    url=f"{self.base_url}{data['_links']['webui']}",
                    version=data["version"]["number"],
                    created_at=datetime.fromisoformat(
                        data["history"]["createdDate"].replace("Z", "+00:00")
                    ),
                    updated_at=datetime.fromisoformat(
                        data["version"]["when"].replace("Z", "+00:00")
                    ),
                    author=data["history"]["createdBy"].get("displayName", "Unknown"),
                    author_id=data["history"]["createdBy"].get("username", "unknown"),
                    parent_id=parent_id,
                    children_ids=children_ids,
                    ancestors=ancestors,
                    labels=labels,
                )
    
    async def get_page_version(self, page_id: str) -> Optional[int]:
        """
        Lightweight version check — only fetches version number.
        
        Used by smart cache to skip full fetch when content is unchanged.
        Much faster than fetch_page() (~50ms vs ~500ms).
        
        Args:
            page_id: Confluence page ID
            
        Returns:
            Version number or None if page not found
        """
        if not page_id.isalnum():
            # Safe to return None for invalid ID in version check
            return None

        async with self._semaphore:
            async with httpx.AsyncClient(auth=self.auth, timeout=10.0) as client:
                url = f"{self.base_url}/rest/api/content/{page_id}"
                params = {"expand": "version"}
                
                try:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    return data["version"]["number"]
                except Exception:
                    return None
    
    async def fetch_pages(self, page_ids: List[str]) -> List[PageContent]:
        """
        Fetch multiple pages in parallel.
        
        Args:
            page_ids: List of page IDs to fetch
            
        Returns:
            List of PageContent (excluding None for not found)
        """
        tasks = [self.fetch_page(pid) for pid in page_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        pages = []
        for result in results:
            if isinstance(result, PageContent):
                pages.append(result)
            elif isinstance(result, Exception):
                # Log error but continue
                pass
        
        return pages
    
    async def fetch_page_with_children(
        self,
        page_id: str,
        max_depth: int = 2,
    ) -> Dict[str, Any]:
        """
        Fetch a page including its child pages recursively.
        
        Args:
            page_id: Root page ID
            max_depth: Maximum depth to traverse
            
        Returns:
            Dict with page and children
        """
        page = await self.fetch_page(page_id)
        if not page:
            return None
        
        result = {
            "page": page,
            "children": [],
        }
        
        if max_depth > 0 and page.children_ids:
            child_pages = await self.fetch_pages(page.children_ids)
            for child in child_pages:
                child_result = await self.fetch_page_with_children(
                    child.page_id, max_depth - 1
                )
                if child_result:
                    result["children"].append(child_result)
        
        return result
