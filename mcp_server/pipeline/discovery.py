"""
Pipeline - Discovery Module

CQL-based page discovery from Confluence spaces.
"""

import httpx
from typing import List, Optional, AsyncIterator
from dataclasses import dataclass
from datetime import datetime

from mcp_server.config import get_settings


@dataclass
class PageSummary:
    """Lightweight page info from discovery."""
    page_id: str
    title: str
    space_key: str
    version: int
    last_updated: datetime
    url: str


class ConfluenceDiscovery:
    """Discovers pages in Confluence spaces using CQL queries."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.confluence.base_url.rstrip("/")
        self.auth = (
            self.settings.confluence.username,
            self.settings.confluence.api_token,
        )
    
    async def discover_pages(
        self,
        space_key: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[PageSummary]:
        """
        Discover all pages in a space using CQL.
        
        Args:
            space_key: Space to search (default: from settings)
            limit: Maximum pages to return (default: from settings)
            
        Yields:
            PageSummary for each discovered page
        """
        space = space_key or self.settings.confluence.space_key
        max_pages = limit or self.settings.pipeline.max_pages
        
        cql = f"space={space} AND type=page ORDER BY lastModified DESC"
        
        async with httpx.AsyncClient(auth=self.auth, timeout=30.0) as client:
            start = 0
            page_size = 100
            total_returned = 0
            
            while total_returned < max_pages:
                url = f"{self.base_url}/rest/api/content/search"
                params = {
                    "cql": cql,
                    "expand": "version",
                    "limit": min(page_size, max_pages - total_returned),
                    "start": start,
                }
                
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                results = data.get("results", [])
                if not results:
                    break
                
                for page in results:
                    yield PageSummary(
                        page_id=page["id"],
                        title=page["title"],
                        space_key=space,
                        version=page["version"]["number"],
                        last_updated=datetime.fromisoformat(
                            page["version"]["when"].replace("Z", "+00:00")
                        ),
                        url=f"{self.base_url}{page['_links']['webui']}",
                    )
                    total_returned += 1
                    
                    if total_returned >= max_pages:
                        break
                
                start += page_size
                
                # Check if more results exist
                if len(results) < page_size:
                    break
    
    async def get_modified_since(
        self,
        since: datetime,
        space_key: Optional[str] = None,
    ) -> AsyncIterator[PageSummary]:
        """
        Get pages modified since a given timestamp.
        Used for incremental sync.
        """
        space = space_key or self.settings.confluence.space_key
        since_str = since.strftime("%Y-%m-%d %H:%M")
        cql = f"space={space} AND type=page AND lastModified >= '{since_str}'"
        
        async with httpx.AsyncClient(auth=self.auth, timeout=30.0) as client:
            start = 0
            page_size = 100
            
            while True:
                url = f"{self.base_url}/rest/api/content/search"
                params = {"cql": cql, "expand": "version", "limit": page_size, "start": start}
                
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                results = data.get("results", [])
                if not results:
                    break
                
                for page in results:
                    yield PageSummary(
                        page_id=page["id"],
                        title=page["title"],
                        space_key=space,
                        version=page["version"]["number"],
                        last_updated=datetime.fromisoformat(
                            page["version"]["when"].replace("Z", "+00:00")
                        ),
                        url=f"{self.base_url}{page['_links']['webui']}",
                    )
                
                start += page_size
                if len(results) < page_size:
                    break
