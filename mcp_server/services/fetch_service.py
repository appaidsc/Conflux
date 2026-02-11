"""
Services - Fetch Service

Live content fetching with caching and formatting.
"""

from typing import Optional, Dict, Any, List
import asyncio

from mcp_server.config import get_settings
from mcp_server.pipeline.fetcher import ContentFetcher, PageContent
from mcp_server.pipeline.parser import ContentParser, ParsedContent
from mcp_server.services.cache_service import CacheService


class FetchService:
    """Fetches and formats page content with caching."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.fetcher = ContentFetcher(settings)
        self.parser = ContentParser(self.settings.confluence.base_url)
        self.cache = CacheService(settings)
    
    async def get_page(
        self,
        page_id: str,
        include_children: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Get full page content with smart version-check caching.
        
        Flow (Stage 2):
        1. Quick version check (~50ms lightweight API call)
        2. If version matches cache → return cached (0ms)
        3. If version changed → full fetch + parse + cache
        
        Args:
            page_id: Confluence page ID
            include_children: Include child page list
            
        Returns:
            Formatted page dict or None
        """
        cache_key = f"page:{page_id}:{include_children}"
        
        # Step 1: Quick version check (lightweight API call)
        current_version = await self.fetcher.get_page_version(page_id)
        if current_version is None:
            return None  # Page not found
        
        # Step 2: Check versioned cache
        cached = self.cache.get_versioned(cache_key, current_version)
        if cached:
            return cached  # Version match — instant return!
        
        # Step 3: Version changed or cache miss — full fetch
        page = await self.fetcher.fetch_page(page_id)
        if not page:
            return None
        
        # Parse content
        parsed = self.parser.parse(page.html_content)
        
        result = {
            "page_id": page.page_id,
            "title": page.title,
            "url": page.url,
            "space_key": page.space_key,
            "space_name": page.space_name,
            "markdown": parsed.markdown,
            "word_count": parsed.word_count,
            "headings": parsed.headings,
            "has_code": len(parsed.code_blocks) > 0,
            "code_languages": list(set(
                b["language"] for b in parsed.code_blocks if b["language"]
            )),
            "has_images": len(parsed.images) > 0,
            "author": page.author,
            "updated_at": page.updated_at.isoformat(),
            "version": page.version,
            "breadcrumb": page.ancestors,
        }
        
        if include_children:
            result["children"] = [
                {"id": cid} for cid in page.children_ids
            ]
        
        # Cache with version tag
        self.cache.set_versioned(cache_key, result, current_version)
        
        return result
    
    async def get_page_section(
        self,
        page_id: str,
        section_heading: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific section from a page by heading.
        
        Args:
            page_id: Confluence page ID
            section_heading: Heading text to search for
            
        Returns:
            Section content dict or None
        """
        page = await self.fetcher.fetch_page(page_id)
        if not page:
            return None
        
        parsed = self.parser.parse(page.html_content)
        
        # Find section by heading
        lines = parsed.markdown.split("\n")
        section_lines = []
        in_section = False
        section_level = 0
        
        for line in lines:
            # Check for heading
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                heading_text = line.lstrip("#").strip()
                
                if heading_text.lower() == section_heading.lower():
                    in_section = True
                    section_level = level
                    section_lines.append(line)
                elif in_section and level <= section_level:
                    # Exiting section
                    break
                elif in_section:
                    section_lines.append(line)
            elif in_section:
                section_lines.append(line)
        
        if not section_lines:
            return None
        
        return {
            "page_id": page_id,
            "page_title": page.title,
            "section_heading": section_heading,
            "content": "\n".join(section_lines),
            "url": f"{page.url}#{section_heading.lower().replace(' ', '-')}",
        }
    
    async def get_page_tree(
        self,
        page_id: str,
        max_depth: int = 2,
    ) -> Dict[str, Any]:
        """
        Get page hierarchy tree.
        
        Args:
            page_id: Root page ID
            max_depth: Maximum depth to traverse
            
        Returns:
            Tree structure dict
        """
        raw_tree = await self.fetcher.fetch_page_with_children(page_id, max_depth)
        
        if not raw_tree:
             return {"id": page_id, "title": "Not Found", "children": []}

        def map_node(node: Dict[str, Any]) -> Dict[str, Any]:
            page = node["page"]
            return {
                "id": page.page_id,
                "title": page.title,
                "url": page.url,
                "children": [map_node(child) for child in node.get("children", [])]
            }

        return map_node(raw_tree)
    
    async def compare_pages(
        self,
        page_id_1: str,
        page_id_2: str,
    ) -> Dict[str, Any]:
        """
        Compare two pages side by side.
        
        Args:
            page_id_1: First page ID
            page_id_2: Second page ID
            
        Returns:
            Comparison dict
        """
        page1, page2 = await asyncio.gather(
            self.get_page(page_id_1),
            self.get_page(page_id_2),
        )
        
        if not page1 or not page2:
            return {"error": "One or both pages not found"}
        
        return {
            "page_1": {
                "title": page1["title"],
                "url": page1["url"],
                "word_count": page1["word_count"],
                "headings": page1["headings"],
                "has_code": page1["has_code"],
            },
            "page_2": {
                "title": page2["title"],
                "url": page2["url"],
                "word_count": page2["word_count"],
                "headings": page2["headings"],
                "has_code": page2["has_code"],
            },
            "common_headings": list(set(page1["headings"]) & set(page2["headings"])),
            "unique_to_1": list(set(page1["headings"]) - set(page2["headings"])),
            "unique_to_2": list(set(page2["headings"]) - set(page1["headings"])),
        }
