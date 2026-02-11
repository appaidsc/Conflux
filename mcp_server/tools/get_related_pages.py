"""
MCP Tool - get_related_pages

Find pages related to a given page.
"""

from fastmcp import FastMCP
from typing import List

from mcp_server.services import SearchService, FetchService

router = FastMCP("get_related_pages")


@router.tool()
async def get_related_pages(
    page_id: str,
    limit: int = 5,
) -> dict:
    """
    Find pages related to a given page.
    
    Uses semantic similarity to find topically related documentation.
    
    Args:
        page_id: Source page ID
        limit: Maximum related pages (1-10, default 5)
        
    Returns:
        List of related pages with similarity scores
    """
    fetch_service = FetchService()
    search_service = SearchService()
    
    # Get the source page
    page = await fetch_service.get_page(page_id)
    if not page:
        return {"error": f"Page {page_id} not found"}
    
    # Use page title + first part of content for similarity search
    query = f"{page['title']} {page['markdown'][:500]}"
    
    # Search for similar pages
    results = await search_service.search(
        query=query,
        limit=min(limit, 10) + 1,  # Extra to exclude self
        rerank=True,
    )
    
    # Filter out the source page
    related = [r for r in results if r.page_id != page_id][:limit]
    
    return {
        "source_page": {
            "page_id": page_id,
            "title": page["title"],
        },
        "related": [
            {
                "page_id": r.page_id,
                "title": r.title,
                "url": r.url,
                "similarity": r.score,
                "snippet": r.snippet,
            }
            for r in related
        ],
        "count": len(related),
    }
