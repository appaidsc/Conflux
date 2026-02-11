"""
MCP Tool - search_pages

Semantic search across Confluence pages.
"""

from fastmcp import FastMCP
from typing import List, Optional

from mcp_server.services import SearchService

router = FastMCP("search_pages")


@router.tool()
async def search_pages(
    query: str,
    limit: int = 10,
    space_filter: Optional[str] = None,
) -> dict:
    """
    Search Confluence pages using semantic search.
    
    Uses hybrid dense+sparse vectors with FlashRank reranking
    for highly accurate results.
    
    Args:
        query: Search query (natural language)
        limit: Maximum results (1-50, default 10)
        space_filter: Optional comma-separated space keys to filter
        
    Returns:
        List of matching pages with title, URL, snippet, and relevance score
    """
    service = SearchService()
    
    spaces = space_filter.split(",") if space_filter else None
    
    results = await service.search(
        query=query,
        limit=min(limit, 50),
        space_filter=spaces,
        rerank=True,
    )
    
    return {
        "results": [
            {
                "page_id": r.page_id,
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet,
                "score": r.score,
                "importance": r.importance,
                "labels": r.labels,
            }
            for r in results
        ],
        "count": len(results),
        "query": query,
    }
