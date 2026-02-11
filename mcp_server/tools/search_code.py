"""
MCP Tool - search_code

Search for code snippets across Confluence pages.
"""

from fastmcp import FastMCP
from typing import Optional

from mcp_server.services import SearchService

router = FastMCP("search_code")


@router.tool()
async def search_code(
    query: str,
    language: Optional[str] = None,
    limit: int = 10,
) -> dict:
    """
    Search for code snippets in Confluence documentation.
    
    Finds pages containing code blocks matching your query.
    Optionally filter by programming language.
    
    Args:
        query: Search query (code-related)
        language: Filter by language (python, java, javascript, etc.)
        limit: Maximum results (1-20, default 10)
        
    Returns:
        Pages with code matching the query
    """
    service = SearchService()
    
    # Search with code filter
    results = await service.search(
        query=query,
        limit=min(limit, 20),
        include_code_pages=True,
        rerank=True,
    )
    
    # Filter by language if specified
    if language:
        language = language.lower()
        results = [
            r for r in results
            if language in [lang.lower() for lang in (r.code_languages or [])]
        ]
    
    return {
        "results": [
            {
                "page_id": r.page_id,
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet,
                "score": r.score,
                "code_languages": getattr(r, "code_languages", []),
            }
            for r in results
        ],
        "count": len(results),
        "query": query,
        "language_filter": language,
    }
