"""
MCP Tool - search_images

Search for diagrams and images in Confluence.
"""

from fastmcp import FastMCP

from mcp_server.services import SearchService

router = FastMCP("search_images")


@router.tool()
async def search_images(
    query: str,
    limit: int = 10,
) -> dict:
    """
    Search for pages containing diagrams and images.
    
    Useful for finding architecture diagrams, flowcharts,
    and visual documentation.
    
    Args:
        query: Search query (what the image is about)
        limit: Maximum results (1-20, default 10)
        
    Returns:
        Pages with images matching the query
    """
    service = SearchService()
    
    # Search for pages with images
    results = await service.search(
        query=query,
        limit=min(limit, 20) * 2,  # Over-fetch to filter
        rerank=True,
    )
    
    # Filter to pages with images
    image_results = [r for r in results if r.has_images][:limit]
    
    return {
        "results": [
            {
                "page_id": r.page_id,
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet,
                "score": r.score,
            }
            for r in image_results
        ],
        "count": len(image_results),
        "query": query,
    }
