"""
MCP Tool - get_page

Retrieve full page content by ID.
"""

from fastmcp import FastMCP

from mcp_server.services import FetchService

router = FastMCP("get_page")


@router.tool()
async def get_page(
    page_id: str,
    include_children: bool = False,
) -> dict:
    """
    Get full content of a Confluence page.
    
    Fetches live content from Confluence (always fresh).
    Returns Markdown-formatted content.
    
    Args:
        page_id: Confluence page ID
        include_children: Include list of child pages
        
    Returns:
        Page content with title, markdown, metadata, and optionally children
    """
    service = FetchService()
    
    page = await service.get_page(
        page_id=page_id,
        include_children=include_children,
    )
    
    if not page:
        return {"error": f"Page {page_id} not found"}
    
    return page
