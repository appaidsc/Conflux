"""
MCP Tool - compare_pages

Compare two Confluence pages side by side.
"""

from fastmcp import FastMCP

from mcp_server.services import FetchService

router = FastMCP("compare_pages")


@router.tool()
async def compare_pages(
    page_id_1: str,
    page_id_2: str,
) -> dict:
    """
    Compare two Confluence pages side by side.
    
    Shows differences in structure, content areas, and metadata.
    
    Args:
        page_id_1: First page ID
        page_id_2: Second page ID
        
    Returns:
        Comparison of the two pages including common/unique sections
    """
    service = FetchService()
    
    comparison = await service.compare_pages(
        page_id_1=page_id_1,
        page_id_2=page_id_2,
    )
    
    return comparison
