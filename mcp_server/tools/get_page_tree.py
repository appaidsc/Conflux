"""
MCP Tool - get_page_tree

Get page hierarchy tree.
"""

from fastmcp import FastMCP

from mcp_server.services import FetchService

router = FastMCP("get_page_tree")


@router.tool()
async def get_page_tree(
    page_id: str,
    max_depth: int = 2,
) -> dict:
    """
    Get the page hierarchy tree starting from a page.
    
    Shows the structure of child pages under the given page.
    
    Args:
        page_id: Root page ID to start tree from
        max_depth: Maximum depth to traverse (1-3, default 2)
        
    Returns:
        Tree structure with page titles and URLs
    """
    service = FetchService()
    
    tree = await service.get_page_tree(
        page_id=page_id,
        max_depth=min(max_depth, 3),
    )
    
    return tree
