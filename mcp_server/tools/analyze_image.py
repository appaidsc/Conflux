"""
MCP Tool - analyze_image

Analyze images/diagrams in Confluence pages (placeholder).
"""

from fastmcp import FastMCP

from mcp_server.services import FetchService

router = FastMCP("analyze_image")


@router.tool()
async def analyze_image(
    page_id: str,
    image_index: int = 0,
) -> dict:
    """
    Analyze an image or diagram from a Confluence page.
    
    Note: Full image analysis requires CLIP model integration.
    Currently returns image metadata only.
    
    Args:
        page_id: Page ID containing the image
        image_index: Index of image on page (0-based)
        
    Returns:
        Image metadata and description (when available)
    """
    service = FetchService()
    
    page = await service.get_page(page_id)
    if not page:
        return {"error": f"Page {page_id} not found"}
    
    if not page.get("has_images"):
        return {"error": "Page has no images"}
    
    # TODO: Implement CLIP-based image analysis
    # For now, return placeholder
    return {
        "page_id": page_id,
        "page_title": page["title"],
        "image_index": image_index,
        "status": "Image analysis not yet implemented",
        "hint": "Use search_images to find pages with diagrams",
    }
