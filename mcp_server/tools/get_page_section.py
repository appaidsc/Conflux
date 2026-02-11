"""
MCP Tool - get_page_section

Get a specific section from a page by heading.
"""

from fastmcp import FastMCP

from mcp_server.services import FetchService

router = FastMCP("get_page_section")


@router.tool()
async def get_page_section(
    page_id: str,
    section_heading: str,
) -> dict:
    """
    Get a specific section from a Confluence page by heading.
    
    Retrieves only the content under a specific heading,
    useful for extracting specific topics.
    
    Args:
        page_id: Confluence page ID
        section_heading: Heading text to search for
        
    Returns:
        Section content under the specified heading
    """
    service = FetchService()
    
    section = await service.get_page_section(
        page_id=page_id,
        section_heading=section_heading,
    )
    
    if not section:
        return {"error": f"Section '{section_heading}' not found in page {page_id}"}
    
    return section
