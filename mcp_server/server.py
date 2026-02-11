"""
Confluence MCP Server - Main Entry Point

FastMCP server with STDIO and SSE transport support.
"""

import asyncio
import argparse
from fastmcp import FastMCP

from mcp_server.config import get_settings

# Import tools (will be registered with decorators)
from mcp_server.tools import (
    search_pages,
    get_page,
    ask_question,
    search_code,
    search_images,
    get_related_pages,
    get_page_tree,
    compare_pages,
    analyze_image,
)


def create_app() -> FastMCP:
    """Create and configure the MCP application."""
    settings = get_settings()
    
    mcp = FastMCP(
        name="confluence-mcp",
        description="Semantic search and Q&A over Confluence documentation",
    )
    
    # Register all tools
    mcp.include_router(search_pages.router)
    mcp.include_router(get_page.router)
    mcp.include_router(ask_question.router)
    mcp.include_router(search_code.router)
    mcp.include_router(search_images.router)
    mcp.include_router(get_related_pages.router)
    mcp.include_router(get_page_tree.router)
    mcp.include_router(compare_pages.router)
    mcp.include_router(analyze_image.router)
    
    return mcp


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Confluence MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default=None,
        help="Transport protocol (default: from env)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for SSE transport (default: from env)"
    )
    args = parser.parse_args()
    
    settings = get_settings()
    transport = args.transport or settings.mcp.transport
    port = args.port or settings.mcp.port
    
    mcp = create_app()
    
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse", host=settings.mcp.host, port=port)


if __name__ == "__main__":
    main()
