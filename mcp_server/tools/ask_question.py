"""
MCP Tool - ask_question

RAG-based question answering over Confluence.
"""

from fastmcp import FastMCP
from typing import Literal

from mcp_server.services import RAGService
from mcp_server.llm import get_provider

router = FastMCP("ask_question")


@router.tool()
async def ask_question(
    question: str,
    mode: str = "hybrid",
    verify_grounding: bool = True,
) -> dict:
    """
    Answer questions using Confluence documentation.
    
    Uses RAG (Retrieval-Augmented Generation) to find relevant
    pages and generate accurate answers with citations.
    
    Args:
        question: Natural language question
        mode: Answer mode - "extractive" (fast), "ai" (LLM), or "hybrid" (default)
        verify_grounding: Check if answer is grounded in sources
        
    Returns:
        Answer with sources, confidence score, and grounding validation
    """
    try:
        llm = get_provider()
    except Exception:
        llm = None
    
    service = RAGService(llm_provider=llm)
    
    # Validate mode
    valid_modes = ("extractive", "ai", "hybrid")
    if mode not in valid_modes:
        mode = "hybrid"
    
    result = await service.ask_question(
        question=question,
        mode=mode,
        verify_grounding=verify_grounding and llm is not None,
    )
    
    return {
        "answer": result.answer,
        "sources": [
            {
                "page_id": s.page_id,
                "title": s.title,
                "url": s.url,
                "relevance": s.relevance,
            }
            for s in result.sources
        ],
        "confidence": result.confidence,
        "mode_used": result.mode_used,
        "grounding_score": result.grounding_score,
    }
