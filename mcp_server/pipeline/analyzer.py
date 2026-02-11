"""
Pipeline - Content Analyzer

Analyzes parsed content for metadata: has_code, has_images, etc.
"""

from typing import List, Dict, Any
from dataclasses import dataclass

from mcp_server.pipeline.parser import ParsedContent


@dataclass
class AnalysisResult:
    """Content analysis results."""
    has_code: bool
    code_languages: List[str]
    code_block_count: int
    has_images: bool
    image_count: int
    has_tables: bool
    table_count: int
    heading_structure: List[str]
    internal_links_count: int
    word_count: int


class ContentAnalyzer:
    """Analyzes parsed content for metadata."""
    
    def analyze(self, parsed: ParsedContent) -> AnalysisResult:
        """
        Analyze parsed content and extract metadata.
        
        Args:
            parsed: ParsedContent from parser
            
        Returns:
            AnalysisResult with content metadata
        """
        # Extract unique code languages
        code_languages = list(set(
            block["language"]
            for block in parsed.code_blocks
            if block["language"]
        ))
        
        return AnalysisResult(
            has_code=len(parsed.code_blocks) > 0,
            code_languages=code_languages,
            code_block_count=len(parsed.code_blocks),
            has_images=len(parsed.images) > 0,
            image_count=len(parsed.images),
            has_tables=len(parsed.tables) > 0,
            table_count=len(parsed.tables),
            heading_structure=parsed.headings,
            internal_links_count=len(parsed.internal_links),
            word_count=parsed.word_count,
        )
