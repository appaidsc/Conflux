"""
Unit Tests for Pipeline Module
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from mcp_server.pipeline.parser import ContentParser, ParsedContent
from mcp_server.pipeline.analyzer import ContentAnalyzer
from mcp_server.pipeline.importance_scorer import ImportanceScorer
from mcp_server.pipeline.summarizer import Summarizer


class TestContentParser:
    """Tests for ContentParser."""
    
    def test_parse_simple_html(self):
        """Test parsing simple HTML to Markdown."""
        parser = ContentParser()
        html = "<h1>Title</h1><p>Hello world</p>"
        
        result = parser.parse(html)
        
        assert isinstance(result, ParsedContent)
        assert "# Title" in result.markdown
        assert "Hello world" in result.markdown
        assert "Title" in result.headings
    
    def test_parse_code_blocks(self):
        """Test extracting code blocks."""
        parser = ContentParser()
        html = '<pre><code class="language-python">print("hello")</code></pre>'
        
        result = parser.parse(html)
        
        assert len(result.code_blocks) == 1
        assert result.code_blocks[0]["language"] == "python"
    
    def test_parse_headings_hierarchy(self):
        """Test heading hierarchy extraction."""
        parser = ContentParser()
        html = "<h1>Main</h1><h2>Sub1</h2><h2>Sub2</h2><h3>SubSub</h3>"
        
        result = parser.parse(html)
        
        assert len(result.heading_hierarchy) == 4
        assert result.heading_hierarchy[0]["level"] == 1
        assert result.heading_hierarchy[1]["level"] == 2


class TestContentAnalyzer:
    """Tests for ContentAnalyzer."""
    
    def test_analyze_page_with_code(self):
        """Test analyzing page with code blocks."""
        analyzer = ContentAnalyzer()
        parsed = ParsedContent(
            markdown="# Test\n\nSome content",
            headings=["Test"],
            heading_hierarchy=[{"level": 1, "text": "Test", "position": 0}],
            code_blocks=[{"language": "python", "code": "x=1", "position": 0}],
            images=[],
            tables=[],
            internal_links=[],
            word_count=10,
        )
        
        result = analyzer.analyze(parsed)
        
        assert result.has_code is True
        assert "python" in result.code_languages


class TestImportanceScorer:
    """Tests for ImportanceScorer."""
    
    def test_high_importance_hub_page(self):
        """Test high importance for hub pages."""
        scorer = ImportanceScorer()
        
        from mcp_server.pipeline.analyzer import AnalysisResult
        analysis = AnalysisResult(
            has_code=True,
            code_languages=["python"],
            code_block_count=3,
            has_images=True,
            image_count=2,
            has_tables=False,
            table_count=0,
            heading_structure=["Overview", "Setup"],
            internal_links_count=10,
            word_count=1000,
        )
        
        result = scorer.score(
            children_count=15,
            depth=0,
            labels=["overview", "guide"],
            analysis=analysis,
        )
        
        assert result.classification == "high"
        assert result.score >= 2.0


class TestSummarizer:
    """Tests for Summarizer."""
    
    def test_extractive_summary(self):
        """Test extractive summary generation."""
        summarizer = Summarizer()
        text = "This is a test. There are many sentences here. Some are important."
        
        result = summarizer.extractive_summary(text, num_sentences=2)
        
        assert len(result) > 0
        assert len(result) <= len(text)
