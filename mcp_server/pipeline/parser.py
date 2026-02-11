"""
Pipeline - Content Parser

HTML → Markdown conversion with code block and Confluence macro handling.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from bs4 import BeautifulSoup
from markdownify import markdownify as md


@dataclass
class ParsedContent:
    """Parsed and structured page content."""
    markdown: str
    headings: List[str]
    heading_hierarchy: List[Dict[str, Any]]
    code_blocks: List[Dict[str, Any]]
    images: List[Dict[str, str]]
    tables: List[str]
    internal_links: List[Dict[str, str]]
    word_count: int


class ContentParser:
    """Parses Confluence HTML to structured Markdown."""
    
    def __init__(self, base_url: str = ""):
        self.base_url = base_url
    
    def parse(self, html_content: str) -> ParsedContent:
        """
        Parse Confluence HTML to structured Markdown.
        
        Args:
            html_content: Raw HTML from Confluence storage format
            
        Returns:
            ParsedContent with markdown and extracted structure
        """
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Process Confluence macros before conversion
        self._process_macros(soup)
        
        # Extract structured data before converting
        headings = self._extract_headings(soup)
        heading_hierarchy = self._extract_heading_hierarchy(soup)
        code_blocks = self._extract_code_blocks(soup)
        images = self._extract_images(soup)
        tables = self._extract_tables(soup)
        internal_links = self._extract_internal_links(soup)
        
        # Convert to Markdown
        markdown = md(str(soup), heading_style="ATX", code_language="")
        markdown = self._clean_markdown(markdown)
        
        # Count words
        word_count = len(markdown.split())
        
        return ParsedContent(
            markdown=markdown,
            headings=headings,
            heading_hierarchy=heading_hierarchy,
            code_blocks=code_blocks,
            images=images,
            tables=tables,
            internal_links=internal_links,
            word_count=word_count,
        )
    
    def _process_macros(self, soup: BeautifulSoup) -> None:
        """Process Confluence-specific macros."""
        # Code macro
        for macro in soup.find_all("ac:structured-macro", {"ac:name": "code"}):
            language = ""
            lang_param = macro.find("ac:parameter", {"ac:name": "language"})
            if lang_param:
                language = lang_param.get_text()
            
            body = macro.find("ac:plain-text-body")
            if body:
                code = body.get_text()
                pre = soup.new_tag("pre")
                code_tag = soup.new_tag("code", attrs={"class": f"language-{language}"})
                code_tag.string = code
                pre.append(code_tag)
                macro.replace_with(pre)
        
        # Info/Note/Warning panels
        for panel_type in ["info", "note", "warning", "tip"]:
            for macro in soup.find_all("ac:structured-macro", {"ac:name": panel_type}):
                body = macro.find("ac:rich-text-body")
                if body:
                    blockquote = soup.new_tag("blockquote")
                    blockquote.append(BeautifulSoup(f"<strong>[{panel_type.upper()}]</strong> ", "html.parser"))
                    for child in list(body.children):
                        blockquote.append(child)
                    macro.replace_with(blockquote)
        
        # Expand macro
        for macro in soup.find_all("ac:structured-macro", {"ac:name": "expand"}):
            title = ""
            title_param = macro.find("ac:parameter", {"ac:name": "title"})
            if title_param:
                title = title_param.get_text()
            
            body = macro.find("ac:rich-text-body")
            if body:
                details = soup.new_tag("details")
                summary = soup.new_tag("summary")
                summary.string = title or "Click to expand"
                details.append(summary)
                for child in list(body.children):
                    details.append(child)
                macro.replace_with(details)
    
    def _extract_headings(self, soup: BeautifulSoup) -> List[str]:
        """Extract all heading text."""
        headings = []
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            headings.append(tag.get_text(strip=True))
        return headings
    
    def _extract_heading_hierarchy(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract headings with level and position."""
        hierarchy = []
        for i, tag in enumerate(soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])):
            level = int(tag.name[1])
            hierarchy.append({
                "level": level,
                "text": tag.get_text(strip=True),
                "position": i,
            })
        return hierarchy
    
    def _extract_code_blocks(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract code blocks with language info."""
        blocks = []
        for i, pre in enumerate(soup.find_all("pre")):
            code = pre.find("code")
            if code:
                lang_class = code.get("class", [])
                language = ""
                for cls in lang_class if isinstance(lang_class, list) else [lang_class]:
                    if cls and cls.startswith("language-"):
                        language = cls.replace("language-", "")
                        break
                
                blocks.append({
                    "language": language,
                    "code": code.get_text(),
                    "position": i,
                })
        return blocks
    
    def _extract_images(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract image URLs and alt text."""
        images = []
        for img in soup.find_all("img"):
            images.append({
                "url": img.get("src", ""),
                "alt": img.get("alt", ""),
            })
        
        # Also check for Confluence image macros
        for macro in soup.find_all("ac:image"):
            attachment = macro.find("ri:attachment")
            if attachment:
                filename = attachment.get("ri:filename", "")
                images.append({
                    "url": f"/attachment/{filename}",
                    "alt": filename,
                })
        
        return images
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[str]:
        """Extract table content as text summaries."""
        tables = []
        for table in soup.find_all("table"):
            tables.append(table.get_text(strip=True)[:200])  # First 200 chars
        return tables
    
    def _extract_internal_links(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract internal page links."""
        links = []
        
        # Standard Confluence page links
        for link in soup.find_all("ac:link"):
            page = link.find("ri:page")
            if page:
                links.append({
                    "page_title": page.get("ri:content-title", ""),
                    "text": link.get_text(strip=True),
                })
        
        return links
    
    def _clean_markdown(self, markdown: str) -> str:
        """Clean up converted Markdown."""
        # Remove excessive blank lines
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        
        # Remove leading/trailing whitespace
        markdown = markdown.strip()
        
        return markdown
