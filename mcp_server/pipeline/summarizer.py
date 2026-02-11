"""
Pipeline - Summarizer (Stage 2)

Map-Reduce summarization for large documents.
Handles edge cases: huge tables, code blocks, very long pages.
"""

import asyncio
import re
from typing import Optional, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk


class Summarizer:
    """
    Stage 2 Summarizer with Map-Reduce for long documents.
    
    Strategy:
    - Short pages (<3500 chars): Single LLM call
    - Long pages: Chunk → Map (parallel summarize) → Reduce (synthesize)
    - Tables: Condensed before chunking to avoid wasted tokens
    """
    
    CHUNK_SIZE = 3500
    CHUNK_OVERLAP = 200
    MAX_TABLE_ROWS = 10
    MAX_TABLE_CHARS = 1500
    
    def __init__(self):
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
    
    # ─────────────────────────────────────────────
    #  Table Handling (Edge Case)
    # ─────────────────────────────────────────────
    
    def condense_tables(self, text: str) -> str:
        """
        Condense huge markdown tables to prevent token waste.
        
        Keeps header + first N rows + summary row count.
        Handles both pipe-delimited and HTML tables.
        """
        # Handle markdown tables (pipe-delimited)
        text = self._condense_markdown_tables(text)
        
        # Handle HTML tables that survived parsing
        text = self._condense_html_tables(text)
        
        return text
    
    def _condense_markdown_tables(self, text: str) -> str:
        """Condense large markdown pipe tables."""
        lines = text.split("\n")
        result = []
        table_lines = []
        in_table = False
        
        for line in lines:
            stripped = line.strip()
            is_table_line = stripped.startswith("|") and stripped.endswith("|")
            is_separator = is_table_line and all(
                c in "-| :" for c in stripped
            )
            
            if is_table_line:
                if not in_table:
                    in_table = True
                    table_lines = []
                table_lines.append(line)
            else:
                if in_table:
                    # Table ended — condense it
                    result.extend(self._truncate_table(table_lines))
                    table_lines = []
                    in_table = False
                result.append(line)
        
        # Handle table at end of text
        if in_table and table_lines:
            result.extend(self._truncate_table(table_lines))
        
        return "\n".join(result)
    
    def _truncate_table(self, table_lines: List[str]) -> List[str]:
        """Keep header + separator + first N data rows + count."""
        if len(table_lines) <= self.MAX_TABLE_ROWS + 2:
            return table_lines  # Small table, keep as-is
        
        # header + separator + first N rows
        kept = table_lines[:2 + self.MAX_TABLE_ROWS]
        total_data_rows = len(table_lines) - 2  # minus header and separator
        omitted = total_data_rows - self.MAX_TABLE_ROWS
        kept.append(f"| ... ({omitted} more rows, {total_data_rows} total) |")
        return kept
    
    def _condense_html_tables(self, text: str) -> str:
        """Replace huge HTML tables with a summary."""
        def replace_large_table(match):
            table_html = match.group(0)
            if len(table_html) > self.MAX_TABLE_CHARS:
                row_count = table_html.count("<tr") + table_html.count("<TR")
                return f"[Table with ~{row_count} rows - condensed for summarization]"
            return table_html
        
        return re.sub(
            r"<table[\s\S]*?</table>",
            replace_large_table,
            text,
            flags=re.IGNORECASE,
        )
    
    # ─────────────────────────────────────────────
    #  Chunking
    # ─────────────────────────────────────────────
    
    def chunk_text(
        self,
        text: str,
        chunk_size: int = None,
        overlap: int = None,
    ) -> List[str]:
        """
        Split text into overlapping chunks, respecting paragraph boundaries.
        
        Args:
            text: Input text
            chunk_size: Max chars per chunk (default 3500)
            overlap: Overlap between chunks (default 200)
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.CHUNK_SIZE
        overlap = overlap or self.CHUNK_OVERLAP
        
        if len(text) <= chunk_size:
            return [text]
        
        # Split on paragraph boundaries (double newline) first
        paragraphs = re.split(r"\n\n+", text)
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph exceeds chunk_size
            if len(current_chunk) + len(para) + 2 > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Start next chunk with overlap from end of current
                    if overlap > 0 and len(current_chunk) > overlap:
                        current_chunk = current_chunk[-overlap:] + "\n\n" + para
                    else:
                        current_chunk = para
                else:
                    # Single paragraph exceeds chunk_size — force split
                    for i in range(0, len(para), chunk_size - overlap):
                        chunks.append(para[i:i + chunk_size])
                    current_chunk = ""
            else:
                current_chunk = (current_chunk + "\n\n" + para).strip()
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text[:chunk_size]]
    
    # ─────────────────────────────────────────────
    #  Extractive Summary (No LLM)
    # ─────────────────────────────────────────────
    
    def extractive_summary(
        self,
        text: str,
        num_sentences: int = 3,
    ) -> str:
        """
        Generate extractive summary using TF-IDF scoring.
        Fast, no LLM needed.
        """
        if not text or not text.strip():
            return ""
        
        # Condense tables first
        text = self.condense_tables(text)
        
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if len(sentences) <= num_sentences:
            return text
        
        if len(sentences) == 0:
            return text[:500] if len(text) > 500 else text
        
        try:
            tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = tfidf.fit_transform(sentences)
            scores = tfidf_matrix.sum(axis=1).A1
            
            top_indices = scores.argsort()[-num_sentences:][::-1]
            top_indices = sorted(top_indices)
            
            return " ".join([sentences[i] for i in top_indices])
        except Exception:
            return " ".join(sentences[:num_sentences])
    
    # ─────────────────────────────────────────────
    #  LLM Summary (Single Call — Short Pages)
    # ─────────────────────────────────────────────
    
    async def llm_summary(
        self,
        text: str,
        title: str,
        llm_provider,
    ) -> Optional[str]:
        """
        Generate LLM summary. For short pages or as map step.
        """
        # Condense tables before sending to LLM
        text = self.condense_tables(text)
        
        prompt = f"""Summarize this Confluence documentation page in 2-3 sentences.
Focus on:
- What the page is about (purpose)
- Key topics or concepts covered
- Target audience if apparent

Be concise and informative. Do not use phrases like "This page" or "This document".

Page Title: {title}
Page Content:
{text[:4000]}

Summary:"""
        
        try:
            response = await llm_provider.complete(prompt, max_tokens=150)
            return response.strip()
        except Exception:
            return None
    
    # ─────────────────────────────────────────────
    #  Map-Reduce Summary (Long Pages)
    # ─────────────────────────────────────────────
    
    async def map_summarize(
        self,
        chunks: List[str],
        title: str,
        llm_provider,
        max_concurrent: int = 3,
    ) -> List[str]:
        """
        MAP step: Summarize each chunk in parallel.
        
        Args:
            chunks: Text chunks from chunk_text()
            title: Page title for context
            llm_provider: LLM provider
            max_concurrent: Max parallel LLM calls
            
        Returns:
            List of chunk summaries
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def summarize_chunk(i: int, chunk: str) -> Tuple[int, str]:
            async with semaphore:
                prompt = f"""Summarize this section (part {i+1} of {len(chunks)}) from "{title}".
Capture the key points in 2-3 sentences. Be factual and specific.

Content:
{chunk}

Summary:"""
                try:
                    response = await llm_provider.complete(prompt, max_tokens=150)
                    return (i, response.strip())
                except Exception:
                    # Fallback to extractive for this chunk
                    return (i, self.extractive_summary(chunk, num_sentences=2))
        
        tasks = [summarize_chunk(i, chunk) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)
        
        # Sort by original order
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
    
    async def reduce_summarize(
        self,
        chunk_summaries: List[str],
        title: str,
        llm_provider,
    ) -> str:
        """
        REDUCE step: Synthesize chunk summaries into a master summary.
        
        Args:
            chunk_summaries: Summaries from map step
            title: Page title
            llm_provider: LLM provider
            
        Returns:
            Master summary (2-3 sentences)
        """
        combined = "\n\n".join(
            f"Section {i+1}: {s}" for i, s in enumerate(chunk_summaries)
        )
        
        prompt = f"""Below are summaries of different sections from the Confluence page "{title}".
Synthesize them into a single cohesive summary of 2-3 sentences.
Cover the most important points across ALL sections.
Do not use phrases like "This page" or "This document".

Section Summaries:
{combined}

Master Summary:"""
        
        try:
            response = await llm_provider.complete(prompt, max_tokens=200)
            return response.strip()
        except Exception:
            # Fallback: join first sentence of each chunk summary
            fallback = " ".join(
                s.split(".")[0] + "." for s in chunk_summaries if s
            )
            return fallback[:500]
    
    async def mapreduce_summary(
        self,
        text: str,
        title: str,
        llm_provider,
    ) -> Optional[str]:
        """
        Full Map-Reduce summary pipeline.
        
        Automatically chooses:
        - Short pages (<3500 chars): Single LLM call
        - Long pages: Map-Reduce
        
        Args:
            text: Full page content
            title: Page title
            llm_provider: LLM provider
            
        Returns:
            Summary string or None
        """
        if not text or not text.strip():
            return None
        
        # Condense tables before processing
        text = self.condense_tables(text)
        
        # Short page: single call
        if len(text) <= self.CHUNK_SIZE:
            return await self.llm_summary(text, title, llm_provider)
        
        # Long page: Map-Reduce
        chunks = self.chunk_text(text)
        
        # Map: parallel chunk summaries
        chunk_summaries = await self.map_summarize(chunks, title, llm_provider)
        
        # Reduce: synthesize master summary
        master = await self.reduce_summarize(chunk_summaries, title, llm_provider)
        
        return master
