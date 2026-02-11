"""
Services - RAG Service (Stage 2)

Retrieval-Augmented Generation with section-level context injection.
Uses FlashRank to score sections against the question and injects
only the top relevant sections, not blind truncation.
"""

import re
from typing import List, Dict, Any, Optional, Literal, Tuple
from dataclasses import dataclass
import asyncio

from mcp_server.config import get_settings
from mcp_server.services.search_service import SearchService, SearchResult
from mcp_server.services.fetch_service import FetchService
from mcp_server.services.grounding_validator import GroundingValidator


@dataclass
class Citation:
    """Source citation."""
    page_id: str
    title: str
    url: str
    relevance: float


@dataclass
class RAGResponse:
    """RAG answer with citations."""
    answer: str
    sources: List[Citation]
    confidence: float
    mode_used: str
    grounding_score: Optional[float] = None


class RAGService:
    """
    Stage 2 RAG with section-level context injection.
    
    Instead of dumping the first 2000 chars of each page,
    we split pages into sections (by headers), score each section
    against the question using FlashRank, and inject only the
    top 3 most relevant sections into the LLM context.
    """
    
    """
    
    def __init__(self, settings=None, llm_provider=None):
        self.settings = settings or get_settings()
        self.max_context_sections = self.settings.rag.max_context_sections
        self.max_section_chars = self.settings.rag.max_section_chars
        
        self.search_service = SearchService(settings)
        self.fetch_service = FetchService(settings)
        self.grounding_validator = GroundingValidator()
        self.llm_provider = llm_provider
        self._reranker = None
    
    @property
    def reranker(self):
        """Lazy load FlashRank for section scoring."""
        if self._reranker is None:
            from flashrank import Ranker
            self._reranker = Ranker(
                model_name="ms-marco-MiniLM-L-12-v2",
                cache_dir=str(self.settings.embedding.cache_dir),
            )
        return self._reranker
    
    # ─────────────────────────────────────────────
    #  Section Splitting
    # ─────────────────────────────────────────────
    
    def _split_into_sections(self, markdown: str) -> List[Dict[str, str]]:
        """
        Split markdown content into logical sections by headers.
        
        Each section includes its heading + content until the next
        heading of same or higher level.
        
        Returns:
            List of {"heading": "...", "content": "...", "level": int}
        """
        lines = markdown.split("\n")
        sections = []
        current_heading = "Introduction"
        current_level = 0
        current_lines = []
        
        for line in lines:
            # Detect markdown heading
            heading_match = re.match(r'^(#{1,6})\s+(.+)', line)
            
            if heading_match:
                # Save previous section
                if current_lines:
                    content = "\n".join(current_lines).strip()
                    if content:
                        sections.append({
                            "heading": current_heading,
                            "content": content,
                            "level": current_level,
                        })
                
                # Start new section
                current_level = len(heading_match.group(1))
                current_heading = heading_match.group(2).strip()
                current_lines = [line]
            else:
                current_lines.append(line)
        
        # Don't forget the last section
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append({
                    "heading": current_heading,
                    "content": content,
                    "level": current_level,
                })
        
        # If no sections found (no headers), treat whole page as one
        if not sections:
            sections.append({
                "heading": "Content",
            sections.append({
                "heading": "Content",
                "content": markdown[:self.max_section_chars],
                "level": 0,
            })
                "level": 0,
            })
        
        return sections
    
    # ─────────────────────────────────────────────
    #  Section Scoring with FlashRank
    # ─────────────────────────────────────────────
    
    def _score_sections(
        self,
        question: str,
        sections: List[Dict[str, str]],
        page_title: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Score sections against the question using FlashRank.
        
        Returns sections sorted by relevance score (highest first).
        """
        if not sections:
            return []
        
        if len(sections) == 1:
            sections[0]["score"] = 1.0
            return sections
        
        from flashrank import RerankRequest
        
        passages = [
            {
                "id": str(i),
                "text": f"{s['heading']}: {s['content'][:1000]}",
                "meta": s,
            }
            for i, s in enumerate(sections)
        ]
        
        request = RerankRequest(query=question, passages=passages)
        reranked = self.reranker.rerank(request)
        
        scored = []
        for item in reranked:
            section = item["meta"]
            section["score"] = item["score"]
            scored.append(section)
        
        return scored
    
    # ─────────────────────────────────────────────
    #  Smart Context Builder (Stage 2)
    # ─────────────────────────────────────────────
    
    def _build_smart_context(
        self,
        question: str,
        pages: List[Dict[str, Any]],
        search_results: List[SearchResult],
    ) -> str:
        """
        Stage 2 context builder: section-level reranking.
        
        Instead of page['markdown'][:2000], we:
        1. Split each page into sections
        2. Score sections against the question
        3. Pick the top N most relevant sections
        
        This means we can answer questions hidden in the footer
        of a massive troubleshooting guide!
        """
        all_scored_sections = []
        
        for page in pages:
            sections = self._split_into_sections(page["markdown"])
            scored = self._score_sections(
                question, sections, page.get("title", "")
            )
            
            # Tag each section with its source page
            for s in scored:
                s["page_title"] = page.get("title", "Unknown")
                s["page_url"] = page.get("url", "")
                all_scored_sections.append(s)
        
        # Sort ALL sections across ALL pages by relevance
        all_scored_sections.sort(
            key=lambda x: x.get("score", 0), reverse=True
        )
        
        # Take top N sections
        top_sections = all_scored_sections[:self.max_context_sections]
        
        # Build context string
        context_parts = []
        for s in top_sections:
            part = (
                f"## {s['page_title']} > {s['heading']}\n\n"
                f"{s['content'][:self.max_section_chars]}\n\n"
            )
            context_parts.append(part)
        
        # Add brief summaries from remaining search results
        for result in search_results[3:]:
            context_parts.append(f"- **{result.title}**: {result.snippet}\n")
        
        return "\n".join(context_parts)
    
    # ─────────────────────────────────────────────
    #  Main RAG Pipeline
    # ─────────────────────────────────────────────
    
    async def ask_question(
        self,
        question: str,
        mode: Literal["extractive", "ai", "hybrid"] = "hybrid",
        top_k: int = 5,
        verify_grounding: bool = True,
    ) -> RAGResponse:
        """
        Answer a question using RAG with section-level context.
        """
        # Get relevant pages
        search_results = await self.search_service.search(
            query=question,
            limit=top_k,
            rerank=True,
        )
        
        if not search_results:
            return RAGResponse(
                answer="No relevant documentation found for your question.",
                sources=[],
                confidence=0.0,
                mode_used=mode,
            )
        
        # Fetch full content for top results
        pages = await asyncio.gather(*[
            self.fetch_service.get_page(r.page_id)
            for r in search_results[:3]
        ])
        pages = [p for p in pages if p]
        
        # Build SMART context (section-level, not truncated)
        context = self._build_smart_context(question, pages, search_results)
        
        # Generate answer based on mode
        if mode == "extractive":
            answer = self._extractive_answer(question, context)
            mode_used = "extractive"
        elif mode == "ai":
            answer = await self._ai_answer(question, context)
            mode_used = "ai"
        else:  # hybrid
            answer = self._extractive_answer(question, context)
            if len(answer) < 50 or "could not find" in answer.lower():
                answer = await self._ai_answer(question, context)
                mode_used = "ai"
            else:
                mode_used = "extractive"
        
        # Build citations
        citations = [
            Citation(
                page_id=r.page_id,
                title=r.title,
                url=r.url,
                relevance=r.score,
            )
            for r in search_results[:5]
        ]
        
        # Calculate confidence
        confidence = self._calculate_confidence(search_results, mode_used)
        
        # Validate grounding
        grounding_score = None
        if verify_grounding and self.llm_provider:
            grounding_result = await self.grounding_validator.validate(
                answer=answer,
                sources=[s["content"][:2000] for s in 
                         self._split_into_sections(p["markdown"])[:2]
                         for p in pages],
                llm_provider=self.llm_provider,
            )
            grounding_score = grounding_result.score
        
        return RAGResponse(
            answer=answer,
            sources=citations,
            confidence=confidence,
            mode_used=mode_used,
            grounding_score=grounding_score,
        )
    
    def _extractive_answer(self, question: str, context: str) -> str:
        """Simple extractive answer from context."""
        question_words = set(
            word.lower()
            for word in re.findall(r'\w+', question)
            if len(word) > 3
        )
        
        sentences = re.split(r'[.!?]\s+', context)
        scored_sentences = []
        
        for sentence in sentences:
            sentence_words = set(word.lower() for word in re.findall(r'\w+', sentence))
            overlap = len(question_words & sentence_words)
            if overlap > 0:
                scored_sentences.append((overlap, sentence))
        
        scored_sentences.sort(reverse=True)
        
        if scored_sentences:
            top_sentences = [s[1] for s in scored_sentences[:3]]
            return " ".join(top_sentences)
        
        return "Could not find a specific answer in the documentation."
    
    async def _ai_answer(self, question: str, context: str) -> str:
        """Generate AI answer using LLM."""
        if not self.llm_provider:
            return self._extractive_answer(question, context)
        
        prompt = f"""Answer the following question based ONLY on the provided documentation context.
If the answer cannot be found in the context, say so.
Be concise and cite which page(s) the information comes from.

Question: {question}

Context:
{context[:6000]}

Answer:"""
        
        try:
            response = await self.llm_provider.complete(prompt, max_tokens=500)
            return response.strip()
        except Exception:
            return self._extractive_answer(question, context)
    
    def _calculate_confidence(
        self,
        results: List[SearchResult],
        mode: str,
    ) -> float:
        """Calculate answer confidence score."""
        if not results:
            return 0.0
        
        top_score = results[0].score if results else 0
        mode_factor = 0.9 if mode == "ai" else 0.7
        
        if results[0].importance == "high":
            importance_factor = 1.1
        else:
            importance_factor = 1.0
        
        return min(1.0, top_score * mode_factor * importance_factor)
