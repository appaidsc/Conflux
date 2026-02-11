"""
Services - Search Service

Hybrid search with RRF fusion, FlashRank reranking, and importance boosting.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    SearchRequest,
    NamedVector,
    NamedSparseVector,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)

from mcp_server.config import get_settings
from mcp_server.pipeline.embedder import Embedder


@dataclass
class SearchResult:
    """Search result with metadata."""
    page_id: str
    title: str
    url: str
    snippet: str
    score: float
    importance: str
    has_code: bool
    labels: List[str]
    space_key: str


class SearchService:
    """Hybrid search with reranking and filtering."""
    
    COLLECTION_NAME = "confluence_pages"
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.embedder = Embedder(settings)
        self._client = None
        self._reranker = None
    
    @property
    def client(self) -> QdrantClient:
        """Lazy load Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(
                host=self.settings.qdrant.host,
                port=self.settings.qdrant.port,
                api_key=self.settings.qdrant.api_key,
            )
        return self._client
    
    @property
    def reranker(self):
        """Lazy load FlashRank reranker."""
        if self._reranker is None:
            from flashrank import Ranker
            self._reranker = Ranker(
                model_name="ms-marco-MiniLM-L-12-v2",
                cache_dir=str(self.settings.embedding.cache_dir),
            )
        return self._reranker
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        space_filter: Optional[List[str]] = None,
        include_code_pages: bool = False,
        rerank: bool = True,
    ) -> List[SearchResult]:
        """
        Hybrid search with dense + sparse vectors.
        
        Args:
            query: Search query text
            limit: Maximum results to return
            space_filter: Only include results from these spaces
            include_code_pages: If True, boost pages with code
            rerank: Apply FlashRank reranking
            
        Returns:
            List of SearchResult
        """
        # Generate query embeddings
        dense_vec, sparse_vec = self.embedder.embed_query(query)
        
        # Build filter
        filter_conditions = []
        if space_filter:
            filter_conditions.append(
                FieldCondition(
                    key="space_key",
                    match=MatchAny(any=space_filter),
                )
            )
        if include_code_pages:
            filter_conditions.append(
                FieldCondition(key="has_code", match=MatchValue(value=True))
            )
        
        search_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        # Hybrid search (both dense and sparse)
        prefetch_limit = limit * 3  # Prefetch more for reranking
        
        # Dense search
        dense_results = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=NamedVector(name="dense", vector=dense_vec),
            limit=prefetch_limit,
            query_filter=search_filter,
            with_payload=True,
        )
        
        # Sparse search
        sparse_results = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=NamedSparseVector(
                name="sparse",
                vector=SparseVector(
                    indices=sparse_vec["indices"],
                    values=sparse_vec["values"],
                ),
            ),
            limit=prefetch_limit,
            query_filter=search_filter,
            with_payload=True,
        )
        
        # RRF fusion
        fused = self._rrf_fusion(dense_results, sparse_results)
        
        # Apply importance boosting
        for result in fused:
            importance = result["payload"].get("importance_classification", "low")
            if importance == "high":
                result["score"] *= 1.15
            elif importance == "medium":
                result["score"] *= 1.05
        
        # Sort by adjusted score
        fused.sort(key=lambda x: x["score"], reverse=True)
        
        # Rerank with FlashRank if enabled
        if rerank and len(fused) > 0:
            fused = self._rerank(query, fused[:limit * 2])
        
        # Convert to SearchResult objects
        results = []
        for item in fused[:limit]:
            payload = item["payload"]
            snippet = payload.get("llm_summary") or payload.get("extractive_summary", "")
            
            results.append(SearchResult(
                page_id=payload["page_id"],
                title=payload["title"],
                url=payload["url"],
                snippet=snippet[:300],
                score=round(item["score"], 4),
                importance=payload.get("importance_classification", "low"),
                has_code=payload.get("has_code", False),
                labels=payload.get("labels", []),
                space_key=payload.get("space_key", ""),
            ))
        
        return results
    
    def _rrf_fusion(
        self,
        dense_results: List,
        sparse_results: List,
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion for combining dense/sparse results.
        
        RRF(d) = sum(1 / (k + rank(d)))
        """
        scores = {}
        payloads = {}
        
        for rank, result in enumerate(dense_results):
            doc_id = result.id
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            payloads[doc_id] = result.payload
        
        for rank, result in enumerate(sparse_results):
            doc_id = result.id
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            payloads[doc_id] = result.payload
        
        fused = [
            {"id": doc_id, "score": score, "payload": payloads[doc_id]}
            for doc_id, score in scores.items()
        ]
        
        return sorted(fused, key=lambda x: x["score"], reverse=True)
    
    def _rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using FlashRank."""
        from flashrank import RerankRequest
        
        passages = [
            {
                "id": str(c["id"]),
                "text": f"{c['payload'].get('title', '')} "
                        f"{c['payload'].get('llm_summary') or c['payload'].get('extractive_summary', '')}",
                "meta": c["payload"],
            }
            for c in candidates
        ]
        
        request = RerankRequest(query=query, passages=passages)
        reranked = self.reranker.rerank(request)
        
        result = []
        for item in reranked:
            result.append({
                "id": item["id"],
                "score": item["score"],
                "payload": item["meta"],
            })
        
        return result
