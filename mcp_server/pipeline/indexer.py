"""
Pipeline - Indexer

Qdrant upsert operations with payload construction.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
    SparseVectorParams,
    SparseIndexParams,
    PayloadSchemaType,
)

from mcp_server.config import get_settings
from mcp_server.pipeline.fetcher import PageContent
from mcp_server.pipeline.parser import ParsedContent
from mcp_server.pipeline.analyzer import AnalysisResult
from mcp_server.pipeline.importance_scorer import ImportanceResult


class QdrantIndexer:
    """Indexes page data into Qdrant vector database."""
    
    COLLECTION_NAME = "confluence_pages"
    DENSE_VECTOR_SIZE = 768  # bge-base-en-v1.5
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._client = None
    
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
    
    def ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.COLLECTION_NAME for c in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config={
                    "dense": VectorParams(
                        size=self.DENSE_VECTOR_SIZE,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )
            
            # Create payload indexes for filtering
            for field, schema in [
                ("space_key", PayloadSchemaType.KEYWORD),
                ("has_code", PayloadSchemaType.BOOL),
                ("importance_classification", PayloadSchemaType.KEYWORD),
                ("labels", PayloadSchemaType.KEYWORD),
            ]:
                self.client.create_payload_index(
                    collection_name=self.COLLECTION_NAME,
                    field_name=field,
                    field_schema=schema,
                )
    
    def _generate_point_id(self, page_id: str) -> int:
        """Generate deterministic int64 point ID from page_id."""
        return int(hashlib.sha256(page_id.encode()).hexdigest()[:15], 16)
    
    def _compute_content_hash(self, content: str) -> str:
        """Compute content hash for change detection."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def build_payload(
        self,
        page: PageContent,
        parsed: ParsedContent,
        analysis: AnalysisResult,
        importance: ImportanceResult,
        extractive_summary: str,
        llm_summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build Qdrant payload from page data."""
        return {
            # Core identifiers
            "page_id": page.page_id,
            "title": page.title,
            "url": page.url,
            "space_key": page.space_key,
            "space_name": page.space_name,
            
            # Dual summaries
            "extractive_summary": extractive_summary,
            "llm_summary": llm_summary,
            "has_llm_summary": llm_summary is not None,
            "word_count": analysis.word_count,
            
            # Hierarchy
            "parent_id": page.parent_id,
            "children_ids": page.children_ids,
            "breadcrumb": page.ancestors,
            "depth": len(page.ancestors),
            "children_count": len(page.children_ids),
            
            # Metadata
            "labels": page.labels,
            "author": page.author,
            "author_id": page.author_id,
            "created_at": page.created_at.isoformat(),
            "updated_at": page.updated_at.isoformat(),
            "version": page.version,
            
            # Content analysis
            "has_code": analysis.has_code,
            "code_languages": analysis.code_languages,
            "code_block_count": analysis.code_block_count,
            "has_images": analysis.has_images,
            "image_count": analysis.image_count,
            "has_tables": analysis.has_tables,
            "table_count": analysis.table_count,
            "heading_structure": analysis.heading_structure,
            "internal_links_count": analysis.internal_links_count,
            
            # Importance scoring
            "importance_score": importance.score,
            "importance_classification": importance.classification,
            "importance_signals": importance.signals,
            
            # Sync tracking
            "indexed_at": datetime.utcnow().isoformat(),
            "content_hash": self._compute_content_hash(parsed.markdown),
        }
    
    def upsert(
        self,
        page_id: str,
        dense_vector: List[float],
        sparse_vector: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> None:
        """
        Upsert a single page to Qdrant.
        
        Args:
            page_id: Confluence page ID
            dense_vector: 768-dim dense embedding
            sparse_vector: {"indices": [...], "values": [...]}
            payload: Page metadata payload
        """
        point_id = self._generate_point_id(page_id)
        
        point = PointStruct(
            id=point_id,
            vector={
                "dense": dense_vector,
                "sparse": sparse_vector,
            },
            payload=payload,
        )
        
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[point],
        )
    
    def upsert_batch(
        self,
        points: List[Dict[str, Any]],
    ) -> None:
        """
        Upsert multiple pages at once.
        
        Args:
            points: List of dicts with page_id, dense_vector, sparse_vector, payload
        """
        qdrant_points = [
            PointStruct(
                id=self._generate_point_id(p["page_id"]),
                vector={
                    "dense": p["dense_vector"],
                    "sparse": p["sparse_vector"],
                },
                payload=p["payload"],
            )
            for p in points
        ]
        
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=qdrant_points,
            batch_size=50,
        )
    
    def get_existing_hash(self, page_id: str) -> Optional[str]:
        """Get content hash for existing page (for incremental sync)."""
        point_id = self._generate_point_id(page_id)
        
        try:
            result = self.client.retrieve(
                collection_name=self.COLLECTION_NAME,
                ids=[point_id],
                with_payload=["content_hash"],
            )
            if result:
                return result[0].payload.get("content_hash")
        except Exception:
            pass
        
        return None
    
    def delete(self, page_id: str) -> None:
        """Delete a page from the index."""
        point_id = self._generate_point_id(page_id)
        self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=[point_id],
        )
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics."""
        info = self.client.get_collection(self.COLLECTION_NAME)
        return {
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status.value,
        }
