"""
Pipeline - Embedder

Dense (BGE) and sparse (SPLADE) vector generation using fastembed.
"""

from typing import List, Tuple, Dict, Any
from pathlib import Path
import numpy as np

from mcp_server.config import get_settings


class Embedder:
    """Generates dense and sparse embeddings using local models."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._dense_model = None
        self._sparse_model = None
    
    @property
    def dense_model(self):
        """Lazy load dense embedding model."""
        if self._dense_model is None:
            from fastembed import TextEmbedding
            cache_dir = str(self.settings.embedding.cache_dir)
            self._dense_model = TextEmbedding(
                model_name=self.settings.embedding.model_dense,
                cache_dir=cache_dir,
            )
        return self._dense_model
    
    @property
    def sparse_model(self):
        """Lazy load sparse embedding model."""
        if self._sparse_model is None:
            from fastembed import SparseTextEmbedding
            cache_dir = str(self.settings.embedding.cache_dir)
            self._sparse_model = SparseTextEmbedding(
                model_name=self.settings.embedding.model_sparse,
                cache_dir=cache_dir,
            )
        return self._sparse_model
    
    def embed_dense(self, texts: List[str]) -> List[List[float]]:
        """
        Generate dense embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of dense vectors (768 dims for bge-base)
        """
        if not texts:
            return []
        
        embeddings = list(self.dense_model.embed(texts))
        return [emb.tolist() for emb in embeddings]
    
    def embed_sparse(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Generate sparse embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of sparse vectors {"indices": [...], "values": [...]}
        """
        if not texts:
            return []
        
        embeddings = list(self.sparse_model.embed(texts))
        return [
            {
                "indices": emb.indices.tolist(),
                "values": emb.values.tolist(),
            }
            for emb in embeddings
        ]
    
    def embed_hybrid(
        self,
        texts: List[str],
    ) -> List[Tuple[List[float], Dict[str, Any]]]:
        """
        Generate both dense and sparse embeddings.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of (dense_vector, sparse_vector) tuples
        """
        dense = self.embed_dense(texts)
        sparse = self.embed_sparse(texts)
        return list(zip(dense, sparse))
    
    def embed_for_index(
        self,
        title: str,
        summary: str,
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        Generate embeddings for a single page (title + summary).
        
        Args:
            title: Page title
            summary: Page summary (extractive or LLM)
            
        Returns:
            (dense_vector, sparse_vector) tuple
        """
        text = f"{title}\n\n{summary}"
        results = self.embed_hybrid([text])
        return results[0] if results else ([], {})
    
    def embed_query(self, query: str) -> Tuple[List[float], Dict[str, Any]]:
        """
        Generate embeddings for a search query.
        
        Args:
            query: Search query text
            
        Returns:
            (dense_vector, sparse_vector) tuple
        """
        results = self.embed_hybrid([query])
        return results[0] if results else ([], {})
