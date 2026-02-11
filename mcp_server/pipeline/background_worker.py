"""
Pipeline - Background LLM Summary Worker

Progressive quality improvement: generates LLM summaries in background
and re-embeds pages for better search quality over time.
"""

import asyncio
import logging
from typing import Optional, List
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, ScrollRequest

from mcp_server.config import get_settings
from mcp_server.pipeline.fetcher import ContentFetcher
from mcp_server.pipeline.parser import ContentParser
from mcp_server.pipeline.summarizer import Summarizer
from mcp_server.pipeline.embedder import Embedder
from mcp_server.pipeline.indexer import QdrantIndexer
from mcp_server.llm import get_provider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BackgroundSummaryWorker:
    """
    Background worker that progressively improves search quality.
    
    Strategy:
    1. Initial indexing uses fast extractive summaries
    2. This worker runs in background, generating LLM summaries
    3. When LLM summary is ready, re-embed and update Qdrant
    4. Quality improves over time without blocking initial indexing
    """
    
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.COLLECTION_NAME = self.settings.qdrant.collection
        self._client = None
        self.fetcher = ContentFetcher(settings)
        self.parser = ContentParser(self.settings.confluence.base_url)
        self.summarizer = Summarizer()
        self.embedder = Embedder(settings)
        self.indexer = QdrantIndexer(settings)
        self._llm_provider = None
        self._running = False
    
    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(
                host=self.settings.qdrant.host,
                port=self.settings.qdrant.port,
                api_key=self.settings.qdrant.api_key,
            )
        return self._client
    
    @property
    def llm_provider(self):
        if self._llm_provider is None:
            self._llm_provider = get_provider(self.settings)
        return self._llm_provider
    
    def get_pages_without_llm_summary(self, limit: int = 10) -> List[dict]:
        """Find pages that don't have LLM summaries yet."""
        results = self.client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="has_llm_summary",
                        match=MatchValue(value=False),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        
        return [
            {
                "point_id": point.id,
                "page_id": point.payload["page_id"],
                "title": point.payload["title"],
                "extractive_summary": point.payload.get("extractive_summary", ""),
            }
            for point in results[0]
        ]
    
    async def process_page(self, page_info: dict) -> bool:
        """
        Generate LLM summary for a page and update its embedding.
        
        Returns True if successful.
        """
        page_id = page_info["page_id"]
        title = page_info["title"]
        
        try:
            # Fetch fresh content
            page = await self.fetcher.fetch_page(page_id)
            if not page:
                logger.warning(f"Could not fetch page {page_id}")
                return False
            
            # Parse to markdown
            parsed = self.parser.parse(page.html_content)
            
            # Generate LLM summary (Map-Reduce for long pages)
            llm_summary = await self.summarizer.mapreduce_summary(
                text=parsed.markdown,
                title=title,
                llm_provider=self.llm_provider,
            )
            
            if not llm_summary:
                logger.warning(f"LLM summary failed for {page_id}")
                return False
            
            # Re-embed with LLM summary (higher quality)
            dense, sparse = self.embedder.embed_for_index(title, llm_summary)
            
            # Update the point in Qdrant with new embedding + LLM summary
            self.client.set_payload(
                collection_name=self.COLLECTION_NAME,
                payload={
                    "llm_summary": llm_summary,
                    "has_llm_summary": True,
                    "llm_summary_at": datetime.utcnow().isoformat(),
                },
                points=[page_info["point_id"]],
            )
            
            # Update vectors
            self.client.update_vectors(
                collection_name=self.COLLECTION_NAME,
                points=[
                    {
                        "id": page_info["point_id"],
                        "vector": {
                            "dense": dense,
                            "sparse": sparse,
                        },
                    }
                ],
            )
            
            logger.info(f"✓ Updated {title} with LLM summary")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {page_id}: {e}")
            return False
    
    async def run_batch(self, batch_size: int = None) -> dict:
        """
        Process a batch of pages without LLM summaries.
        
        Returns statistics.
        """
        batch_size = batch_size or self.settings.pipeline.llm_summary_batch_size
        
        pages = self.get_pages_without_llm_summary(limit=batch_size)
        
        if not pages:
            logger.info("No pages without LLM summaries")
            return {"processed": 0, "success": 0, "remaining": 0}
        
        stats = {"processed": 0, "success": 0}
        
        for page in pages:
            success = await self.process_page(page)
            stats["processed"] += 1
            if success:
                stats["success"] += 1
            
            # Small delay to avoid overwhelming LLM
            await asyncio.sleep(1)
        
        # Check remaining
        remaining = self.get_pages_without_llm_summary(limit=1)
        stats["remaining"] = len(remaining) > 0
        
        logger.info(f"Batch complete: {stats}")
        return stats
    
    async def run_continuous(self, interval_seconds: int = 60):
        """
        Run continuously in background.
        
        Processes batches with delays between them.
        """
        self._running = True
        logger.info("Starting background LLM summary worker")
        
        while self._running:
            try:
                stats = await self.run_batch()
                
                if not stats.get("remaining"):
                    logger.info("All pages have LLM summaries!")
                    await asyncio.sleep(interval_seconds * 5)  # Longer wait
                else:
                    await asyncio.sleep(interval_seconds)
                    
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(interval_seconds * 2)
    
    def stop(self):
        """Stop the continuous worker."""
        self._running = False


async def run_background_worker():
    """CLI entry point for background worker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Background LLM Summary Worker")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run single batch and exit",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Pages per batch (default: 5)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between batches (default: 60)",
    )
    
    args = parser.parse_args()
    
    worker = BackgroundSummaryWorker()
    
    if args.batch:
        await worker.run_batch(args.batch_size)
    else:
        await worker.run_continuous(args.interval)


if __name__ == "__main__":
    asyncio.run(run_background_worker())
