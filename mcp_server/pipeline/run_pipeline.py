"""
Pipeline - Run Pipeline

CLI entry point for full/incremental/single-page sync.
"""

import asyncio
import argparse
from datetime import datetime
from typing import Optional
import logging

from mcp_server.config import get_settings
from mcp_server.pipeline.discovery import ConfluenceDiscovery
from mcp_server.pipeline.fetcher import ContentFetcher
from mcp_server.pipeline.parser import ContentParser
from mcp_server.pipeline.analyzer import ContentAnalyzer
from mcp_server.pipeline.importance_scorer import ImportanceScorer
from mcp_server.pipeline.summarizer import Summarizer
from mcp_server.pipeline.embedder import Embedder
from mcp_server.pipeline.indexer import QdrantIndexer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrates the full data ingestion pipeline."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        
        # Initialize components
        self.discovery = ConfluenceDiscovery(settings)
        self.fetcher = ContentFetcher(settings)
        self.parser = ContentParser(self.settings.confluence.base_url)
        self.analyzer = ContentAnalyzer()
        self.scorer = ImportanceScorer()
        self.summarizer = Summarizer()
        self.embedder = Embedder(settings)
        self.indexer = QdrantIndexer(settings)
    
    async def run_full(self, space_key: Optional[str] = None) -> dict:
        """
        Run full pipeline: discover all pages and index them.
        
        Args:
            space_key: Optional space override
            
        Returns:
            Statistics dict
        """
        logger.info("Starting full pipeline run...")
        
        # Ensure Qdrant collection exists
        self.indexer.ensure_collection()
        
        stats = {
            "discovered": 0,
            "indexed": 0,
            "skipped": 0,
            "errors": 0,
            "start_time": datetime.utcnow().isoformat(),
        }
        
        # Discover pages
        async for page_summary in self.discovery.discover_pages(space_key):
            stats["discovered"] += 1
            
            try:
                # Fetch full content
                page = await self.fetcher.fetch_page(page_summary.page_id)
                if not page:
                    stats["skipped"] += 1
                    continue
                
                # Parse HTML to Markdown
                parsed = self.parser.parse(page.html_content)
                
                # Analyze content
                analysis = self.analyzer.analyze(parsed)
                
                # Score importance
                importance = self.scorer.score(
                    children_count=len(page.children_ids),
                    depth=len(page.ancestors),
                    labels=page.labels,
                    analysis=analysis,
                )
                
                # Generate extractive summary
                extractive = self.summarizer.extractive_summary(parsed.markdown)
                
                # Generate embeddings
                dense, sparse = self.embedder.embed_for_index(page.title, extractive)
                
                # Build payload
                payload = self.indexer.build_payload(
                    page=page,
                    parsed=parsed,
                    analysis=analysis,
                    importance=importance,
                    extractive_summary=extractive,
                )
                
                # Upsert to Qdrant
                self.indexer.upsert(
                    page_id=page.page_id,
                    dense_vector=dense,
                    sparse_vector=sparse,
                    payload=payload,
                )
                
                stats["indexed"] += 1
                
                if stats["indexed"] % 10 == 0:
                    logger.info(f"Indexed {stats['indexed']} pages...")
                
            except Exception as e:
                logger.error(f"Error processing page {page_summary.page_id}: {e}")
                stats["errors"] += 1
        
        stats["end_time"] = datetime.utcnow().isoformat()
        logger.info(f"Pipeline complete: {stats}")
        return stats
    
    async def run_incremental(
        self,
        since: datetime,
        space_key: Optional[str] = None,
    ) -> dict:
        """
        Run incremental sync for pages modified since timestamp.
        
        Args:
            since: Only process pages modified after this time
            space_key: Optional space override
            
        Returns:
            Statistics dict
        """
        logger.info(f"Starting incremental sync since {since}...")
        
        stats = {
            "checked": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 0,
        }
        
        async for page_summary in self.discovery.get_modified_since(since, space_key):
            stats["checked"] += 1
            
            try:
                # Check if content actually changed
                page = await self.fetcher.fetch_page(page_summary.page_id)
                if not page:
                    continue
                
                parsed = self.parser.parse(page.html_content)
                
                # Check content hash
                import hashlib
                new_hash = hashlib.sha256(parsed.markdown.encode()).hexdigest()[:16]
                existing_hash = self.indexer.get_existing_hash(page_summary.page_id)
                
                if existing_hash == new_hash:
                    stats["skipped"] += 1
                    continue
                
                # Process and update
                analysis = self.analyzer.analyze(parsed)
                importance = self.scorer.score(
                    len(page.children_ids),
                    len(page.ancestors),
                    page.labels,
                    analysis,
                )
                extractive = self.summarizer.extractive_summary(parsed.markdown)
                dense, sparse = self.embedder.embed_for_index(page.title, extractive)
                
                payload = self.indexer.build_payload(
                    page, parsed, analysis, importance, extractive
                )
                
                self.indexer.upsert(page.page_id, dense, sparse, payload)
                stats["updated"] += 1
                
            except Exception as e:
                logger.error(f"Error updating {page_summary.page_id}: {e}")
                stats["errors"] += 1
        
        logger.info(f"Incremental sync complete: {stats}")
        return stats
    
    async def run_single_page(self, page_id: str) -> dict:
        """
        Index a single page.
        
        Args:
            page_id: Confluence page ID
            
        Returns:
            Result dict
        """
        logger.info(f"Indexing single page: {page_id}")
        
        self.indexer.ensure_collection()
        
        page = await self.fetcher.fetch_page(page_id)
        if not page:
            return {"success": False, "error": "Page not found"}
        
        parsed = self.parser.parse(page.html_content)
        analysis = self.analyzer.analyze(parsed)
        importance = self.scorer.score(
            len(page.children_ids),
            len(page.ancestors),
            page.labels,
            analysis,
        )
        extractive = self.summarizer.extractive_summary(parsed.markdown)
        dense, sparse = self.embedder.embed_for_index(page.title, extractive)
        
        payload = self.indexer.build_payload(
            page, parsed, analysis, importance, extractive
        )
        
        self.indexer.upsert(page.page_id, dense, sparse, payload)
        
        return {
            "success": True,
            "page_id": page_id,
            "title": page.title,
            "importance": importance.classification,
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Confluence MCP Data Pipeline")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full pipeline (discover and index all pages)",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Run incremental sync",
    )
    parser.add_argument(
        "--page-id",
        type=str,
        help="Index a single page by ID",
    )
    parser.add_argument(
        "--space",
        type=str,
        help="Override space key",
    )
    parser.add_argument(
        "--since",
        type=str,
        help="Incremental sync since (ISO datetime)",
    )
    
    args = parser.parse_args()
    
    runner = PipelineRunner()
    
    if args.full:
        asyncio.run(runner.run_full(args.space))
    elif args.incremental:
        since = datetime.fromisoformat(args.since) if args.since else datetime.utcnow()
        asyncio.run(runner.run_incremental(since, args.space))
    elif args.page_id:
        result = asyncio.run(runner.run_single_page(args.page_id))
        print(result)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
