# Confluence MCP Server - File Documentation

Complete reference for all 45 files in the project.

---

## Core Package (`mcp_server/`)

### `__init__.py`
Package initializer with version info (`__version__ = "1.0.0"`).

### `config.py`
**Pydantic Settings for all configuration.** Loads values from environment variables.

| Class | Purpose | Key Fields |
|-------|---------|------------|
| `ConfluenceSettings` | Confluence API | `base_url`, `username`, `api_token`, `space_key` |
| `QdrantSettings` | Vector DB | `host`, `port`, `collection`, `api_key` |
| `LLMSettings` | LLM provider | `provider`, `base_url`, `model`, `timeout_ms` |
| `EmbeddingSettings` | Embedding models | `model_dense`, `model_sparse`, `cache_dir` |
| `MCPSettings` | MCP server | `transport`, `port`, `host` |
| `PipelineSettings` | Ingestion | `max_pages`, `concurrency`, `enable_llm_summaries` |
| `CacheSettings` | Caching | `enabled`, `ttl_query`, `ttl_content` |
| `LogSettings` | Logging | `level`, `format` |

**Usage:** `settings = get_settings()`

### `server.py`
**FastMCP entry point.** Creates the MCP application, registers all tools, and runs the server.

```bash
python -m mcp_server.server --transport sse --port 8080
python -m mcp_server.server --transport stdio
```

---

## Pipeline Module (`mcp_server/pipeline/`)

Data ingestion pipeline: Confluence → Parse → Embed → Qdrant

### `discovery.py`
**CQL-based page discovery.** Finds pages in Confluence spaces.

| Method | Purpose |
|--------|---------|
| `discover_pages()` | Iterate all pages in a space (with pagination) |
| `get_modified_since()` | Get pages modified after timestamp (incremental sync) |

**Returns:** `AsyncIterator[PageSummary]` with `page_id`, `title`, `version`, `last_updated`

### `fetcher.py`
**REST API content fetching.** Gets full page content with metadata.

| Method | Purpose |
|--------|---------|
| `fetch_page(page_id)` | Fetch single page with HTML content |
| `get_page_version(page_id)` | **Stage 2:** Lightweight version-only check (~50ms) |
| `fetch_pages(page_ids)` | Parallel fetch multiple pages |
| `fetch_page_with_children()` | Recursive fetch with children |

**Returns:** `PageContent` with `html_content`, `ancestors`, `children_ids`, `labels`

### `parser.py`
**HTML → Markdown conversion.** Handles Confluence macros.

| Method | Purpose |
|--------|---------|
| `parse(html)` | Convert HTML to structured Markdown |
| `_process_macros()` | Handle code, info/note panels, expand macros |
| `_extract_headings()` | Extract heading structure |
| `_extract_code_blocks()` | Extract code with language info |

**Returns:** `ParsedContent` with `markdown`, `headings`, `code_blocks`, `images`

### `analyzer.py`
**Content metadata extraction.** Analyzes parsed content.

| Method | Purpose |
|--------|---------|
| `analyze(parsed)` | Extract metadata flags |

**Returns:** `AnalysisResult` with `has_code`, `code_languages`, `has_images`, `word_count`

### `importance_scorer.py`
**Importance scoring (0.0-3.0).** Classifies pages as high/medium/low.

**Scoring Formula:**
```
Score = Structural (max 1.5) + Hierarchy (max 0.5) + Labels (max 1.0) + Content (max 0.8)
```

| Signal | Weight | Factors |
|--------|--------|---------|
| Structural | 1.5 | Children count × 0.1, Internal links × 0.05 |
| Hierarchy | 0.5 | Depth 0-1 = 0.5, Depth 2 = 0.3 |
| Labels | 1.0 | Matching important labels × 0.25 |
| Content | 0.8 | Has code (+0.3), images (+0.3), >500 words (+0.2) |

### `summarizer.py`
**Stage 2: Map-Reduce summarization** with table handling.

| Method | Purpose |
|--------|---------|
| `extractive_summary(text)` | TF-IDF scored sentence extraction (fast, no LLM) |
| `llm_summary(text, llm)` | Single-call LLM summary (short pages) |
| `condense_tables(text)` | **Stage 2:** Truncate huge tables (50 rows → 10 + count) |
| `chunk_text(text)` | **Stage 2:** Split into 3500-char overlapping chunks |
| `map_summarize(chunks, llm)` | **Stage 2:** Parallel LLM summary for each chunk |
| `reduce_summarize(summaries, llm)` | **Stage 2:** Synthesize master summary |
| `mapreduce_summary(text, llm)` | **Stage 2:** Full pipeline (auto-picks single vs map-reduce) |

### `background_worker.py`
**Progressive quality improvement.** Runs after initial indexing.

| Method | Purpose |
|--------|---------|
| `get_pages_without_llm_summary()` | Find pages needing LLM summary |
| `process_page(page_info)` | Map-Reduce summary → re-embed → update Qdrant |
| `run_batch(batch_size)` | Process N pages |
| `run_continuous(interval)` | Background loop with configurable delay |

### `embedder.py`
**Vector generation.** Dense (BGE) + sparse (SPLADE).

| Method | Purpose | Output |
|--------|---------|--------|
| `embed_dense(texts)` | 768-dim dense vectors | `List[List[float]]` |
| `embed_sparse(texts)` | Sparse vectors | `List[{indices, values}]` |
| `embed_hybrid(texts)` | Both dense + sparse | `List[(dense, sparse)]` |
| `embed_for_index(title, summary)` | Index a single page | `(dense, sparse)` |
| `embed_query(query)` | Embed search query | `(dense, sparse)` |

### `indexer.py`
**Qdrant operations.** Upsert, delete, collection management.

| Method | Purpose |
|--------|---------|
| `ensure_collection()` | Create collection if missing |
| `build_payload()` | Construct Qdrant payload from page data |
| `upsert(page_id, vectors, payload)` | Insert/update single page |
| `upsert_batch(points)` | Batch insert multiple pages |
| `get_existing_hash(page_id)` | Check content hash (incremental sync) |
| `delete(page_id)` | Remove page from index |

### `run_pipeline.py`
**CLI orchestrator.** Full, incremental, single-page sync.

```bash
python -m mcp_server.pipeline.run_pipeline --full
python -m mcp_server.pipeline.run_pipeline --incremental --since 2024-01-01
python -m mcp_server.pipeline.run_pipeline --page-id 12345678
```

---

## Services Module (`mcp_server/services/`)

Business logic layer for search, fetch, RAG.

### `search_service.py`
**Hybrid search with reranking.**

| Method | Purpose |
|--------|---------|
| `search(query, limit, filters)` | Hybrid dense+sparse search |
| `_rrf_fusion()` | Reciprocal Rank Fusion for result merging |
| `_rerank()` | FlashRank cross-encoder reranking |

**Search Flow:**
1. Generate query embeddings (dense + sparse)
2. Search both vector types
3. RRF fusion to merge results
4. Apply importance boosting (high=+15%, medium=+5%)
5. Rerank with FlashRank

### `fetch_service.py`
**Live content retrieval with smart version-check caching (Stage 2).**

| Method | Purpose |
|--------|---------|
| `get_page(page_id)` | **Stage 2:** Version check → cached if unchanged, fetch if changed |
| `get_page_section(page_id, heading)` | Extract section by heading |
| `get_page_tree(page_id, depth)` | Build hierarchy tree |
| `compare_pages(id1, id2)` | Side-by-side comparison |

### `rag_service.py`
**Stage 2 RAG with section-level context injection.**

| Method | Purpose |
|--------|---------|
| `ask_question(q, mode)` | Full RAG pipeline |
| `_split_into_sections(markdown)` | **Stage 2:** Split page by headers into sections |
| `_score_sections(question, sections)` | **Stage 2:** FlashRank score each section vs question |
| `_build_smart_context(question, pages)` | **Stage 2:** Top 3 sections across all pages |
| `_extractive_answer()` | Keyword-based extraction |
| `_ai_answer()` | LLM-generated answer |

**Modes:**
- `extractive` - Fast, keyword-based, no LLM
- `ai` - LLM-generated answer
- `hybrid` - Try extractive, fall back to AI

### `grounding_validator.py`
**Hallucination detection.** Validates answers against sources.

| Method | Purpose |
|--------|---------|
| `validate()` | Keyword overlap scoring |
| `validate_with_llm()` | LLM-based NLI validation |

**Returns:** `GroundingResult` with `score` (0.0-1.0), `valid_claims`, `ungrounded_claims`

### `cache_service.py`
**TTL-based caching with version-aware methods (Stage 2).**

| Tier | TTL | Purpose |
|------|-----|---------|
| Query | 300s | Search results |
| Content | 900s | Page content (+ version check) |

| Method | Purpose |
|--------|---------|
| `get(key)` | Standard TTL cache get |
| `set(key, value)` | Standard TTL cache set |
| `get_versioned(key, version)` | **Stage 2:** Return cached only if version matches |
| `set_versioned(key, value, version)` | **Stage 2:** Store with version tag |

---

## LLM Module (`mcp_server/llm/`)

Provider abstraction for local and cloud LLMs.

### `base_provider.py`
**Abstract base class.** Defines interface for all providers.

| Method | Purpose |
|--------|---------|
| `complete(prompt)` | Text completion |
| `chat(messages)` | Chat completion |
| `is_available()` | Health check |

### `lm_studio_provider.py`
**LM Studio provider.** Uses OpenAI-compatible local API.

**Default:** `http://localhost:1234/v1`

### `openai_provider.py`
**OpenAI/Azure provider.** Standard OpenAI API.

---

## Schemas Module (`mcp_server/schemas/`)

Pydantic models for type safety and validation.

### `page.py`
| Model | Purpose |
|-------|---------|
| `PageMetadata` | Lightweight index data |
| `PageContent` | Full page with markdown |
| `PageTreeNode` | Recursive hierarchy node |

### `search.py`
| Model | Purpose |
|-------|---------|
| `SearchQuery` | Search input validation |
| `SearchResult` | Single result |
| `SearchResponse` | Full response with metadata |

### `rag.py`
| Model | Purpose |
|-------|---------|
| `Citation` | Source reference |
| `GroundingResult` | Validation result |
| `RAGResponse` | Answer + sources + confidence |

---

## Tools Module (`mcp_server/tools/`)

MCP tools exposed to AI assistants.

| Tool | Purpose | Key Args |
|------|---------|----------|
| `search_pages` | Semantic search | `query`, `limit`, `space_filter` |
| `get_page` | Fetch full content | `page_id`, `include_children` |
| `ask_question` | RAG Q&A | `question`, `mode`, `verify_grounding` |
| `search_code` | Find code snippets | `query`, `language` |
| `search_images` | Find diagrams | `query` |
| `get_related_pages` | Semantic similarity | `page_id` |
| `get_page_tree` | Page hierarchy | `page_id`, `max_depth` |
| `compare_pages` | Side-by-side diff | `page_id_1`, `page_id_2` |
| `get_page_section` | Extract section | `page_id`, `section_heading` |
| `analyze_image` | (Placeholder) | `page_id`, `image_index` |

---

## Tests (`tests/`)

### `test_pipeline.py`
Unit tests for `ContentParser`, `ContentAnalyzer`, `ImportanceScorer`, `Summarizer`.

### `test_services.py`
Integration tests for `SearchService`, `FetchService`, `RAGService`, `CacheService`.

---

## Configuration Files

| File | Purpose |
|------|---------|
| `.env.example` | Environment variable template |
| `.gitignore` | Excludes venv, models_cache, .env |
| `docker-compose.yml` | Qdrant container definition |
| `pytest.ini` | Test configuration |
| `requirements.txt` | Python dependencies |
