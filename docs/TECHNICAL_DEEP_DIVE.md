# Confluence MCP Server - Technical Deep Dive & Architecture Report

**Version**: 2.1 (Combined Reference)
**System**: Retrieval-Augmented Generation (RAG) for Confluence

This document serves as the **definitive technical reference** for the Confluence MCP Server. It details the **Core Architecture**, the **Data Ingestion Pipeline**, the **Vector Schema**, and provides a full **Operational Dry Run** trace.

---

## 1. Executive Summary

The Confluence MCP Server implements a **Stage 2 "Smart Context" RAG Architecture**. This architecture significantly improves upon standard RAG implementations by moving from page-level retrieval to **section-level context injection**, supported by a hybrid search pipeline (Dense + Sparse) and a "Fetch-on-Demand" strategy.

### The Problem with Standard RAG
*   **Traditional Approach**: Split pages into 500-token chunks *during indexing*. Store chunks as vectors.
*   **Flaw**: A 5000-word "Troubleshooting Guide" becomes 20 disjointed chunks. If the answer requires context from the top (OS version) and bottom (Error Code) of the page, standard RAG fails because it only retrieves one isolated chunk.

### The Stage 2 Solution
We index **whole pages** but reason about **sections** at query time (`fetch_service.py` + `rag_service.py`).

---

## 2. Technology Stack

The server is built on a modern, async-first Python stack designed for performance, modularity, and local-first AI capabilities.

| Category | Technology | Usage |
| :--- | :--- | :--- |
| **Core Framework** | **FastMCP** | Implements the Model Context Protocol (MCP) server, handling tool registration, SSE transport, and JSON-RPC. |
| **Runtime** | **Python 3.10+** | Logic execution, utilizing `asyncio` for high-concurrency fetching and processing. |
| **Vector Database** | **Qdrant** | Stores document embeddings. We use `qdrant-client` in **Hybrid Mode** (Dense + Sparse embeddings). |
| **Embedding** | **FastEmbed** | **BGE-Base-EN-v1.5** (Dense) and **Splade-PP-EN-v1** (Sparse). Runs locally on CPU/GPU. |
| **Reranking** | **FlashRank** | **ms-marco-MiniLM-L-12-v2**. Light-weight cross-encoder for reranking search results and scoring context sections. |
| **NLP & ML** | **NLTK, Scikit-learn** | Sentence tokenization and TF-IDF for extractive summarization. |
| **HTTP Client** | **HTTPX** | Async HTTP client for concurrent Confluence REST API calls. |
| **Configuration** | **Pydantic Settings** | Type-safe configuration management from environment variables (`.env`). |
| **LLM Interface** | **OpenAI / LM Studio** | Standard OpenAI-compatible API client for local/remote LLMs. |

---

## 3. Core Workflow: "Smart Context" Pipeline (RAG)

The RAG pipeline is the decision engine that transforms a raw user question into a grounded answer.

### Step-by-Step Execution Flow

**1. Hybrid Search & Retrieval** (`search_service.py`)
*   **Input**: User query $Q$ ("How do I configure the firewall?")
*   **Vector Construction**:
    *   **Dense**: `BGE-Base-EN-v1.5` creates a 768-dimension semantic vector $V_d$.
    *   **Sparse**: `Splade-PP-EN-v1` identifies high-value keywords ($V_s$).
*   **Parallel Retrieval**:
    *   Qdrant executes `Search(V_d)` -> Returns Top 30 semantically similar pages.
    *   Qdrant executes `Search(V_s)` -> Returns Top 30 keyword-matched pages.
*   **Fusion (RRF)**:
    *   We combine results using Reciprocal Rank Fusion: $Score(d) = \sum \frac{1}{60 + Rank(d)}$.
*   **Initial Rerank**: `FlashRank` (Cross-Encoder) re-scores the top fused results.
*   **Output**: Top 3-5 most relevant **Page IDs**.

**2. Just-In-Time Fetching** (`fetch_service.py`)
*   **Smart Caching Protocol**:
    1.  **Lightweight Check**: Call Confluence API `GET /content/{id}?expand=version`. Response size: ~1KB. Latency: <50ms.
    2.  **Cache Lookup**: Check local Redis-style dictionary: `If Cache[id].version == API.version` -> **Return Cached**.
    3.  **Full Fetch**: Only if versions mismatch, download full HTML `body.storage`.
*   **Result**: Always working with the **latest live version**.

**3. Dynamic Sectioning & Scoring** (`rag_service.py`)
*   **Segmentation**: Regex Parser `^(#{1,6})\s+(.+)` slices page into logical units ("Introduction", "Configuration", etc.).
*   **Cross-Encoder Scoring**:
    *   We construct customized scoring pairs: `Query + [Section Heading] + [Section Content]`.
    *   **FlashRank** assigns a relevance probability (0.0 - 1.0) to *each specific section*.
    *   *Example*:
        *   Query: "firewall port"
        *   Section A "Installation" -> Score 0.1
        *   Section B "Network Configuration" (mentions ports) -> Score **0.95**

**4. Context Injection**
*   **Selection**: We select the top `RAG_MAX_CONTEXT_SECTIONS` (default 3) *across all retrieved pages*.
*   **Prompt Construction**:
    ```markdown
    Answer the question based ONLY on this context:
    ## Server Setup > Network Configuration
    Open port 8080...
    ## Security Guide > Firewalls
    Ensure UFW is enabled...
    ```

---

## 4. Ingestion Pipeline Mechanics

The ingestion system (`run_pipeline.py`) transforms raw Confluence HTML into searchable vectors.

### Detailed Mechanics
1.  **Discovery**: A CQL query `space=KEY AND lastModified > LastSync` identifies changed pages incrementally.
2.  **Fetch & Parse**: HTML is converted to Markdown. Macros are stripped. Tables are preserved as ASCII representations.
3.  **Importance Scoring**: A heuristic boosts "Hub Pages".
    *   $Score = (Children \times 0.1) + (CodeBlocks \times 0.3) + (IsRoot \times 0.5)$
    *   Saved in Qdrant payload to boost standard search scores.
4.  **Summarization (Map-Reduce)**:
    *   **Map**: Split long pages (>4000 tokens) into 4 chunks. Summarize each.
    *   **Reduce**: Combine 4 summaries into one "Master Summary".
    *   *Result*: The embedding captures the *entire* conceptual scope of the page.
5.  **Indexing**: Vectors and Payload (JSON) are Upserted to Qdrant. Idempotency is enforced via Content Hash checks.

---

## 5. Vector Schema Specifications

Our Qdrant collection `confluence_pages` uses a specialized schema.

| Vector Name | Dimensions | Model | Purpose |
| :--- | :--- | :--- | :--- |
| `dense` | 768 | `BAAI/bge-base-en-v1.5` | Conceptual/Semantic search ("how to fix crash") |
| `sparse` | ~30k | `prithivida/Splade_PP_en_v1` | Acronyms/Keywords ("Error 503", "APIKey") |

**Payload Fields**:
*   `page_id` (Keyword): Primary Key.
*   `title` (Text): Page Title.
*   `content_hash` (Keyword): For idempotency (`indexer.py`).
*   `importance_score` (Float): Signal boost factor (`importance_scorer.py`).
*   `llm_summary` (Text): High-quality summary used for reranking.
*   `breadcrumb` (List): `["Home", "Developers", "API"]` for context.
*   `has_code` (Bool): Faceted search filter.

---

## 6. Operational Dry Run

Let's trace a single request to visualization exactly what happens.

**Scenario**: User invokes `ask_question("What is the refund policy?")`.

1.  **Request Received**: `server.py` routes the request to `ask_question`.
2.  **Config Load**: `RAGService` initializes with `top_k=5` and `mode="hybrid"`.
3.  **Search Step**:
    *   `Embedder` generates vectors for "What is the refund policy?".
    *   `Qdrant` finds Page A ("Sales Guide") and Page B ("Support Manual").
    *   `RRF` ranks them. Page A is #1.
4.  **Fetch Step**:
    *   `FetchService` calls `get_page("PageA")`.
    *   **Cache Check**: `FetchService` calls `get_page_version("PageA")`. API returns `v3`.
    *   **Cache Hit/Miss**: Local cache has `v2`. **Miss**.
    *   **Download**: `ContentFetcher` downloads full HTML of Page A.
    *   **Parse**: Converted to Markdown. Cache updated with `v3`.
5.  **Smart Context Step**:
    *   Page A is split into: "Introduction", "Pricing", "Refunds", "Support".
    *   `FlashRank` scores "Refunds" section as **0.98** relevance. "Pricing" is **0.4**.
    *   The "Refunds" section is selected.
6.  **LLM Step**:
    *   Prompt constructed: *Context: ## Page A > Refunds ... The policy is 30 days...*
    *   LLM Output: "The refund policy allows returns within 30 days."
7.  **Response**: JSON returned to user with valid citations.

---

## 7. Configuration Reference

All behavior is controlled via environment variables.

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| **RAG Settings** | | |
| `RAG_MAX_CONTEXT_SECTIONS` | 3 | How many distinct sections to feed the LLM. |
| `RAG_MAX_SECTION_CHARS` | 2000 | Character limit per section to prevent context overflow. |
| `RAG_VERIFY_GROUNDING` | True | Double-check answer support with LLM. |
| **Search Settings** | | |
| `QDRANT_HOST` | localhost | Vector DB location. |
| `QDRANT_COLLECTION` | confluence_pages | Bucket name. |
| **Pipeline Settings** | | |
| `PIPELINE_CONCURRENCY` | 10 | Max parallel concurrent requests to Confluence. |
| `ENABLE_LLM_SUMMARIES` | True | Use Map-Reduce for ingestion summaries. |
| **LLM Settings** | | |
| `LLM_PROVIDER` | lm_studio | AI Backend (`openai`, `azure`, `lm_studio`). |

---

## 8. Error Handling & Resilience

*   **HTML Parsing Failures**: If `ContentParser` fails, we fallback to stripping HTML tags via Regex.
*   **LLM Timeout**: If summarization fails, we fall back to `Extractive Summary` (First 3 sentences + TF-IDF keywords).
*   **Rate Limits**: `fetcher.py` uses exponential backoff for 429 errors.
*   **Sync Interruption**: The `last_synced` timestamp is only updated after a successful batch, ensuring no data loss on crash.
