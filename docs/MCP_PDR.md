# Product Design Report: Confluence MCP Server

**Project Name:** Enterprise Confluence AI - MCP Integration  
**Version:** 1.0  
**Date:** February 2026  
**Author:** Engineering Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Proposed Solution](#3-proposed-solution)
4. [System Architecture](#4-system-architecture)
5. [Data Pipeline](#5-data-pipeline)
   - 5.1-5.5: Discovery, Fetch, Parse, Analyze, Summarize
   - 5.6: Embedding Models & Dimensions
   - 5.7: Storage in Qdrant
   - 5.8: Incremental Sync
   - 5.9: Performance Summary
   - 5.10: Two-Phase Async Pipeline (Optional)
6. [MCP Tools Specification](#6-mcp-tools-specification)
7. [LLM Integration](#7-llm-integration)
8. [Qdrant Schema](#8-qdrant-schema)
9. [Performance Analysis](#9-performance-analysis)
10. [Resilience & Error Handling](#10-resilience--error-handling)
11. [Security](#11-security)
12. [Deployment Plan](#12-deployment-plan)
13. [Testing Strategy](#13-testing-strategy)
14. [Observability](#14-observability)
15. [Roadmap](#15-roadmap)
16. [Planned Improvements](#16-planned-improvements) ⭐ NEW

---

## 1. Executive Summary

This PDR outlines the design for extending the existing Enterprise Confluence AI RAG system with a **Model Context Protocol (MCP) server**. The MCP layer enables seamless integration with AI assistants like **Claude Desktop**, **GitHub Copilot**, **LM Studio**, and **Microsoft Teams**.

### Key Innovation: Skeletal RAG

Instead of indexing full document content (which becomes stale), we implement **Skeletal RAG**:
- Index only **titles + extractive summaries**
- **Live fetch** full content from Confluence API at query time
- Result: **Always-fresh responses** with a tiny vector database footprint

### Goals

| Goal | Metric |
|------|--------|
| Reduce doc search time | 10-15 min → 3 seconds |
| Answer accuracy | >90% grounding score |
| Content freshness | Real-time (live fetch) |
| Multi-client support | 4+ AI clients |

---

## 2. Problem Statement

### Current Pain Points

```
┌─────────────────────────────────────────────────────────────────┐
│                      CURRENT STATE                              │
├─────────────────────────────────────────────────────────────────┤
│  ❌ Finding documentation takes 10-15 minutes                   │
│  ❌ Keyword search misses semantically relevant pages           │
│  ❌ No natural language Q&A capability                          │
│  ❌ RAG answers become stale after indexing                     │
│  ❌ Scattered across multiple tools (Confluence, Slack, etc.)   │
│  ❌ Context switching between IDE and browser                   │
└─────────────────────────────────────────────────────────────────┘
```

### Impact

- **Developer Productivity**: 30-60 min/day lost searching for docs
- **Knowledge Silos**: New team members struggle to find information
- **Stale Answers**: Indexed content diverges from live documentation

---

## 3. Proposed Solution

### Solution Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PROPOSED STATE                              │
├─────────────────────────────────────────────────────────────────┤
│  ✅ Semantic search in <1 second                                │
│  ✅ Natural language Q&A with citations                         │
│  ✅ Always-fresh content (live fetch)                           │
│  ✅ Access directly from IDE (Copilot, Claude Desktop)          │
│  ✅ Unified MCP interface for all AI clients                    │
│  ✅ Grounding validation prevents hallucinations                │
└─────────────────────────────────────────────────────────────────┘
```

### Skeletal RAG Approach

| Aspect | Traditional RAG | Skeletal RAG (Ours) |
|--------|-----------------|---------------------|
| Indexed Content | Full documents | Titles + extractive summaries |
| Vector DB Size | Large (GBs) | Small (MBs) |
| Content Freshness | Stale after indexing | Always current |
| Index Speed | Slow (LLM summaries) | Fast (TF-IDF) |
| Query Flow | Search → Return cached | Search → Live Fetch → Return fresh |

---

## 4. System Architecture

### 4.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       CLIENT LAYER                              │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │
│   │  Claude  │ │  GitHub  │ │ LM Studio│ │ Microsoft Teams  │  │
│   │  Desktop │ │  Copilot │ │          │ │   (Future)       │  │
│   └────┬─────┘ └────┬─────┘ └────┬─────┘ └────────┬─────────┘  │
└────────┼────────────┼────────────┼─────────────────┼───────────┘
         │   STDIO    │   STDIO    │   SSE          │   HTTP
         │            │            │                │
┌────────▼────────────▼────────────▼────────────────▼───────────┐
│                     INTEGRATION LAYER                          │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                   MCP SERVER                            │  │
│   │   • FastMCP (Python)                                    │  │
│   │   • Transport: STDIO, SSE, Streamable HTTP              │  │
│   │   • Port: 8080 (SSE), STDIO (embedded)                  │  │
│   └─────────────────────────────────────────────────────────┘  │
└────────────────────────────┬──────────────────────────────────┘
                             │
┌────────────────────────────▼──────────────────────────────────┐
│                    BUSINESS LOGIC LAYER                        │
│   ┌────────────┐  ┌────────────┐  ┌────────────────────────┐  │
│   │  10 MCP    │  │    LLM     │  │      SERVICES          │  │
│   │   TOOLS    │  │  PROVIDER  │  │                        │  │
│   │            │  │  LAYER     │  │  • SearchService       │  │
│   │ search     │  │            │  │  • FetchService        │  │
│   │ get_page   │  │ • LMStudio │  │  • RAGService          │  │
│   │ ask_question│ │ • Copilot  │  │  • GroundingValidator  │  │
│   │ etc.       │  │ • OpenAI   │  │  • CacheService        │  │
│   └────────────┘  └────────────┘  └────────────────────────┘  │
└────────────────────────────┬──────────────────────────────────┘
                             │
┌────────────────────────────▼──────────────────────────────────┐
│                       DATA LAYER                               │
│   ┌────────────┐  ┌────────────┐  ┌────────────────────────┐  │
│   │   QDRANT   │  │ CONFLUENCE │  │     LOCAL MODELS       │  │
│   │            │  │    API     │  │                        │  │
│   │ • Dense    │  │            │  │ • bge-base (768d)      │  │
│   │ • Sparse   │  │ • Pages    │  │ • SPLADE (sparse)      │  │
│   │ • Payloads │  │ • Search   │  │ • FlashRank (rerank)   │  │
│   └────────────┘  └────────────┘  └────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

### 4.2 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    ASK_QUESTION FLOW                            │
└─────────────────────────────────────────────────────────────────┘

User: "How do I configure Kafka authentication?"
                    │
                    ▼
┌───────────────────────────────────────┐
│ PHASE 1: Query Understanding (20ms)   │
│ • Spell check: "kafka" → "Kafka"      │
│ • Intent: HOWTO                       │
│ • Entities: [Kafka, authentication]   │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│ PHASE 2: Semantic Search (200ms)      │
│ • Dense embedding (bge-base)          │
│ • Sparse embedding (SPLADE)           │
│ • Hybrid search with RRF fusion       │
│ • Return: [page_123, page_456, ...]   │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│ PHASE 2.5: Reranking (80ms)           │
│ • Model: FlashRank (ms-marco-MiniLM)  │
│ • Uses LLM summary when available     │  ← Stage 2
│ • Reorder top-20 → top-5             │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│ PHASE 3: Smart Fetch (Stage 2)        │
│ • Version check first (~50ms)         │  ← Stage 2
│ • Cache hit if unchanged (0ms!)       │  ← Stage 2
│ • Full fetch only on version mismatch │
│ • Parallel fetch + Parse HTML → MD    │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│ PHASE 4: Smart Context (Stage 2)      │
│ • Split pages into sections by header │  ← Stage 2
│ • Score each section with FlashRank   │  ← Stage 2
│ • Inject top 3 most relevant sections │  ← Stage 2
│ • No more blind truncation!           │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│ PHASE 5: Answer Synthesis (1550ms)    │
│ MODE: HYBRID (default)                │
│ • Base: extractive summary            │
│ • LLM refines for coherence           │
│ • Stream response chunks              │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│ PHASE 6: Grounding Validation (100ms) │
│ • Score: 0.89 (89% grounded)          │
│ • Citations: 3 valid                  │
│ • Confidence: 88%                     │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│ RESPONSE                              │
│                                       │
│ Answer: To configure Kafka auth...    │
│ Sources: [3 pages with links]         │
│ Confidence: 88%                       │
│ Latency: 3.2s                         │
└───────────────────────────────────────┘
```

---

## 5. Data Pipeline

### 5.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION PIPELINE                             │
│                                                                             │
│  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌──────┐ │
│  │DISCOVER│──▶│ FETCH  │──▶│ PARSE  │──▶│ANALYZE │──▶│ EMBED  │──▶│STORE │ │
│  │        │   │        │   │        │   │        │   │        │   │      │ │
│  │ CQL    │   │ REST   │   │ HTML→  │   │ Score  │   │ Dense  │   │Qdrant│ │
│  │ Query  │   │ API    │   │ MD     │   │ +Tags  │   │+Sparse │   │      │ │
│  │        │   │        │   │        │   │        │   │        │   │      │ │
│  │100/sec │   │10/sec  │   │100/sec │   │50/sec  │   │20/sec  │   │50/sec│ │
│  └────────┘   └────────┘   └────────┘   └────────┘   └────────┘   └──────┘ │
│       │            │            │            │            │           │     │
│       ▼            ▼            ▼            ▼            ▼           ▼     │
│   page_ids     raw_html     markdown    importance    vectors     indexed   │
│                + meta       + struct     scores       768d+sp     in DB     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Phase 1: Discovery

**Purpose:** Find all pages in a Confluence space

```python
# API Call
GET /rest/api/content/search
?cql=space={SPACE_KEY} AND type=page
&expand=version
&limit=100
&start=0

# Output per page
{
    "page_id": "123456789",
    "title": "Kafka Authentication Guide",
    "version": 12,
    "last_updated": "2024-01-15T10:30:00Z"
}

# Performance
Speed: ~100 pages/second (paginated API)
Rate Limit: 100 requests/minute
Pagination: 100 results per request
```

### 5.3 Phase 2: Content Fetch

**Purpose:** Download full page content and metadata

```python
# API Call
GET /rest/api/content/{page_id}
?expand=body.storage,ancestors,children.page,metadata.labels

# Output
{
    "id": "123456789",
    "type": "page",
    "title": "Kafka Authentication Guide",
    "body": {
        "storage": {
            "value": "<p>This guide explains...</p>",  # Raw HTML
            "representation": "storage"
        }
    },
    "ancestors": [
        {"id": "111", "title": "Engineering"},
        {"id": "222", "title": "Kafka"}
    ],
    "children": {
        "page": {
            "results": [{"id": "333", "title": "SASL Setup"}]
        }
    },
    "metadata": {
        "labels": {
            "results": [{"name": "kafka"}, {"name": "security"}]
        }
    },
    "version": {"number": 12, "when": "2024-01-15T10:30:00Z"},
    "_links": {"webui": "/pages/viewpage.action?pageId=123456789"}
}

# Performance
Speed: ~10 pages/second (parallel requests)
Concurrency: 5-10 parallel requests
Rate Limit Handling: Exponential backoff on 429
```

### 5.4 Phase 3: Content Parsing

**Purpose:** Convert HTML to structured Markdown with metadata extraction

```python
# Input: Raw HTML
<h1>Kafka Setup</h1>
<p>This guide covers authentication.</p>
<ac:structured-macro ac:name="code">
    <ac:plain-text-body>bootstrap.servers=localhost:9092</ac:plain-text-body>
</ac:structured-macro>
<img src="/download/attachments/123/arch.png" alt="Architecture">

# Output: Parsed Content
{
    "markdown": "# Kafka Setup\n\nThis guide covers authentication.\n\n```properties\nbootstrap.servers=localhost:9092\n```\n\n![Architecture](/download/attachments/123/arch.png)",
    
    "structure": {
        "headings": ["Kafka Setup", "Configuration", "Testing"],
        "heading_hierarchy": [
            {"level": 1, "text": "Kafka Setup", "position": 0},
            {"level": 2, "text": "Configuration", "position": 120}
        ]
    },
    
    "code_blocks": [
        {
            "language": "properties",
            "code": "bootstrap.servers=localhost:9092",
            "position": 85
        }
    ],
    
    "images": [
        {
            "url": "/download/attachments/123/arch.png",
            "alt": "Architecture",
            "position": 150
        }
    ],
    
    "tables": [],
    
    "internal_links": [
        {"page_id": "456", "text": "security guide"}
    ],
    
    "word_count": 1850
}

# Libraries Used
- BeautifulSoup4: HTML parsing
- markdownify: HTML to Markdown conversion
- Custom macros: Confluence-specific elements

# Performance
Speed: ~100 pages/second
Memory: ~5MB per page (temporary)
```

### 5.5 Phase 4: Content Analysis & Importance Scoring

**Purpose:** Calculate importance score and extract smart metadata

```python
# Importance Score Calculation (0.0 - 3.0)

def calculate_importance(page_data: dict) -> tuple[float, dict]:
    signals = {}
    
    # STRUCTURAL (max 1.5)
    # Hub pages with many children are important
    structural = min(1.5,
        page_data["children_count"] * 0.1 +    # 10 children = 1.0
        page_data["internal_links"] * 0.05      # 10 links = 0.5
    )
    signals["structural"] = structural
    
    # HIERARCHY (max 0.5)
    # Top-level pages are more important
    depth = page_data["depth"]
    hierarchy = 0.5 if depth <= 1 else (0.3 if depth == 2 else 0.0)
    signals["hierarchy"] = hierarchy
    
    # LABELS (max 1.0)
    # Certain labels indicate important content
    important_labels = {
        "overview", "guide", "tutorial", "architecture",
        "getting-started", "reference", "api", "howto",
        "best-practices", "documentation", "runbook"
    }
    matched = set(page_data["labels"]) & important_labels
    label_score = min(1.0, len(matched) * 0.25)
    signals["labels"] = label_score
    signals["matched_labels"] = list(matched)
    
    # CONTENT (max 0.8)
    # Rich content indicates valuable documentation
    content = 0.0
    if page_data["has_code"]:
        content += 0.3
    if page_data["has_images"]:
        content += 0.3
    if page_data["word_count"] > 500:
        content += 0.2
    signals["content"] = min(0.8, content)
    
    total = structural + hierarchy + label_score + signals["content"]
    
    return total, signals

# Classification
def classify(score: float) -> str:
    if score >= 2.0:
        return "high"      # Hub pages, architecture docs
    elif score >= 1.0:
        return "medium"    # Technical docs with substance
    else:
        return "low"       # Meeting notes, simple pages

# Performance
Speed: ~50 pages/second
```

### 5.6 Phase 5: Hybrid Summarization ⭐ NEW

**Purpose:** Generate BOTH extractive and LLM summaries for maximum flexibility

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID SUMMARY STRATEGY                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: Full page content (Markdown)                            │
│                    │                                            │
│              ┌─────┴─────┐                                      │
│              │           │                                      │
│              ▼           ▼                                      │
│     ┌────────────┐ ┌────────────┐                              │
│     │ EXTRACTIVE │ │    LLM     │                              │
│     │  (TF-IDF)  │ │  SUMMARY   │                              │
│     │            │ │            │                              │
│     │ ~100/sec   │ │ ~2/sec     │                              │
│     │ Free       │ │ LLM cost   │                              │
│     │ Always run │ │ Optional   │                              │
│     └────────────┘ └────────────┘                              │
│              │           │                                      │
│              ▼           ▼                                      │
│     extractive_summary  llm_summary                             │
│     (stored always)     (stored if generated)                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Extractive Summary (Always Generated)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

def generate_extractive_summary(text: str, num_sentences: int = 3) -> str:
    """
    Extract most important sentences using TF-IDF scoring.
    No LLM needed - pure algorithmic extraction.
    """
    # Split into sentences
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text
    
    # Score sentences by TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(sentences)
    scores = tfidf_matrix.sum(axis=1).A1
    
    # Get top N sentence indices (preserve original order)
    top_indices = scores.argsort()[-num_sentences:][::-1]
    top_indices = sorted(top_indices)
    
    return " ".join([sentences[i] for i in top_indices])

# Example
text = "Kafka uses SASL for authentication. There are several mechanisms..."
extractve = generate_extractive_summary(text, num_sentences=3)
# Result: "Kafka uses SASL for authentication. The recommended approach 
#          for production is SCRAM-SHA-256. Configure the broker properties."

# Performance: ~100 pages/second (CPU only, no LLM)
```

#### LLM Summary — Map-Reduce (Stage 2 Enhancement)

```python
async def mapreduce_summary(text: str, title: str, llm_client) -> str:
    """
    Stage 2: Map-Reduce for documents of ANY length.
    Old approach: content[:4000] → lost data from end of long pages.
    New approach: chunk → parallel MAP → REDUCE master summary.
    """
    # Step 1: Condense huge tables (50 rows → 10 + count)
    text = condense_tables(text)
    
    # Step 2: Short pages → single LLM call
    if len(text) <= 3500:
        return await llm_summary(text, title, llm_client)
    
    # Step 3: MAP — chunk and summarize each piece in parallel
    chunks = chunk_text(text, size=3500, overlap=200)
    chunk_summaries = await map_summarize(chunks, title, llm_client)
    
    # Step 4: REDUCE — synthesize master summary
    master = await reduce_summarize(chunk_summaries, title, llm_client)
    return master

# Example: 20,000 char page → 6 chunks → 6 chunk summaries → 1 master
# Result: Vector DB now understands the ENTIRE document!

# Performance: ~2 pages/second (LLM bottleneck)
# Runs in background_worker.py, not blocking initial indexing
```

#### When to Use Which Summary

| Scenario | Use | Why |
|----------|-----|-----|
| Indexing (always) | Extractive | Fast, free, guaranteed |
| Indexing (if LLM available) | Both | Best of both worlds |
| Query: mode="fast" | Extractive | Low latency |
| Query: mode="quality" | LLM | Better coherence |
| LLM service down | Extractive | Graceful fallback |
| Search result snippets | Extractive | Quick display |
| Detailed answers | LLM | More context |

### 5.7 Phase 6: Embedding Generation

**Purpose:** Generate dense and sparse vectors for semantic search

#### What Are Embedding Dimensions?

**Dimensions** = The number of floating-point numbers representing text as a vector.

```
Text: "Kafka authentication guide"
           │
           ▼ (Embedding Model)
           
768-Dimensional Vector:
[0.023, -0.156, 0.089, 0.445, -0.012, 0.334, ... 768 numbers total]
```

| Aspect | Lower (384) | Higher (768-1024) |
|--------|-------------|-------------------|
| Storage | ✅ Smaller | ⚠️ Larger |
| Speed | ✅ Faster | ⚠️ Slower |
| Quality | ⚠️ Less nuance | ✅ More semantic detail |

**For 1000 pages:** `1000 × 768 × 4 bytes = ~3 MB` (negligible)

#### Recommended Models (All Open Source ✅)

| Type | Model | Dims | License | Use Case |
|------|-------|------|---------|----------|
| **Dense** | `BAAI/bge-base-en-v1.5` | 768 | MIT ✅ | Semantic similarity |
| **Sparse** | `prithivida/Splade_PP_en_v1` | ~30K | CC-BY-NC | Keyword matching |
| **Image** | `openai/clip-vit-base-patch32` | 512 | MIT ✅ | Diagram search |
| **Code** | `microsoft/codebert-base` | 768 | MIT ✅ | Code block search |

**All models run 100% locally - no API keys, no costs, no internet after first download.**

#### Implementation

```python
from fastembed import TextEmbedding, SparseTextEmbedding

# Models download once to ~/.cache/huggingface/ (~400MB each)
embedder = TextEmbedding("BAAI/bge-base-en-v1.5")
sparse_embedder = SparseTextEmbedding("prithivida/Splade_PP_en_v1")

# What Gets Embedded (title + best summary)
if llm_summary:
    embedded_text = f"{title}\n\n{llm_summary}"  # Prefer LLM if available
else:
    embedded_text = f"{title}\n\n{extractive_summary}"

# Generate vectors
dense_vector = list(embedder.embed([embedded_text]))[0]   # [768 floats]
sparse_vector = list(sparse_embedder.embed([embedded_text]))[0]
# sparse: {"indices": [123, 456, ...], "values": [0.5, 0.3, ...]}

# Performance
# Dense: ~50ms per text
# Sparse: ~30ms per text
# Batch: 20 pages/second (CPU), 60-80/sec (GPU)
```

#### Why These Models?

| Model | Why Chosen |
|-------|------------|
| **bge-base-en-v1.5** | Best balance of quality (63.55 MTEB) and speed. Proven in production. |
| **SPLADE** | Native Qdrant support. Captures exact keywords like "Kafka", "SASL". |
| **CLIP** | Best for matching text queries to diagram images. |
| **CodeBERT** | Understands code structure, not just text similarity. |

#### Alternatives

| If You Need... | Consider |
|----------------|----------|
| Faster/smaller | `bge-small-en-v1.5` (384d) |
| Higher quality | `bge-large-en-v1.5` (1024d) |
| Newest SOTA | `jina-embeddings-v3` (1024d) |

### 5.8 Phase 7: Storage in Qdrant

**Purpose:** Store vectors and payload for retrieval

```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

client = QdrantClient(host="localhost", port=6333)

# Create Collection (One-time)
client.create_collection(
    collection_name="confluence_pages",
    vectors_config={
        "dense": VectorParams(size=768, distance=Distance.COSINE)
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(modifier=Modifier.IDF)
    }
)

# Upsert Point
point = PointStruct(
    id=hash(page_id),  # Unique int64
    vector={
        "dense": dense_vector,      # [768 floats]
        "sparse": sparse_vector     # {"indices": [...], "values": [...]}
    },
    payload={
        # === CORE IDENTIFIERS ===
        "page_id": "123456789",
        "title": "Kafka Authentication Guide",
        "url": "https://confluence.example.com/pages/123456789",
        "space_key": "DOCS",
        "space_name": "Engineering Documentation",
        
        # === DUAL SUMMARIES (Hybrid Strategy) ===
        "extractive_summary": "Kafka uses SASL for authentication. The recommended approach for production is SCRAM-SHA-256 with SSL encryption. Configure the broker properties file before deployment.",
        "llm_summary": "This guide explains how to configure SASL authentication for Kafka clusters. It covers SCRAM-SHA-256 setup, broker configuration, and client authentication with detailed code examples.",
        "has_llm_summary": True,  # Flag to check if LLM summary exists
        "word_count": 1850,
        
        # === HIERARCHY ===
        "parent_id": "987654",
        "children_ids": ["111111", "222222", "333333"],  # Direct child pages
        "breadcrumb": [
            {"id": "111", "title": "Engineering"},
            {"id": "222", "title": "Kafka"}
        ],
        "depth": 2,
        "children_count": 3,
        
        # === METADATA ===
        "labels": ["kafka", "security", "guide"],
        "author": "John Doe",
        "author_id": "jdoe",
        "created_at": "2024-01-10T09:00:00Z",
        "updated_at": "2024-01-15T14:30:00Z",
        "version": 12,
        
        # === CONTENT ANALYSIS ===
        "has_code": True,
        "code_languages": ["java", "properties"],
        "code_block_count": 5,
        "has_images": True,
        "image_count": 3,
        "has_tables": True,
        "table_count": 1,
        "heading_structure": ["Overview", "SASL Config", "Testing"],
        "internal_links_count": 7,
        
        # === IMPORTANCE SCORING ===
        "importance_score": 2.4,
        "importance_classification": "high",
        "importance_signals": {
            "structural": 0.5,
            "hierarchy": 0.3,
            "labels": 1.0,
            "content": 0.6
        },
        
        # === SYNC TRACKING ===
        "indexed_at": "2024-01-15T15:00:00Z",
        "content_hash": "a3f2b8c9d1e4f5..."
    }
)

client.upsert(collection_name="confluence_pages", points=[point])

# Create Payload Indexes (for filtering)
for field in ["space_key", "has_code", "importance_classification", "labels"]:
    client.create_payload_index(
        collection_name="confluence_pages",
        field_name=field,
        field_schema="keyword"
    )

# Performance
Upsert speed: ~50 points/second
Storage: ~2KB per point (vectors + payload)
1000 pages = ~2MB storage
```

### 5.8 Incremental Sync

**Purpose:** Keep index fresh without full re-indexing

```python
# Sync Strategy
┌─────────────────────────────────────────────────────────────────┐
│                    INCREMENTAL SYNC FLOW                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Get last sync timestamp from metadata                       │
│  2. Query Confluence: lastModified > lastSync                   │
│  3. For each modified page:                                     │
│     a. Fetch current version from Qdrant                        │
│     b. Compare version numbers                                  │
│     c. If changed: re-fetch, re-process, upsert                 │
│  4. Update lastSync timestamp                                   │
│                                                                 │
│  Schedule: Every 1 hour (cron)                                  │
│  Speed: 10x faster than full re-index                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

# Implementation
async def incremental_sync():
    last_sync = get_last_sync_timestamp()
    
    # Query for modified pages
    modified = confluence_client.search(
        cql=f"lastModified > '{last_sync}' AND space = '{SPACE_KEY}'"
    )
    
    for page_summary in modified:
        # Check if version changed
        existing = qdrant_client.retrieve(
            collection_name="confluence_pages",
            ids=[hash(page_summary.id)]
        )
        
        if not existing or existing[0].payload["version"] < page_summary.version:
            # Re-process changed page
            page = await fetch_page(page_summary.id)
            parsed = parse_content(page)
            analyzed = analyze_content(parsed)
            vectors = generate_embeddings(analyzed)
            upsert_to_qdrant(vectors, analyzed)
    
    save_last_sync_timestamp(datetime.now())

# Sync Modes
FULL:        python -m mcp_server.pipeline.run_pipeline --full
INCREMENTAL: python -m mcp_server.pipeline.run_pipeline --incremental
SINGLE PAGE: python -m mcp_server.pipeline.run_pipeline --page-id 123456
```

### 5.9 Pipeline Performance Summary

| Phase | Speed (CPU) | Speed (GPU) | Bottleneck? |
|-------|-------------|-------------|-------------|
| Discovery | 100/sec | 100/sec | No |
| Fetch | 10/sec | 10/sec | Rate limited |
| Parse | 100/sec | 100/sec | No |
| Analyze | 50/sec | 50/sec | No |
| Embed | **20/sec** | **60/sec** | **YES** |
| Store | 50/sec | 50/sec | No |

**Total Time Estimates:**

| Scale | CPU Only | With GPU |
|-------|----------|----------|
| 100 pages | 30 sec | 15 sec |
| 1,000 pages | 4 min | 2 min |
| 10,000 pages | 40 min | 15 min |

### 5.10 Optional: Two-Phase Async Pipeline ⭐

For **faster time-to-value**, run extractive indexing first, then LLM enhancement in background.

```
┌─────────────────────────────────────────────────────────────────┐
│                   TWO-PHASE PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PHASE 1: FAST (System usable immediately!)                     │
│  ─────────────────────────────────────────                      │
│  Discovery → Parse → Extractive → Embed → Store                 │
│  Time: ~4 min for 1000 pages                                    │
│  Result: Search works! ✅                                       │
│                                                                 │
│  PHASE 2: BACKGROUND (Quality upgrade, optional)                │
│  ───────────────────────────────────────────────                │
│  For each page (async):                                         │
│    → Generate LLM summary                                       │
│    → Update Qdrant payload                                      │
│    → Re-embed with LLM summary                                  │
│  Time: ~10 min for 1000 pages                                   │
│  Result: Better search quality ✅                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Benefits

| Benefit | Explanation |
|---------|-------------|
| **Fast time-to-value** | System usable after Phase 1 (~4 min) |
| **No LLM blocking** | LLM failures don't break pipeline |
| **Gradual improvement** | Search quality improves as Phase 2 runs |
| **Interruptible** | Can stop Phase 2 anytime |

#### Implementation

```python
# Phase 1: Fast indexing (mandatory)
async def phase1_fast_index():
    for page in pages:
        content = await fetch_page(page.id)
        extractive = generate_extractive_summary(content)  # Fast, local
        vector = embed(f"{page.title}\n\n{extractive}")
        
        await qdrant.upsert(
            id=page.id,
            vector=vector,
            payload={
                "extractive_summary": extractive,
                "llm_summary": None,
                "has_llm_summary": False,  # Flag for Phase 2
            }
        )

# Phase 2: Background LLM enhancement (optional)
async def phase2_llm_enhancement():
    pages_without_llm = await qdrant.scroll(
        filter={"has_llm_summary": False}
    )
    
    for page in pages_without_llm:
        llm_summary = await generate_llm_summary(page.content)
        new_vector = embed(f"{page.title}\n\n{llm_summary}")
        
        await qdrant.update(
            id=page.id,
            vector=new_vector,
            payload={
                "llm_summary": llm_summary,
                "has_llm_summary": True
            }
        )
```

#### CLI Commands

```bash
# Run Phase 1 only (fast)
python -m mcp_server.pipeline.run_pipeline --phase 1

# Run Phase 2 in background
python -m mcp_server.pipeline.run_pipeline --phase 2 --background

# Run both phases sequentially
python -m mcp_server.pipeline.run_pipeline --full
```

#### Timeline for 1000 Pages

```
0 min ──────► 4 min ──────────────────► 14 min
   │            │                          │
   ▼            ▼                          ▼
 Start      Phase 1 Done              Phase 2 Done
            (Search works!)         (Search improved)
```

---

## 6. MCP Tools Specification

### 6.1 Tools Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MCP TOOLS ARCHITECTURE                               │
└─────────────────────────────────────────────────────────────────────────────┘

                                    USER
                                      │
                                      ▼
                         ┌─────────────────────┐
                         │   AI ASSISTANT      │
                         │ (Claude/LM Studio)  │
                         └──────────┬──────────┘
                                    │ MCP Protocol
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MCP SERVER                                      │
├──────────────┬──────────────┬──────────────┬──────────────┬────────────────┤
│  SEARCH      │   FETCH      │  NAVIGATION  │  ANALYSIS    │  ORCHESTRATOR  │
│  TOOLS       │   TOOLS      │  TOOLS       │  TOOLS       │                │
├──────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
│search_pages  │get_page_with │get_related   │compare_pages │  ask_question  │
│search_code   │  _children   │  _pages      │analyze_image │  (uses all     │
│search_images │get_page      │get_page_tree │              │   others)      │
│              │  _section    │              │              │                │
└──────┬───────┴──────┬───────┴──────┬───────┴──────┬───────┴────────┬───────┘
       │              │              │              │                │
       ▼              ▼              ▼              ▼                ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐
│   QDRANT    │ │ CONFLUENCE  │ │   QDRANT    │ │     LLM     │ │ COMBINES   │
│   (Index)   │ │    API      │ │  (Links)    │ │  (Analysis) │ │    ALL     │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘
```

### 6.2 Tool Categories

| Category | Tools | Data Source | Purpose |
|----------|-------|-------------|---------|
| **Search** | `search_pages`, `search_code`, `search_images` | Qdrant Index | Fast discovery |
| **Fetch** | `get_page_with_children`, `get_page_section` | Live Confluence | Fresh content |
| **Navigation** | `get_related_pages`, `get_page_tree` | Qdrant Index | Structure |
| **Analysis** | `compare_pages`, `analyze_image` | Live + LLM | Deep insights |
| **Orchestrator** | `ask_question` | All of above | Full Q&A |

### 6.3 Tool Summary Table

| Tool | Category | Description | Avg Latency |
|------|----------|-------------|-------------|
| `search_pages` | Search | Semantic search over index | 200ms |
| `get_page_with_children` | Fetch | Live fetch page + children | 1200ms |
| `ask_question` | RAG | Full Q&A with synthesis | 3200ms |
| `get_page_section` | Fetch | Extract specific section | 500ms |
| `search_code` | Search | Find pages with code | 300ms |
| `search_images` | Search | Find pages with diagrams | 300ms |
| `get_related_pages` | Navigation | Find similar/linked pages | 400ms |
| `get_page_tree` | Navigation | Get hierarchy | 200ms |
| `compare_pages` | Analysis | Diff two pages | 1500ms |
| `analyze_image` | Analysis | Describe embedded images | 2000ms |

### 6.4 How Each Tool Works

#### Search Tools (Fast - Use Qdrant Index)

```
┌─────────────────────────────────────────────────────────────────┐
│ search_pages / search_code / search_images                      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  1. Embed query (dense + sparse)        │
│  2. Hybrid search in Qdrant             │
│  3. Rerank top results (FlashRank)      │
│  4. Return SUMMARIES (not full content) │  ← Fast because no live fetch
└─────────────────────────────────────────┘
         │
         ▼
Returns: [{page_id, title, snippet, score}, ...]
```

#### Fetch Tools (Slower - Live Confluence API)

```
┌─────────────────────────────────────────────────────────────────┐
│ get_page_with_children / get_page_section                       │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  1. GET /content/{id}?expand=body       │  ← LIVE API call
│  2. Parse HTML → Markdown               │
│  3. Fetch children (if requested)       │
│  4. Return FULL content                 │  ← Complete, fresh content
└─────────────────────────────────────────┘
         │
         ▼
Returns: {page: {content: "Full markdown..."}, children: [...]}
```

#### ask_question Orchestration Flow (The Main Tool)

```
┌─────────────────────────────────────────────────────────────────┐
│                   ask_question Tool                              │
│                                                                  │
│  User: "How do I configure Kafka SASL authentication?"          │
│                                                                  │
│  STEP 1: Query Understanding (20ms)                             │
│  ├── Detect intent: "HOWTO"                                     │
│  └── Extract entities: ["Kafka", "SASL", "authentication"]      │
│                                                                  │
│  STEP 2: Semantic Search (200ms)                                │
│  └── search_pages("kafka SASL authentication")                  │
│      → Found: [page_123, page_456, page_789]                    │
│                                                                  │
│  STEP 3: Reranking (80ms) — uses LLM summary when available     │
│  └── FlashRank reorder → [page_123, page_789, page_456]         │
│                                                                  │
│  STEP 4: Smart Fetch (Stage 2 — Version Check)                  │
│  ├── Check versions (~50ms each)                                │
│  ├── Cache hit if unchanged (0ms!)                              │
│  └── Full fetch only on version mismatch                        │
│                                                                  │
│  STEP 5: Section-Level Context (Stage 2)                        │
│  └── Split → Score sections → Inject top 3 most relevant        │
│                                                                  │
│  STEP 6: Answer Synthesis with LLM (1500ms)                     │
│  └── Generate answer with citations                             │
│                                                                  │
│  STEP 7: Grounding Validation (100ms)                           │
│  └── Verify citations match sources (89% grounded)              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
Returns: {
  answer: "To configure Kafka SASL authentication:\n1. Edit...",
  sources: [{title: "Kafka Auth Guide", url: "..."}],
  confidence: 0.88,
  grounding_score: 0.89,
  latency_ms: 3200
}
```

### 6.5 Tool Interaction Diagram

```
                    ask_question (ORCHESTRATOR)
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
     search_pages    get_page_with    get_page_section
           │          _children              │
           │               │                 │
           └───────────────┼─────────────────┘
                           ▼
                    ┌─────────────┐
                    │    LLM      │
                    │  Synthesis  │
                    └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Grounding  │
                    │  Validator  │
                    └─────────────┘
                           │
                           ▼
                      FINAL ANSWER
```

### 6.6 Tool Specifications

#### `search_pages`

```yaml
Name: search_pages
Description: Semantic search over Confluence documentation
Parameters:
  query: string (required) - Natural language query
  space_filter: list[string] (optional) - Limit to specific spaces
  limit: int (optional, default=10) - Max results
  include_code_pages: bool (optional) - Include pages with code
  
Returns:
  results: list
    - page_id: string
    - title: string
    - url: string
    - snippet: string (extractive summary)
    - score: float (0.0-1.0)
    - importance: string (high/medium/low)

Example:
  Input: search_pages(query="kafka authentication", limit=5)
  Output: [
    {page_id: "123", title: "Kafka Auth Guide", score: 0.92, ...},
    {page_id: "456", title: "Security Overview", score: 0.85, ...}
  ]
```

#### `ask_question`

```yaml
Name: ask_question
Description: Natural language Q&A with RAG synthesis
Parameters:
  question: string (required) - User's question
  mode: enum (optional) - "extractive" | "ai" | "hybrid" (default)
  max_sources: int (optional, default=3) - Max source pages
  
Returns:
  answer: string - Synthesized answer
  sources: list
    - title: string
    - url: string
    - relevance: float
  confidence: float (0.0-1.0)
  grounding_score: float (0.0-1.0)
  mode_used: string
  latency_ms: int

Modes:
  extractive: Fast (100ms), uses pre-stored summaries, no LLM
  ai: Best quality (3000ms), full LLM synthesis
  hybrid: Balanced (1500ms), extractive base + LLM refinement
```

#### `get_page_with_children`

```yaml
Name: get_page_with_children
Description: Fetch live page content including child pages
Parameters:
  page_id: string (required) - Confluence page ID
  include_children: bool (optional, default=true)
  max_depth: int (optional, default=2) - Child page depth
  format: enum (optional) - "markdown" | "html" | "raw"
  
Returns:
  page:
    id: string
    title: string
    url: string
    content: string (markdown)
    last_updated: datetime
    author: string
  children: list[page] (recursive)
  breadcrumb: list[{id, title}]

Example:
  Input: get_page_with_children(page_id="123456", max_depth=1)
  Output: {
    page: {id: "123456", title: "Kafka Setup", content: "# Kafka Setup\n..."},
    children: [{id: "123457", title: "Producer Config", ...}],
    breadcrumb: [{id: "111", title: "Engineering"}]
  }
```

#### `get_page_section`

```yaml
Name: get_page_section
Description: Extract a specific section from a page by heading
Parameters:
  page_id: string (required) - Confluence page ID
  heading: string (required) - Section heading to extract
  include_subsections: bool (optional, default=true)
  
Returns:
  section_title: string
  content: string (markdown)
  subsections: list[string] - Child headings
  page_title: string
  page_url: string

Example:
  Input: get_page_section(page_id="123", heading="Configuration")
  Output: {
    section_title: "Configuration",
    content: "## Configuration\n\nTo configure the service...",
    subsections: ["Environment Variables", "YAML Config"],
    page_title: "Kafka Setup Guide"
  }
```

#### `search_code`

```yaml
Name: search_code
Description: Find pages containing code blocks, optionally filtered by language
Parameters:
  query: string (required) - Search query (context for code)
  language: string (optional) - Filter by language (python, java, bash, etc.)
  limit: int (optional, default=10)
  
Returns:
  results: list
    - page_id: string
    - title: string
    - url: string
    - code_snippet: string (first matching code block)
    - language: string
    - code_block_count: int
    - score: float

Example:
  Input: search_code(query="kafka producer", language="python")
  Output: [
    {
      page_id: "456",
      title: "Kafka Python Examples",
      code_snippet: "from kafka import KafkaProducer\n...",
      language: "python",
      code_block_count: 5,
      score: 0.91
    }
  ]
```

#### `search_images`

```yaml
Name: search_images
Description: Find pages containing images/diagrams matching a description
Parameters:
  query: string (required) - Description of image content
  limit: int (optional, default=10)
  
Returns:
  results: list
    - page_id: string
    - title: string
    - url: string
    - image_url: string
    - alt_text: string
    - caption: string
    - score: float

Example:
  Input: search_images(query="kafka architecture diagram")
  Output: [
    {
      page_id: "789",
      title: "Kafka Architecture Overview",
      image_url: "https://confluence.../architecture.png",
      alt_text: "Kafka cluster architecture",
      score: 0.87
    }
  ]
```

#### `get_related_pages`

```yaml
Name: get_related_pages
Description: Find pages related to a given page (linked or semantically similar)
Parameters:
  page_id: string (required) - Source page ID
  relation_type: enum (optional) - "linked" | "similar" | "both" (default)
  limit: int (optional, default=5)
  
Returns:
  linked_pages: list - Pages this page links to
    - page_id: string
    - title: string
    - url: string
    - link_text: string
  similar_pages: list - Semantically similar pages
    - page_id: string
    - title: string
    - url: string
    - similarity_score: float

Example:
  Input: get_related_pages(page_id="123", relation_type="both")
  Output: {
    linked_pages: [
      {page_id: "456", title: "Kafka Security", link_text: "security guide"}
    ],
    similar_pages: [
      {page_id: "789", title: "RabbitMQ Setup", similarity_score: 0.78}
    ]
  }
```

#### `get_page_tree`

```yaml
Name: get_page_tree
Description: Get the page hierarchy (ancestors and descendants)
Parameters:
  page_id: string (optional) - Starting page (defaults to space root)
  space_key: string (optional) - Space to explore
  max_depth: int (optional, default=3)
  
Returns:
  root: PageNode
    - id: string
    - title: string
    - url: string
    - children: list[PageNode] (recursive)
  total_pages: int

Example:
  Input: get_page_tree(space_key="DOCS", max_depth=2)
  Output: {
    root: {
      id: "1", title: "Documentation Home",
      children: [
        {id: "2", title: "Getting Started", children: [...]},
        {id: "3", title: "API Reference", children: [...]}
      ]
    },
    total_pages: 45
  }
```

#### `compare_pages`

```yaml
Name: compare_pages
Description: Compare two pages and show differences
Parameters:
  page_id_1: string (required) - First page ID
  page_id_2: string (required) - Second page ID
  compare_type: enum (optional) - "content" | "structure" | "full" (default)
  
Returns:
  page_1: {id, title, url, word_count}
  page_2: {id, title, url, word_count}
  similarity_score: float
  differences:
    - type: string (added | removed | modified)
    - section: string
    - description: string
  common_topics: list[string]

Example:
  Input: compare_pages(page_id_1="123", page_id_2="456")
  Output: {
    page_1: {title: "Kafka v1 Setup"},
    page_2: {title: "Kafka v2 Setup"},
    similarity_score: 0.72,
    differences: [
      {type: "modified", section: "Configuration", description: "New YAML format in v2"},
      {type: "added", section: "Docker Setup", description: "Only in v2"}
    ]
  }
```

#### `analyze_image`

```yaml
Name: analyze_image
Description: Describe and analyze an image embedded in a page
Parameters:
  page_id: string (required) - Page containing the image
  image_index: int (optional, default=0) - Which image (0-based)
  analysis_type: enum (optional) - "describe" | "extract_text" | "both"
  
Returns:
  image_url: string
  description: string - AI-generated description
  extracted_text: string - OCR text if present
  image_type: string - diagram, screenshot, photo, etc.
  elements: list[string] - Key elements identified

Example:
  Input: analyze_image(page_id="123", image_index=0)
  Output: {
    image_url: "https://confluence.../arch.png",
    description: "Architecture diagram showing 3 Kafka brokers connected to ZooKeeper",
    image_type: "diagram",
    elements: ["Kafka Broker x3", "ZooKeeper", "Producer", "Consumer"]
  }
```

---

## 7. LLM Integration

### 7.1 Provider Abstraction

```python
class LLMProvider(ABC):
    """Abstract interface for LLM providers."""
    
    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion."""
        pass
    
    @abstractmethod
    async def complete_streaming(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream completion chunks."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check provider availability."""
        pass
```

### 7.2 Supported Providers

| Provider | Use Case | Config |
|----------|----------|--------|
| **LM Studio** | Local development | `http://localhost:1234/v1` |
| **GitHub Copilot** | Production (enterprise) | GitHub token |
| **OpenAI** | Fallback | API key |
| **Azure OpenAI** | Enterprise production | Azure endpoint |

### 7.3 Answer Modes

```
┌─────────────────────────────────────────────────────────────────┐
│                     ANSWER MODES                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ MODE 1: EXTRACTIVE                                      │   │
│  │ • Uses pre-stored TF-IDF extracted sentences            │   │
│  │ • No LLM call required                                  │   │
│  │ • Latency: 100ms                                        │   │
│  │ • Quality: ⭐⭐⭐                                        │   │
│  │ • Hallucination risk: ZERO                              │   │
│  │ • Best for: Quick lookups, simple facts                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ MODE 2: AI                                              │   │
│  │ • Full LLM synthesis from live content                  │   │
│  │ • Fetches and processes latest source docs              │   │
│  │ • Latency: 3000ms                                       │   │
│  │ • Quality: ⭐⭐⭐⭐⭐                                     │   │
│  │ • Hallucination risk: LOW (grounding validation)        │   │
│  │ • Best for: Complex questions, detailed explanations    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ MODE 3: HYBRID (DEFAULT)                                │   │
│  │ • Extractive summary as base context                    │   │
│  │ • LLM refines for coherence and completeness            │   │
│  │ • Latency: 1500ms                                       │   │
│  │ • Quality: ⭐⭐⭐⭐                                       │   │
│  │ • Hallucination risk: VERY LOW                          │   │
│  │ • Best for: Most queries (balanced)                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.4 LLM Prompt Templates

#### Summary Generation Prompt

```python
SUMMARY_PROMPT = """Summarize the following Confluence documentation page in 2-3 sentences.
Focus on:
- What the page is about (purpose)
- Key topics or concepts covered
- Target audience if apparent

Be concise and informative. Do not include phrases like "This page" or "This document".

Page Title: {title}
Page Content:
{content}

Summary:"""
```

#### Question Answering Prompt (AI Mode)

```python
QA_PROMPT = """You are a helpful documentation assistant. Answer the user's question based ONLY on the provided source documents.

RULES:
1. Only use information from the provided sources
2. If the answer isn't in the sources, say "I don't have information about that in the documentation"
3. Include specific details like code examples, configuration values when relevant
4. Cite sources using [Source: Page Title] format
5. Keep your answer concise but complete

SOURCES:
{sources}

USER QUESTION: {question}

ANSWER:"""
```

#### Hybrid Mode Prompt (Refinement)

```python
HYBRID_REFINE_PROMPT = """Refine the following extractive answer to make it more coherent and complete.
Keep all factual information intact. Do not add information not present in the original.
Make the language flow naturally while preserving accuracy.

Extractive Answer:
{extractive_answer}

Original Sources:
{source_titles}

Refined Answer:"""
```

---

## 8. Qdrant Schema

### 8.1 Collections

| Collection | Vectors | Purpose |
|------------|---------|---------|
| `confluence_pages` | Dense (768) + Sparse | Main search |
| `confluence_images` | CLIP (512) | Image search |
| `confluence_code` | CodeBERT (768) | Code search |

### 8.2 Page Payload Fields

| Category | Fields | Purpose |
|----------|--------|---------|
| **Core** | page_id, title, url, space_key, space_name | Identification |
| **Content** | extractive_summary, llm_summary, has_llm_summary, word_count | Fast answers + Hybrid mode |
| **Hierarchy** | breadcrumb, parent_id, **children_ids**, depth, children_count | Navigation (bidirectional) |
| **Metadata** | labels, author, author_id, created_at, updated_at, version | Filtering |
| **Analysis** | has_code, code_languages, code_block_count, has_images, image_count, has_tables, heading_structure, internal_links_count | Smart search |
| **Scoring** | importance_score (0-3.0), importance_classification, importance_signals | Ranking |
| **Sync** | indexed_at, content_hash | Incremental updates |

### 8.3 Importance Scoring Algorithm

```
Score = Structural + Hierarchy + Labels + Content

Structural (max 1.5):
  • children_count × 0.1 (hub pages)
  • internal_links × 0.05 (reference docs)

Hierarchy (max 0.5):
  • depth ≤ 1: +0.5
  • depth = 2: +0.3
  • depth > 2: +0.0

Labels (max 1.0):
  • Matches: overview, guide, architecture, tutorial, etc.
  • Each match: +0.25

Content (max 0.8):
  • has_code: +0.3
  • has_images: +0.3
  • word_count > 500: +0.2

Classification:
  HIGH (≥2.0): Hub pages, architecture docs
  MEDIUM (≥1.0): Technical docs
  LOW (<1.0): Meeting notes, simple pages
```

---

## 9. Performance Analysis

### 9.1 Pipeline Performance

| Scale | Text Only | Multi-Modal | GPU Speedup |
|-------|-----------|-------------|-------------|
| 1,000 pages | 3-4 min | 8-10 min | 2-3x |
| 10,000 pages | 30-40 min | 80-100 min | 3-5x |

**Bottleneck:** Embedding generation (Phase 5) - GPU recommended

### 9.2 Query Latency Breakdown

```
┌─────────────────────────────────────────────────────────────────┐
│ ask_question LATENCY (Total: 3.2s)                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Phase 1: Query Understanding        20ms    ▓░░░░░░░░░░  0.6%  │
│ Phase 2: Semantic Search           200ms    ▓▓░░░░░░░░░  6.3%  │
│ Phase 3: Content Fetch            1200ms    ▓▓▓▓▓▓▓░░░░ 37.5%  │ ← BOTTLENECK
│ Phase 4: Context Building           50ms    ▓░░░░░░░░░░  1.6%  │
│ Phase 5: Answer Generation        1550ms    ▓▓▓▓▓▓▓▓░░░ 48.4%  │ ← BOTTLENECK
│ Phase 6: Grounding Validation      100ms    ▓░░░░░░░░░░  3.1%  │
│ Phase 7: Response Formatting        50ms    ▓░░░░░░░░░░  1.6%  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

OPTIMIZATION STRATEGIES:
• Phase 3: Parallel fetching, HTTP/2, connection pooling
• Phase 5: Use HYBRID mode instead of AI mode
• Caching: Cache frequent queries and fetched content
```

### 9.3 Caching Strategy

| Layer | TTL | Max Size | Purpose |
|-------|-----|----------|---------|
| L1: Query Cache | 5 min | 1000 | Repeated questions |
| L2: Content Cache | 15 min | 1 GB | Live-fetched pages |
| L3: Embedding Cache | 30 min | 500 | Query embeddings |

---

## 10. Resilience & Error Handling

### 10.1 Circuit Breaker Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                    CIRCUIT BREAKER STATES                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CLOSED ──────────────▶ OPEN ──────────────▶ HALF_OPEN         │
│    │                      │                      │              │
│    │  5 failures          │  30s timeout         │  1 success   │
│    │  in 1 minute         │                      │              │
│    │                      ▼                      ▼              │
│    │              Use Fallback              Test Request        │
│    │                                             │              │
│    │◀────────────────────────────────────────────┘              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 Fallback Strategies

| Failure | Detection | Fallback |
|---------|-----------|----------|
| Confluence down | HTTP 5xx, timeout | Serve from cache + warning |
| LLM down | API error, timeout | Extractive mode (no LLM) |
| Qdrant down | Connection refused | Full-text via Confluence API |
| Rate limited | HTTP 429 | Exponential backoff + queue |

### 10.3 Error Codes & Messages

| Code | Name | Description | User Message |
|------|------|-------------|--------------|
| `ERR_CONFLUENCE_UNAVAILABLE` | Confluence Down | Cannot reach Confluence API | "Confluence is temporarily unavailable. Showing cached results." |
| `ERR_CONFLUENCE_AUTH` | Auth Failed | PAT invalid or expired | "Authentication failed. Please check Confluence credentials." |
| `ERR_CONFLUENCE_FORBIDDEN` | Access Denied | User lacks page permission | "You don't have access to this page." |
| `ERR_CONFLUENCE_RATE_LIMIT` | Rate Limited | 429 from Confluence | "Too many requests. Retrying in {n} seconds..." |
| `ERR_QDRANT_UNAVAILABLE` | Qdrant Down | Cannot connect to Qdrant | "Search index unavailable. Using Confluence search." |
| `ERR_LLM_UNAVAILABLE` | LLM Down | LLM provider not responding | "AI synthesis unavailable. Returning extractive answer." |
| `ERR_LLM_TIMEOUT` | LLM Timeout | Response took >30s | "Response generation timed out. Try a simpler question." |
| `ERR_NO_RESULTS` | No Results | Search returned empty | "No matching pages found. Try different keywords." |
| `ERR_PAGE_NOT_FOUND` | Page Not Found | Invalid page_id | "Page not found. It may have been deleted." |
| `ERR_INVALID_INPUT` | Bad Input | Malformed parameters | "Invalid request: {validation_error}" |

### 10.4 Retry Configuration

```python
RETRY_CONFIG = {
    "confluence": {
        "max_retries": 3,
        "base_delay_ms": 1000,
        "max_delay_ms": 30000,
        "exponential_base": 2,
        "retry_on": [429, 500, 502, 503, 504]
    },
    "qdrant": {
        "max_retries": 2,
        "base_delay_ms": 500,
        "max_delay_ms": 5000
    },
    "llm": {
        "max_retries": 2,
        "base_delay_ms": 2000,
        "timeout_ms": 30000
    }
}
```

---

## 11. Security

### 11.1 Authentication Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    SECURITY LAYERS                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Client  │───▶│ MCP Server   │───▶│ Confluence API       │  │
│  │          │    │              │    │                      │  │
│  │ Claude   │    │ • Validate   │    │ • PAT Authentication │  │
│  │ Copilot  │    │   request    │    │ • ACL enforcement    │  │
│  │ LM Studio│    │ • Rate limit │    │                      │  │
│  └──────────┘    └──────────────┘    └──────────────────────┘  │
│                                                                 │
│  Secrets: Environment variables, never logged                   │
│  Transport: HTTPS for SSE, secure STDIO                         │
│  Rate Limits: Per-client request limits                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 11.2 Security Measures

| Layer | Protection |
|-------|------------|
| Transport | HTTPS for SSE, secure STDIO |
| Confluence PAT | Stored in env vars, never logged |
| Rate Limiting | Per-client request limits |
| Input Validation | Query sanitization, length limits |
| Page Permissions | Respect Confluence ACLs |

### 11.3 Rate Limiting Configuration

```python
RATE_LIMITS = {
    # Per-client limits (identified by client_id header)
    "default": {
        "requests_per_minute": 60,
        "requests_per_hour": 1000,
        "concurrent_requests": 5
    },
    
    # Tool-specific limits (more expensive operations)
    "ask_question": {
        "requests_per_minute": 20,   # LLM calls are expensive
        "burst_limit": 5
    },
    "get_page_with_children": {
        "requests_per_minute": 30,   # Live API calls
        "burst_limit": 10
    },
    "search_pages": {
        "requests_per_minute": 60,   # Cached/fast
        "burst_limit": 20
    },
    
    # Global limits (protect Confluence API)
    "confluence_api": {
        "requests_per_minute": 100,  # Respect Confluence limits
        "concurrent_requests": 10
    }
}

# Response on rate limit exceeded
# HTTP 429 with Retry-After header
```

---

## 12. Deployment Plan

### 12.1 File Structure

```
mcp_server/
├── __init__.py
├── server.py                    # FastMCP entry point
├── config.py                    # Pydantic settings
│
├── pipeline/                    # Data ingestion
│   ├── discovery.py
│   ├── fetcher.py
│   ├── parser.py
│   ├── analyzer.py
│   ├── importance_scorer.py
│   ├── embedder.py
│   ├── indexer.py
│   └── run_pipeline.py
│
├── tools/                       # MCP tools
│   ├── search_pages.py
│   ├── get_page.py
│   ├── ask_question.py
│   └── ...
│
├── services/                    # Business logic
│   ├── search_service.py
│   ├── fetch_service.py
│   ├── rag_service.py
│   ├── grounding_validator.py
│   └── cache_service.py
│
├── llm/                         # LLM abstraction
│   ├── base_provider.py
│   ├── lm_studio_provider.py
│   └── openai_provider.py
│
└── models/                      # Pydantic schemas
    └── ...
```

### 12.2 Quick Start Commands

```bash
# 1. Run data pipeline (one-time)
python -m mcp_server.pipeline.run_pipeline --full

# 2. Start MCP server (SSE for LM Studio)
python -m mcp_server.server --transport sse --port 8080

# 3. Start MCP server (STDIO for Claude Desktop)
python -m mcp_server.server --transport stdio

# 4. Run incremental sync
python -m mcp_server.pipeline.run_pipeline --incremental

# 5. Health check
curl http://localhost:8080/health
```

### 12.3 Client Configuration

**Claude Desktop** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "confluence": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "C:/path/to/enterprise_confluence_ai"
    }
  }
}
```

**LM Studio**:
```json
{
  "mcp": {
    "servers": [{
      "name": "confluence",
      "transport": "sse",
      "url": "http://localhost:8080/sse"
    }]
  }
}
```

### 12.4 Environment Variables (.env Template)

```bash
# ═══════════════════════════════════════════════════════════════
# CONFLUENCE CONFIGURATION
# ═══════════════════════════════════════════════════════════════
CONFLUENCE_BASE_URL=https://your-company.atlassian.net/wiki
CONFLUENCE_USERNAME=your-email@company.com
CONFLUENCE_API_TOKEN=your-personal-access-token
CONFLUENCE_SPACE_KEY=DOCS

# ═══════════════════════════════════════════════════════════════
# QDRANT CONFIGURATION
# ═══════════════════════════════════════════════════════════════
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=confluence_pages
QDRANT_API_KEY=            # Optional, for Qdrant Cloud

# ═══════════════════════════════════════════════════════════════
# LLM CONFIGURATION
# ═══════════════════════════════════════════════════════════════
LLM_PROVIDER=lm_studio     # lm_studio | openai | azure
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=               # Required for OpenAI/Azure
LLM_MODEL=qwen2.5-7b-instruct
LLM_TIMEOUT_MS=30000

# ═══════════════════════════════════════════════════════════════
# EMBEDDING MODELS
# ═══════════════════════════════════════════════════════════════
EMBEDDING_MODEL_DENSE=BAAI/bge-base-en-v1.5
EMBEDDING_MODEL_SPARSE=prithivida/Splade_PP_en_v1
EMBEDDING_BATCH_SIZE=32

# ═══════════════════════════════════════════════════════════════
# MCP SERVER
# ═══════════════════════════════════════════════════════════════
MCP_TRANSPORT=sse          # sse | stdio
MCP_PORT=8080
MCP_HOST=0.0.0.0

# ═══════════════════════════════════════════════════════════════
# PIPELINE SETTINGS
# ═══════════════════════════════════════════════════════════════
MAX_PAGES_TO_CRAWL=3000
PIPELINE_CONCURRENCY=10
ENABLE_LLM_SUMMARIES=true  # Set false to skip Phase 3 LLM

# ═══════════════════════════════════════════════════════════════
# CACHING
# ═══════════════════════════════════════════════════════════════
CACHE_ENABLED=true
CACHE_TTL_QUERY_SECONDS=300
CACHE_TTL_CONTENT_SECONDS=900
CACHE_MAX_SIZE_MB=1024

# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════
LOG_LEVEL=INFO             # DEBUG | INFO | WARNING | ERROR
LOG_FORMAT=json            # json | text
```

---

## 13. Testing Strategy

### 13.1 Test Pyramid

```
                    ┌─────────┐
                    │   E2E   │  ← Browser tests, MCP client tests
                    │  (10%)  │     ~10 tests, run on deploy
                    ├─────────┤
                    │  Integ  │  ← API tests, Qdrant tests
                    │  (30%)  │     ~50 tests, run on PR
                    ├─────────┤
                    │  Unit   │  ← Pure function tests
                    │  (60%)  │     ~200 tests, run always
                    └─────────┘
```

### 13.2 Unit Tests

| Module | Key Tests | Framework |
|--------|-----------|-----------|
| `parser.py` | HTML→Markdown conversion, code block extraction | pytest |
| `importance_scorer.py` | Score calculation, classification | pytest |
| `summarizer.py` | Extractive summary generation | pytest |
| `grounding_validator.py` | Citation matching, score calculation | pytest |

```python
# Example: test_importance_scorer.py
def test_hub_page_gets_high_score():
    page = PageData(children_count=15, depth=1, labels=["architecture"])
    score, signals = calculate_importance(page)
    assert score >= 2.0
    assert signals["structural"] >= 1.0

def test_meeting_notes_get_low_score():
    page = PageData(children_count=0, depth=5, labels=["meeting"])
    score, signals = calculate_importance(page)
    assert score < 1.0
```

### 13.3 Integration Tests

| Test Suite | Purpose | Dependencies |
|------------|---------|--------------|
| `test_confluence_api.py` | API connectivity, auth | Live Confluence |
| `test_qdrant_operations.py` | CRUD, search, filters | Qdrant container |
| `test_embedding_pipeline.py` | Full pipeline flow | fastembed models |
| `test_mcp_tools.py` | Tool execution, response format | All services |

```bash
# Run integration tests (requires services)
pytest tests/integration/ --run-integration

# With Docker Compose
docker-compose -f docker-compose.test.yml up -d
pytest tests/integration/
```

### 13.4 E2E Tests

| Scenario | Steps | Expected |
|----------|-------|----------|
| **Search Flow** | Query → Results → Click page | Page content returned |
| **Ask Question** | Question → Answer → Citations | Valid answer with sources |
| **Sync Pipeline** | Run pipeline → Verify Qdrant | All pages indexed |

```python
# Example: test_e2e_search.py
async def test_search_returns_relevant_results():
    async with MCPClient("http://localhost:8080") as client:
        result = await client.call_tool("search_pages", {"query": "kafka"})
        assert len(result["results"]) > 0
        assert "kafka" in result["results"][0]["title"].lower()
```

### 13.5 Test Commands

```bash
# Unit tests (fast, no deps)
pytest tests/unit/ -v

# All tests with coverage
pytest --cov=mcp_server --cov-report=html

# Specific tool tests
pytest tests/integration/test_mcp_tools.py -k "ask_question"
```

---

## 14. Observability

### 14.1 Structured Logging Format

```python
# Log format: JSON for machine parsing
{
    "timestamp": "2024-01-15T10:30:00.123Z",
    "level": "INFO",
    "logger": "mcp_server.tools.search_pages",
    "message": "Search completed",
    "context": {
        "request_id": "req_abc123",
        "tool_name": "search_pages",
        "query": "kafka authentication",
        "result_count": 5,
        "latency_ms": 187,
        "client_id": "claude_desktop"
    }
}
```

### 14.2 Log Levels

| Level | When to Use | Example |
|-------|-------------|---------|
| **DEBUG** | Detailed tracing | "Embedding vector generated: 768 dims" |
| **INFO** | Normal operations | "Search completed: 5 results in 187ms" |
| **WARNING** | Degraded but working | "LLM timeout, falling back to extractive" |
| **ERROR** | Operation failed | "Confluence API returned 503" |

### 14.3 Key Metrics (Prometheus-style)

```python
# Counter metrics
mcp_tool_calls_total{tool="search_pages", status="success"}
mcp_tool_calls_total{tool="ask_question", status="error"}

# Histogram metrics
mcp_tool_latency_seconds{tool="search_pages", quantile="0.95"}
mcp_tool_latency_seconds{tool="ask_question", quantile="0.99"}

# Gauge metrics
mcp_active_requests{client="claude_desktop"}
qdrant_collection_points{collection="confluence_pages"}
confluence_api_rate_limit_remaining
```

### 14.4 Tracing

```python
# Trace context propagation
{
    "trace_id": "abc123",
    "span_id": "def456",
    "parent_span_id": "xyz789",
    "operation": "ask_question",
    "spans": [
        {"name": "query_understanding", "duration_ms": 20},
        {"name": "semantic_search", "duration_ms": 200},
        {"name": "content_fetch", "duration_ms": 1200},
        {"name": "answer_synthesis", "duration_ms": 1550}
    ]
}
```

### 14.5 Health Check Endpoint

```python
GET /health

Response:
{
    "status": "healthy",  # healthy | degraded | unhealthy
    "version": "1.0.0",
    "uptime_seconds": 3600,
    "checks": {
        "qdrant": {"status": "ok", "latency_ms": 5},
        "confluence": {"status": "ok", "latency_ms": 150},
        "llm": {"status": "ok", "latency_ms": 200}
    }
}
```

---

## 15. Roadmap

### Phase 1: Core Implementation (Current)

- [x] Architecture design
- [x] Documentation (PDR)
- [ ] Data pipeline implementation
- [ ] MCP server implementation
- [ ] Core tools (search, get_page, ask_question)
- [ ] Basic client integration

### Phase 2: Advanced Features

- [ ] Multi-modal search (images, code)
- [ ] Microsoft Teams integration
- [ ] Webhook-based real-time sync
- [ ] Advanced caching layer
- [ ] Multi-space support

### Phase 3: Production Hardening

- [ ] Full observability (Prometheus, structured logging)
- [ ] Load testing
- [ ] Security audit
- [ ] Performance optimization
- [ ] Documentation and training

---

## 16. Planned Improvements ⭐ NEW

This section documents architectural improvements identified for Phase 2+.

### 16.1 Semantic Chunking Strategy

**Current Gap:** Pages parsed HTML → Markdown without chunking. Large pages (50K+ tokens) exceed LLM context.

**Solution:** Smart chunking with hierarchy preservation.

```python
from dataclasses import dataclass
from typing import List, Literal

@dataclass
class Chunk:
    content: str           # Markdown content
    token_count: int
    heading_path: List[str]  # ["Engineering", "Kafka", "Authentication"]
    chunk_type: Literal["overview", "detail", "code", "procedure"]
    importance_boost: float  # Prioritize overviews, procedures

def chunk_page(markdown: str, max_tokens: int = 1500) -> List[Chunk]:
    """
    Preserve semantic boundaries:
    1. Never split code blocks
    2. Never split procedures (numbered lists)
    3. Keep headings with their content
    4. Prioritize "Overview" and "Configuration" sections
    """
```

**Qdrant Storage with Chunks:**
```python
# Store chunks as separate points with shared parent_id
payload = {
    "page_id": "123",
    "chunk_index": 0,
    "chunk_type": "overview",
    "heading_path": ["Kafka", "Authentication"],
    "parent_importance_score": 2.4,  # Inherit from page
}
```

### 16.2 Query Intent Routing

**Current Gap:** Single `ask_question` tool processes all queries identically (3.2s).

**Solution:** Route queries based on intent for optimized paths.

```python
from enum import Enum

class QueryIntent(Enum):
    FACTUAL_LOOKUP = "factual"      # "What port does Kafka use?"
    NAVIGATION = "nav"              # "Find the auth guide"
    COMPARISON = "compare"          # "Compare v1 and v2"
    SYNTHESIS = "synthesis"         # "How do I secure Kafka?"
    TROUBLESHOOTING = "troubleshoot" # "Why is Kafka down?"

ROUTING_TABLE = {
    QueryIntent.FACTUAL_LOOKUP: {
        "tools": ["search_pages"],  # Skip live fetch
        "mode": "extractive",       # No LLM (return snippets)
        "max_latency_ms": 500
    },
    QueryIntent.NAVIGATION: {
        "tools": ["search_pages", "get_page_tree"],
        "mode": "structured",
        "max_latency_ms": 800
    },
    QueryIntent.SYNTHESIS: {
        "tools": ["search_pages", "get_page_with_children"],
        "mode": "hybrid",           # Full RAG
        "max_latency_ms": 5000
    }
}
```

**Expected Impact:**
| Query Type | Current | With Routing |
|------------|---------|--------------|
| Factual | 3.2s | **500ms** |
| Navigation | 3.2s | **800ms** |
| Synthesis | 3.2s | 3.2s |

### 16.3 Constrained Grounding (Real-time Validation)

**Current Gap:** Generate → Validate post-hoc → Wasted work if ungrounded.

**Solution:** Constrained generation with mandatory citations.

```python
GROUNDED_PROMPT = """Answer using ONLY the provided sources.
For each claim, cite the source immediately: [Source: Title]

If information isn't in sources, respond: 
"I don't have information about [topic] in the documentation."

Sources:
{formatted_sources}

Question: {question}

Rules:
1. Every sentence must cite a source
2. No external knowledge
3. If sources conflict, note the conflict"""

def validate_citations(answer: str, available_sources: List[str]) -> bool:
    """Lightweight check - citations are in answer format."""
    citations = extract_citations(answer)
    return all(c in available_sources for c in citations)
```

### 16.4 Implicit Feedback Loop

**Current Gap:** No mechanism to learn from user interactions.

**Solution:** Track implicit signals for continuous improvement.

```python
feedback_signals = {
    "query": "How does Kafka auth work?",
    "results_shown": ["page_123", "page_456"],
    "answer_generated": True,
    
    # Implicit signals (tracked without user action)
    "time_to_next_query_seconds": 45,  # Quick follow-up = bad answer
    "copied_text": "bootstrap.servers=...",  # Tracked via client
    "clicked_sources": ["page_123"],  # Which citations were clicked
    "answer_scroll_depth": 0.3,  # Did they read full answer?
    
    # Derived metrics
    "satisfaction_proxy": calculate_satisfaction(signals)
}

# Use for:
# 1. Boost pages that get clicked/copied → increase importance_score
# 2. Demote pages with high bounce rate
# 3. Fine-tune importance scores automatically
```

### 16.5 Pragmatic Multi-Modal Phasing

**Current Plan:** CLIP (images) + CodeBERT (code) + BGE (text) + Splade (sparse) all at once.

**Reality Check:** 80% of queries are text-based. Ship text search first.

**Revised Approach:**

```
Phase 1 (MVP): Text only
├── BGE-base (dense) ✅
├── Splade (sparse) ✅
└── Image search by alt-text only (no CLIP)

Phase 2 (If metrics show need): 
└── Add CLIP for top 20% of pages by importance_score

Phase 3 (If code search popular):
└── Add CodeBERT for pages with >5 code blocks
```

**Rationale:** Defer complexity until core works and data shows need.

### 16.6 Enhanced Error Handling

**Current Gap:** Generic error strings, fixed rate limits.

**Solution:** Structured errors with retry hints and adaptive limits.

```python
# Structured error codes
{
    "code": "CONFLUENCE_TIMEOUT",
    "message": "Confluence API did not respond",
    "retry_after_seconds": 30,
    "fallback_available": True,
    "fallback_source": "cached_summary"
}

# Degraded health state
{
    "status": "degraded",  # healthy | degraded | unhealthy
    "components": {
        "qdrant": "ok",
        "confluence": "slow",  # New state beyond ok/error
        "llm": "ok"
    },
    "recommendation": "Using cached summaries, live fetch disabled"
}

# Adaptive rate limiting
class AdaptiveRateLimiter:
    def __init__(self):
        self.base_limit = 10  # requests/sec
        self.current_limit = 10
    
    def on_confluence_slow(self):
        self.current_limit = max(2, self.current_limit - 2)
    
    def on_confluence_recovered(self):
        self.current_limit = min(self.base_limit, self.current_limit + 1)
```

### 16.7 Content Hashing for Sync

**Current Gap:** Timestamp-based sync vulnerable to race conditions.

```
Problem:
- Page modified at 10:00:00, indexed at 10:00:05
- Page modified again at 10:00:03 (clock skew)
- Second change missed
```

**Solution:** Content hash comparison.

```python
import hashlib

def should_reindex(page_id: str, new_content: str) -> bool:
    new_hash = hashlib.sha256(new_content.encode()).hexdigest()[:16]
    
    existing = qdrant.retrieve(page_id)
    if not existing:
        return True  # New page
    
    old_hash = existing.payload.get("content_hash")
    if old_hash != new_hash:
        return True  # Content changed
    
    return False  # Same content, skip

# Store hash in payload
payload = {
    "page_id": "123",
    "content_hash": "a1b2c3d4e5f6g7h8",
    # ...rest of payload
}
```

### 16.8 Improvement Priority Matrix

| Improvement | Impact | Effort | Phase |
|-------------|--------|--------|-------|
| Query Intent Routing | 🔴 High | Medium | 2a |
| Semantic Chunking | 🔴 High | High | 2a |
| Content Hashing | 🟡 Medium | Low | 2a |
| Constrained Grounding | 🟡 Medium | Low | 2b |
| Enhanced Error Handling | 🟡 Medium | Medium | 2b |
| Feedback Loop | 🟢 Medium | High | 3 |
| Pragmatic Multi-Modal | 🟢 Low | - | Defer |

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **MCP** | Model Context Protocol - Standard for AI tool integration |
| **RAG** | Retrieval-Augmented Generation |
| **Skeletal RAG** | Our approach: index metadata, fetch content live |
| **Grounding** | Ensuring AI answers are based on source documents |
| **Dense Vector** | Fixed-size embeddings for semantic similarity |
| **Sparse Vector** | Keyword-based embeddings (like SPLADE) |
| **RRF** | Reciprocal Rank Fusion - hybrid search algorithm |

---

## Appendix B: References

- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [FastEmbed](https://github.com/qdrant/fastembed)
- [FlashRank Reranker](https://github.com/PrithivirajDamodaran/FlashRank)
- [LM Studio](https://lmstudio.ai/)
- [BGE Embeddings](https://huggingface.co/BAAI/bge-base-en-v1.5)

---

*Document Version: 1.2*  
*Last Updated: February 2026*  
*Changes: Added Planned Improvements (Sec 16) - Semantic Chunking, Query Routing, Constrained Grounding, Feedback Loop, Pragmatic Multi-Modal, Enhanced Error Handling, Content Hashing*
