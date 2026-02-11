# Confluence MCP Server - Complete Learning Guide

A beginner-friendly explanation of the data flow, storage, caching, and implementation.

---

## Part 1: The Big Picture

### What Are We Building?

```
┌─────────────────────────────────────────────────────────────────┐
│                         AI ASSISTANTS                           │
│   (Claude Desktop, GitHub Copilot, LM Studio, VS Code, etc.)    │
└──────────────────────────────┬──────────────────────────────────┘
                               │ MCP Protocol (SSE/STDIO)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MCP SERVER (Our Code)                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      10 MCP TOOLS                        │    │
│  │  search_pages, get_page, ask_question, search_code, etc.│    │
│  └─────────────────────────────────────────────────────────┘    │
│                               │                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      SERVICES LAYER                      │    │
│  │    SearchService, FetchService, RAGService, Cache        │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │    QDRANT    │   │  CONFLUENCE  │   │  LM STUDIO   │
    │ Vector DB    │   │   REST API   │   │  Local LLM   │
    │ (Indexed)    │   │  (Live Data) │   │  (Answers)   │
    └──────────────┘   └──────────────┘   └──────────────┘
```

### The Two Main Flows

1. **INGESTION FLOW** (Run once initially, then periodically)
   - Crawl Confluence → Extract data → Create embeddings → Store in Qdrant

2. **QUERY FLOW** (Every time user asks something)
   - User query → Search Qdrant → Fetch live content → Generate answer

---

## Part 2: Data Flow Explained

### Flow 1: Data Ingestion Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE (Stage 2)                         │
│                                                                          │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌────────┐ │
│  │DISCOVERY│───▶│ FETCHER │───▶│ PARSER  │───▶│ANALYZER │───▶│ SCORER │ │
│  │         │    │         │    │         │    │         │    │        │ │
│  │ Find    │    │ Get     │    │ HTML →  │    │ has_code│    │ 0-3    │ │
│  │ pages   │    │ content │    │ Markdown│    │ has_img │    │ score  │ │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └────────┘ │
│                                                                     │    │
│  ┌─────────┐    ┌─────────┐    ┌──────────────────────────────────────┘  │
│  │ INDEXER │◀───│EMBEDDER │◀───│ SUMMARIZER (Map-Reduce)     │           │
│  │         │    │         │    │                              │           │
│  │ Store   │    │ Vectors │    │ Stage 1: TF-IDF (fast)       │           │
│  │ Qdrant  │    │ 768-dim │    │ Stage 2: Chunk → Map → Reduce│           │
│  └─────────┘    └─────────┘    └──────────────────────────────┘           │
│                                                                          │
│  ┌─ BACKGROUND WORKER (runs after initial indexing) ─────────────────┐  │
│  │  Finds pages without LLM summary → Map-Reduce → Re-embed → Update │  │
│  │  Quality improves progressively over time                          │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

#### Step-by-Step Explanation:

**1. DISCOVERY (`discovery.py`)**
```python
# What it does: Finds all pages in your Confluence space
# Uses: CQL (Confluence Query Language)

# Example CQL query:
cql = "space=DOCS AND type=page ORDER BY lastModified DESC"

# Returns lightweight info:
PageSummary(page_id="12345", title="API Guide", version=5, ...)
```

**2. FETCHER (`fetcher.py`)**
```python
# What it does: Gets the full page content from Confluence API

# API call:
GET /rest/api/content/{page_id}?expand=body.storage,ancestors,labels

# Returns:
PageContent(
    html_content="<h1>Title</h1><p>Content...</p>",
    ancestors=[{id: "parent1", title: "Parent"}],
    labels=["api", "guide"],
    ...
)
```

**3. PARSER (`parser.py`)**
```python
# What it does: Converts Confluence HTML to clean Markdown
# Why: Markdown is easier to search and embed

# Before (Confluence HTML):
"<ac:structured-macro ac:name='code'><ac:parameter ac:name='language'>python</ac:parameter>..."

# After (Markdown):
"```python\nprint('hello')\n```"
```

**4. ANALYZER (`analyzer.py`)**
```python
# What it does: Extracts metadata about the content

# Output:
AnalysisResult(
    has_code=True,
    code_languages=["python", "java"],
    has_images=True,
    image_count=3,
    word_count=1500,
)
```

**5. IMPORTANCE SCORER (`importance_scorer.py`)**
```python
# What it does: Calculates how important a page is (0.0 - 3.0)
# Why: So important pages rank higher in search

# Formula:
score = structural + hierarchy + labels + content

# Example:
- Page with 10 children = +1.0 structural
- Top-level page = +0.5 hierarchy
- Has "overview" label = +0.25 labels
- Has code = +0.3 content
# Total = 2.05 → classification = "high"
```

**6. SUMMARIZER (`summarizer.py`) — Stage 2: Map-Reduce**
```python
# PROBLEM: Old summarizer only saw first 4000 chars of long pages
# SOLUTION: Map-Reduce summarization

# Stage 1 (fast, no LLM):
extractive = summarizer.extractive_summary(text)  # TF-IDF, ~10ms

# Stage 2 (Map-Reduce, background worker):
# Step 1: Condense huge tables (50 rows → 10 + count)
condensed = summarizer.condense_tables(text)

# Step 2: Split into 3500-char overlapping chunks
chunks = summarizer.chunk_text(condensed)  # e.g. 20000 chars → 6 chunks

# Step 3: MAP — parallel LLM summary for each chunk
chunk_summaries = await summarizer.map_summarize(chunks, title, llm)

# Step 4: REDUCE — synthesize into master summary
master = await summarizer.reduce_summarize(chunk_summaries, title, llm)

# Result: Vector DB now understands the END of long documents!
```

> **Edge Case — Huge Tables:** Pages with 50+ row tables get condensed
> before summarization. Header + first 10 rows + row count are kept.
> HTML tables over 1500 chars are replaced with a description.

**7. EMBEDDER (`embedder.py`)**
```python
# What it does: Converts text to vectors (numbers)
# Why: Computers compare numbers, not text

# Two types of embeddings:
# 1. DENSE (BGE model): 768 numbers representing meaning
# 2. SPARSE (SPLADE): Many numbers, most are zero, for keywords

# Example dense vector:
[0.123, -0.456, 0.789, ...] # 768 dimensions

# Example sparse vector:
{indices: [42, 156, 789], values: [0.8, 0.5, 0.3]}
```

**8. INDEXER (`indexer.py`)**
```python
# What it does: Stores everything in Qdrant vector database

# What gets stored:
PointStruct(
    id=hash(page_id),  # Unique identifier
    vector={
        "dense": [0.123, -0.456, ...],  # 768 dims
        "sparse": {indices: [...], values: [...]}
    },
    payload={
        "page_id": "12345",
        "title": "API Guide",
        "url": "https://...",
        "extractive_summary": "This page explains...",
        "importance_score": 2.05,
        "has_code": True,
        # ... 20+ more fields
    }
)
```

---

### Flow 2: Query/Search Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                             QUERY FLOW                                   │
│                                                                          │
│  User: "How do I authenticate with the API?"                            │
│                               │                                          │
│                               ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                        1. EMBED QUERY                              │ │
│  │  "How do I authenticate..." → [0.234, -0.567, ...] (768d)          │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                               │                                          │
│                               ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                      2. HYBRID SEARCH                              │ │
│  │  Search Qdrant with BOTH dense AND sparse vectors                  │ │
│  │  Dense: semantic meaning ("authenticate" ≈ "login" ≈ "sign in")    │ │
│  │  Sparse: exact keywords ("API", "authenticate")                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                               │                                          │
│                               ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                       3. RRF FUSION                                │ │
│  │  Combine results from both searches using Reciprocal Rank Fusion   │ │
│  │  RRF(d) = 1/(60+rank_dense) + 1/(60+rank_sparse)                   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                               │                                          │
│                               ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │              4. RERANK (uses LLM summary when available)           │ │
│  │  FlashRank re-scores using high-quality Map-Reduce summaries       │ │
│  │  "login" now matches "authentication" (semantic, not keyword)      │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                               │                                          │
│                               ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │              5. SMART FETCH (Stage 2 — Version Check)              │ │
│  │  Check page version first (~50ms lightweight call)                 │ │
│  │  If version unchanged → return cached instantly (0ms!)             │ │
│  │  If version changed → full fetch + parse + cache with new version  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                               │                                          │
│                               ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │          6. SECTION-LEVEL CONTEXT (Stage 2 — Smart RAG)            │ │
│  │  Split page into sections by headers                               │ │
│  │  Score EACH SECTION against the question using FlashRank           │ │
│  │  Inject only top 3 most relevant sections (not blind truncation)   │ │
│  │  → Can answer questions hidden in the footer of a huge page!       │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                               │                                          │
│                               ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                     7. GENERATE ANSWER                             │ │
│  │  Mode: extractive (fast) or AI (LLM) or hybrid                     │ │
│  │  Answer: "To authenticate, use a Personal Access Token..."        │ │
│  │  Sources: [Page: "API Auth Guide" > "OAuth Setup"]                │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Part 3: Data Storage Explained

### Where Data Lives

| Data | Storage | Purpose |
|------|---------|---------|
| Page summaries + metadata | **Qdrant** | Fast vector search |
| Full page content | **Confluence** (live) | Always fresh |
| ML models | **models_cache/** | Embeddings, reranking |
| Search results | **In-memory cache** | Speed up repeated queries |
| Page content | **In-memory cache** | Avoid repeated API calls |

### Qdrant Schema

```
Collection: confluence_pages
├── id: int64 (hash of page_id)
├── vectors:
│   ├── dense: float[768]    # BGE embedding
│   └── sparse: {indices, values}  # SPLADE embedding
└── payload:
    ├── page_id: string
    ├── title: string
    ├── url: string
    ├── space_key: string
    ├── extractive_summary: string
    ├── llm_summary: string (optional)
    ├── importance_score: float (0-3)
    ├── importance_classification: "high"|"medium"|"low"
    ├── has_code: bool
    ├── code_languages: string[]
    ├── has_images: bool
    ├── labels: string[]
    ├── breadcrumb: [{id, title}, ...]
    ├── children_count: int
    ├── word_count: int
    ├── updated_at: datetime
    ├── indexed_at: datetime
    └── content_hash: string (for change detection)
```

### Why Not Store Full Content?

**Skeletal RAG** = Store only "skeleton" (title + summary), fetch "body" live.

| Approach | Storage | Freshness | Speed |
|----------|---------|-----------|-------|
| Full content in Qdrant | 10+ GB | Stale | Fast |
| **Skeletal RAG** | ~100 MB | Always fresh | Medium |

---

## Part 4: Caching Explained (Stage 2 — Smart Version-Check Caching)

### What is Caching?

**Caching** = Storing data temporarily so you don't have to fetch it again.

### Stage 1 vs Stage 2 Caching

```
STAGE 1 (Blind TTL Cache):
Fetch → Store with TTL → Expires after 15 min → Fetch again
Problem: Re-fetches even if page hasn't changed!

STAGE 2 (Smart Version-Check Cache):
Check version (~50ms) → If same version → Return cached (0ms!)
                       → If version changed → Fetch fresh → Cache with version
Result: Never re-fetches unchanged content!
```

### Our Caching Strategy

We use **TTL + Version-Check** caching with two tiers:

```python
# cache_service.py

# TIER 1: Query Cache (short TTL)
_query_cache = TTLCache(maxsize=1000, ttl=300)  # 5 minutes
# For: Search results

# TIER 2: Content Cache (longer TTL + version check)
_content_cache = TTLCache(maxsize=500, ttl=900)  # 15 minutes
# For: Full page content — now with version awareness!
```

### How Smart Caching Works (Stage 2)

```python
# When you call get_page():

async def get_page(page_id):
    cache_key = f"page:{page_id}"
    
    # Step 1: Quick version check (~50ms, tiny API call)
    current_version = await fetcher.get_page_version(page_id)  # Lightweight!
    
    # Step 2: Check if cached version matches
    cached = cache.get_versioned(cache_key, current_version)
    if cached:
        return cached  # VERSION MATCH → 0ms, no fetch!
    
    # Step 3: Version changed or cache miss — full fetch
    page = await fetcher.fetch_page(page_id)  # Full API call (~500ms)
    
    # Step 4: Cache with version tag
    cache.set_versioned(cache_key, page, current_version)
    return page
```

### Version-Check Visualization

```
Time: 0s    - User A requests page 123 (v5) → MISS → Full fetch + Cache (v5)
Time: 30s   - User B requests page 123      → Version check: v5=v5 → HIT (0ms!)
Time: 120s  - Someone edits page 123 in Confluence → Now v6
Time: 150s  - User C requests page 123      → Version check: v5≠v6 → STALE!
                                              → Fresh fetch → Cache (v6)
Time: 200s  - User D requests page 123      → Version check: v6=v6 → HIT (0ms!)
```

---

## Part 5: Implementation From Scratch

### Step 1: Environment Setup

```bash
# 1. Create project
mkdir confluence-mcp && cd confluence-mcp

# 2. Create virtual environment
py -3.11 -m venv venv

# 3. Install uv (fast package manager)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 4. Install dependencies
uv pip install -r requirements.txt --python venv\Scripts\python.exe

# 5. Copy environment file
cp .env.example .env
# Edit .env with your Confluence credentials
```

### Step 2: Start Qdrant

```bash
# Using Docker
docker-compose up -d

# Verify it's running
curl http://localhost:6333/health
# Should return: {"status":"ok"}
```

### Step 3: Configure Confluence

Edit `.env`:
```env
CONFLUENCE_BASE_URL=https://your-company.atlassian.net/wiki
CONFLUENCE_USERNAME=your-email@company.com
CONFLUENCE_API_TOKEN=your-personal-access-token
CONFLUENCE_SPACE_KEY=DOCS
```

**Getting an API Token:**
1. Go to https://id.atlassian.com/manage-profile/security/api-tokens
2. Create API token
3. Copy and paste into `.env`

### Step 4: Run Ingestion Pipeline

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Run full pipeline
python -m mcp_server.pipeline.run_pipeline --full

# Watch the output:
# INFO: Starting full pipeline run...
# INFO: Indexed 10 pages...
# INFO: Indexed 20 pages...
# INFO: Pipeline complete: {'discovered': 150, 'indexed': 148, 'errors': 2}
```

### Step 5: Start MCP Server

```bash
# SSE transport (for web clients)
python -m mcp_server.server --transport sse --port 8080

# STDIO transport (for Claude Desktop)
python -m mcp_server.server --transport stdio
```

### Step 6: Connect AI Assistant

**Claude Desktop config** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "confluence": {
      "command": "python",
      "args": ["-m", "mcp_server.server", "--transport", "stdio"],
      "cwd": "C:\\Users\\you\\CONFLUENCE_MCP"
    }
  }
}
```

---

## Part 6: Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `Connection refused` | Qdrant not running | `docker-compose up -d` |
| `401 Unauthorized` | Wrong Confluence token | Regenerate API token |
| `No results` | Pipeline not run | Run `--full` pipeline |
| `Slow search` | No cache | Check cache is enabled |
| `Out of memory` | Large model | Use `bge-small` instead of `bge-base` |

### Useful Commands

```bash
# Check Qdrant collection
curl http://localhost:6333/collections/confluence_pages

# Run tests
pytest tests/ -v

# Check if imports work
python -c "from mcp_server.config import get_settings; print('OK')"
```
