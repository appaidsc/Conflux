# Qdrant Schema Documentation

**Vector Database Schema for Confluence MCP**

---

## Overview

This document defines the complete Qdrant schema for the Confluence MCP system, including collections, vector configurations, and payload structures.

---

## Collections

The system uses 3 Qdrant collections:

| Collection | Purpose | Vector Types | Estimated Size |
|------------|---------|--------------|----------------|
| `confluence_pages` | Main page index | Dense + Sparse | 1,000 pages ≈ 50MB |
| `confluence_images` | Image search (optional) | Dense (CLIP) | 5,000 images ≈ 30MB |
| `confluence_code` | Code search (optional) | Dense (CodeBERT) | 10,000 blocks ≈ 40MB |

---

## Collection 1: `confluence_pages` (Primary)

### Vector Configuration

```python
vectors_config = {
    "dense": VectorParams(
        size=768,                    # bge-base-en-v1.5
        distance=Distance.COSINE,
        on_disk=False               # Keep in RAM for speed
    ),
    "sparse": SparseVectorParams(
        modifier=Modifier.IDF,      # SPLADE vectors
        index=SparseIndexParams(
            on_disk=False
        )
    )
}
```

### Payload Schema

```python
@dataclass
class PagePayload:
    """Complete Qdrant payload for a Confluence page."""
    
    # ═══════════════════════════════════════════════════════════════
    # CORE IDENTIFIERS
    # ═══════════════════════════════════════════════════════════════
    page_id: str              # Confluence page ID (e.g., "123456789")
    title: str                # Page title (e.g., "Kafka Authentication Guide")
    url: str                  # Full Confluence URL
    space_key: str            # Space key (e.g., "DOCS", "ENGINEERING")
    space_name: str           # Space display name (e.g., "Documentation")
    
    # ═══════════════════════════════════════════════════════════════
    # CONTENT - Dual Summary Strategy (Hybrid)
    # ═══════════════════════════════════════════════════════════════
    extractive_summary: str   # TF-IDF extracted sentences (200-500 chars)
                              # Always generated - fast, free, no LLM
                              # Used for: fallback, fast mode, search snippets
    
    llm_summary: str | None   # LLM-generated summary (optional, higher quality)
                              # Generated if LLM available during indexing
                              # Used for: quality mode, detailed answers
    
    has_llm_summary: bool     # Flag to check if LLM summary exists
                              # Enables fallback logic
    
    word_count: int           # Total words in page content
                              # Used for importance scoring
    
    # ═══════════════════════════════════════════════════════════════
    # HIERARCHY - For Navigation & Context
    # ═══════════════════════════════════════════════════════════════
    parent_id: str | None     # Parent page ID (null if root)
    
    children_ids: list[str]   # Child page IDs for navigation
                              # e.g., ["333", "444", "555"]
                              # Enables "show children" functionality
    
    breadcrumb: list[dict]    # Ancestor chain for context
                              # Format: [{"id": "1", "title": "Engineering"},
                              #          {"id": "2", "title": "Kafka"}]
    
    depth: int                # Hierarchy depth (0 = root, 1 = child, etc.)
                              # Used for importance scoring
    
    children_count: int       # Number of child pages (len(children_ids))
                              # High count = hub page = higher importance
    
    # ═══════════════════════════════════════════════════════════════
    # METADATA - For Filtering & Display
    # ═══════════════════════════════════════════════════════════════
    labels: list[str]         # Confluence labels/tags
                              # e.g., ["kafka", "security", "guide", "tutorial"]
    
    author: str               # Page author display name
    author_id: str            # Confluence user ID
    
    created_at: str           # ISO 8601 timestamp (e.g., "2024-01-15T10:30:00Z")
    updated_at: str           # ISO 8601 timestamp
    
    version: int              # Confluence version number
                              # Used for incremental sync
    
    # ═══════════════════════════════════════════════════════════════
    # CONTENT ANALYSIS - For Smart Filtering
    # ═══════════════════════════════════════════════════════════════
    has_code: bool            # Contains code blocks?
    code_languages: list[str] # Languages detected (e.g., ["python", "bash", "java"])
    code_block_count: int     # Number of code blocks
    
    has_images: bool          # Contains images/diagrams?
    image_count: int          # Number of images
    
    has_tables: bool          # Contains tables?
    table_count: int          # Number of tables
    
    heading_structure: list[str]  # Top-level headings for TOC
                                  # e.g., ["Overview", "Setup", "Configuration"]
    
    internal_links_count: int     # Links to other Confluence pages
                                  # High count = reference doc = higher importance
    
    # ═══════════════════════════════════════════════════════════════
    # IMPORTANCE SCORING - For Result Ranking
    # ═══════════════════════════════════════════════════════════════
    importance_score: float   # Calculated score (0.0 - 3.0)
                              # HIGH (≥2.0): Hub pages, architecture docs
                              # MEDIUM (≥1.0): Technical docs
                              # LOW (<1.0): Meeting notes, simple pages
    
    importance_classification: str  # "high" | "medium" | "low"
    
    importance_signals: dict  # Breakdown of score components
                              # {
                              #   "structural": 0.8,
                              #   "hierarchy": 0.5,
                              #   "labels": 1.0,
                              #   "content": 0.6
                              # }
    
    # ═══════════════════════════════════════════════════════════════
    # CHUNKING - For Large Page Support (Phase 2)
    # ═══════════════════════════════════════════════════════════════
    is_chunk: bool             # True if this is a chunk, False if full page
                               # Default: False (for backward compatibility)
    
    chunk_index: int | None    # Chunk number within page (0, 1, 2, ...)
                               # Only set if is_chunk=True
    
    chunk_type: str | None     # "overview" | "detail" | "code" | "procedure"
                               # Used for prioritization
    
    heading_path: list[str]    # Heading hierarchy for chunk context
                               # e.g., ["Kafka", "Authentication", "SASL Setup"]
    
    parent_page_id: str | None # Original page_id if this is a chunk
                               # Enables chunk → page navigation
    
    # ═══════════════════════════════════════════════════════════════
    # SYNC TRACKING - For Incremental Updates
    # ═══════════════════════════════════════════════════════════════
    indexed_at: str           # When we last indexed this page (ISO 8601)
    content_hash: str         # SHA256[:16] hash of content for change detection
```

### Example Payload

```json
{
    "page_id": "123456789",
    "title": "Kafka Authentication Guide",
    "url": "https://confluence.example.com/pages/viewpage.action?pageId=123456789",
    "space_key": "DOCS",
    "space_name": "Engineering Documentation",
    
    "extractive_summary": "This guide explains how to configure SASL authentication for Kafka clusters. The recommended approach uses SASL/SCRAM with SSL encryption. Configure the broker properties file with security settings before deployment.",
    "llm_summary": "Comprehensive guide to Kafka SASL authentication covering SCRAM-SHA-256 setup, broker security configuration, client-side auth, and troubleshooting with code examples for Java and Python.",
    "has_llm_summary": true,
    "summary_sentence_count": 3,
    "word_count": 1850,
    
    "parent_id": "987654321",
    "breadcrumb": [
        {"id": "111", "title": "Engineering"},
        {"id": "222", "title": "Kafka"},
        {"id": "987654321", "title": "Security"}
    ],
    "depth": 3,
    "children_count": 5,
    
    "labels": ["kafka", "security", "authentication", "guide", "sasl"],
    "author": "John Doe",
    "author_id": "jdoe",
    "created_at": "2024-01-10T09:00:00Z",
    "updated_at": "2024-01-15T14:30:00Z",
    "version": 12,
    
    "has_code": true,
    "code_languages": ["java", "properties", "bash"],
    "code_block_count": 8,
    "has_images": true,
    "image_count": 3,
    "has_tables": true,
    "table_count": 2,
    "heading_structure": [
        "Overview",
        "Prerequisites",
        "SASL Configuration",
        "SSL Setup",
        "Testing",
        "Troubleshooting"
    ],
    "internal_links_count": 7,
    
    "importance_score": 2.4,
    "importance_classification": "high",
    "importance_signals": {
        "structural": 0.5,
        "hierarchy": 0.3,
        "labels": 1.0,
        "content": 0.6
    },
    
    "indexed_at": "2024-01-15T15:00:00Z",
    "content_hash": "a3f2b8c9d1e4f5a6b7c8d9e0f1a2b3c4"
}
```

---

## Collection 2: `confluence_images` (Optional)

### Vector Configuration

```python
vectors_config = {
    "clip": VectorParams(
        size=512,                    # CLIP-ViT-Base
        distance=Distance.COSINE,
        on_disk=True                # Larger collection, use disk
    )
}
```

### Payload Schema

```python
@dataclass
class ImagePayload:
    """Qdrant payload for searchable images."""
    
    # Identifiers
    image_id: str             # Unique ID: "{page_id}_img_{index}"
    page_id: str              # Parent page ID
    page_title: str           # Parent page title (for display)
    page_url: str             # Parent page URL
    
    # Image data
    image_url: str            # Full image URL
    alt_text: str             # Alt text from HTML
    caption: str | None       # Figure caption if present
    
    # Position
    section_heading: str      # Heading under which image appears
    position_index: int       # Order in page (0, 1, 2...)
    
    # Image metadata
    width: int | None         # Image width in pixels
    height: int | None        # Image height in pixels
    file_type: str            # "png", "jpg", "svg", etc.
    
    # Context
    surrounding_text: str     # Text before/after image (100 chars)
    
    # Sync
    indexed_at: str
```

### Example Payload

```json
{
    "image_id": "123456789_img_0",
    "page_id": "123456789",
    "page_title": "Kafka Architecture Overview",
    "page_url": "https://confluence.example.com/pages/123456789",
    
    "image_url": "https://confluence.example.com/download/attachments/123456789/kafka-arch.png",
    "alt_text": "Kafka cluster architecture diagram",
    "caption": "Figure 1: High-level Kafka architecture with producers and consumers",
    
    "section_heading": "Architecture",
    "position_index": 0,
    
    "width": 1200,
    "height": 800,
    "file_type": "png",
    
    "surrounding_text": "The following diagram shows the overall Kafka architecture with multiple brokers...",
    
    "indexed_at": "2024-01-15T15:00:00Z"
}
```

---

## Collection 3: `confluence_code` (Optional)

### Vector Configuration

```python
vectors_config = {
    "codebert": VectorParams(
        size=768,                    # CodeBERT
        distance=Distance.COSINE,
        on_disk=True
    )
}
```

### Payload Schema

```python
@dataclass
class CodePayload:
    """Qdrant payload for searchable code blocks."""
    
    # Identifiers
    code_id: str              # Unique ID: "{page_id}_code_{index}"
    page_id: str              # Parent page ID
    page_title: str           # Parent page title
    page_url: str             # Parent page URL
    
    # Code content
    code: str                 # Full code content
    language: str             # Language (python, java, bash, etc.)
    
    # Position
    section_heading: str      # Heading under which code appears
    position_index: int       # Order in page
    
    # Context
    description: str          # Text before code block (explanation)
    
    # Analysis
    line_count: int           # Number of lines
    has_imports: bool         # Contains import statements?
    is_complete: bool         # Looks like complete example vs snippet?
    
    # Sync
    indexed_at: str
```

### Example Payload

```json
{
    "code_id": "123456789_code_2",
    "page_id": "123456789",
    "page_title": "Kafka Producer Configuration",
    "page_url": "https://confluence.example.com/pages/123456789",
    
    "code": "from kafka import KafkaProducer\n\nproducer = KafkaProducer(\n    bootstrap_servers=['localhost:9092'],\n    security_protocol='SASL_SSL',\n    sasl_mechanism='SCRAM-SHA-256'\n)",
    "language": "python",
    
    "section_heading": "Python Example",
    "position_index": 2,
    
    "description": "Here's a complete Python example for connecting to a secured Kafka cluster:",
    
    "line_count": 7,
    "has_imports": true,
    "is_complete": true,
    
    "indexed_at": "2024-01-15T15:00:00Z"
}
```

---

## Importance Scoring Algorithm

### Score Calculation

```python
def calculate_importance_score(page: PageData) -> tuple[float, dict]:
    """
    Calculate importance score from 0.0 to 3.0
    
    Returns:
        (score, signals_breakdown)
    """
    signals = {}
    
    # ═══════════════════════════════════════════════════════
    # STRUCTURAL SIGNALS (max 1.5 points)
    # Hub pages with many children are important
    # ═══════════════════════════════════════════════════════
    structural = min(1.5, 
        page.children_count * 0.1 +           # Hub pages
        page.internal_links_count * 0.05      # Reference docs
    )
    signals["structural"] = structural
    
    # ═══════════════════════════════════════════════════════
    # HIERARCHY SIGNALS (max 0.5 points)
    # Top-level pages are more important
    # ═══════════════════════════════════════════════════════
    if page.depth <= 1:
        hierarchy = 0.5
    elif page.depth == 2:
        hierarchy = 0.3
    else:
        hierarchy = 0.0
    signals["hierarchy"] = hierarchy
    
    # ═══════════════════════════════════════════════════════
    # LABEL SIGNALS (max 1.0 points)
    # Certain labels indicate important content
    # ═══════════════════════════════════════════════════════
    important_labels = {
        "overview", "guide", "tutorial", "architecture",
        "getting-started", "reference", "api", "howto",
        "best-practices", "documentation"
    }
    matched_labels = set(page.labels) & important_labels
    label_score = min(1.0, len(matched_labels) * 0.25)
    signals["labels"] = label_score
    signals["matched_labels"] = list(matched_labels)
    
    # ═══════════════════════════════════════════════════════
    # CONTENT SIGNALS (max 0.8 points)
    # Rich content indicates valuable documentation
    # ═══════════════════════════════════════════════════════
    content = 0.0
    if page.has_code:
        content += 0.3
    if page.has_images:
        content += 0.3
    if page.word_count > 500:
        content += 0.2
    content = min(0.8, content)
    signals["content"] = content
    
    # ═══════════════════════════════════════════════════════
    # TOTAL SCORE
    # ═══════════════════════════════════════════════════════
    total = structural + hierarchy + label_score + content
    
    return total, signals


def classify_importance(score: float) -> str:
    """Classify score into categories."""
    if score >= 2.0:
        return "high"      # Hub pages, architecture docs
    elif score >= 1.0:
        return "medium"    # Technical docs with substance
    else:
        return "low"       # Meeting notes, simple pages
```

---

## Search Queries

### Basic Semantic Search

```python
# Search with hybrid (dense + sparse)
results = client.query_points(
    collection_name="confluence_pages",
    query=dense_vector,
    using="dense",
    limit=20,
    with_payload=True
)
```

### Filtered Search Examples

```python
# Pages with code in Python
filter = Filter(
    must=[
        FieldCondition(
            key="has_code",
            match=MatchValue(value=True)
        ),
        FieldCondition(
            key="code_languages",
            match=MatchAny(any=["python"])
        )
    ]
)

# High importance pages only
filter = Filter(
    must=[
        FieldCondition(
            key="importance_classification",
            match=MatchValue(value="high")
        )
    ]
)

# Pages in specific space
filter = Filter(
    must=[
        FieldCondition(
            key="space_key",
            match=MatchValue(value="ENGINEERING")
        )
    ]
)

# Pages updated in last 7 days
filter = Filter(
    must=[
        FieldCondition(
            key="updated_at",
            range=Range(gte="2024-01-08T00:00:00Z")
        )
    ]
)

# Pages with diagrams
filter = Filter(
    must=[
        FieldCondition(
            key="has_images",
            match=MatchValue(value=True)
        ),
        FieldCondition(
            key="image_count",
            range=Range(gte=2)
        )
    ]
)
```

---

## Index Configuration

### Create Collection Script

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, 
    SparseVectorParams, SparseIndexParams, Modifier
)

client = QdrantClient(host="localhost", port=6333)

# Main pages collection
client.create_collection(
    collection_name="confluence_pages",
    vectors_config={
        "dense": VectorParams(
            size=768,
            distance=Distance.COSINE
        )
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(
            modifier=Modifier.IDF
        )
    },
    # Payload indexing for fast filtering
    hnsw_config=HnswConfigDiff(
        m=16,
        ef_construct=100
    )
)

# Create payload indexes for common filters
for field in ["space_key", "has_code", "importance_classification", "labels"]:
    client.create_payload_index(
        collection_name="confluence_pages",
        field_name=field,
        field_schema="keyword"
    )

client.create_payload_index(
    collection_name="confluence_pages",
    field_name="importance_score",
    field_schema="float"
)

client.create_payload_index(
    collection_name="confluence_pages",
    field_name="updated_at",
    field_schema="datetime"
)
```

---

## Summary Table

| Field | Type | Purpose | Used In |
|-------|------|---------|---------|
| `page_id` | str | Unique identifier | All operations |
| `title` | str | Display & search | Search results |
| `url` | str | Link generation | Response links |
| `space_key` | str | Space filtering | Multi-space search |
| `extractive_summary` | str | Fast answers | Extractive/Hybrid mode |
| `llm_summary` | str\|None | Quality answers, reranking | AI mode, Stage 2 reranking |
| `has_llm_summary` | bool | Summary selection | Background worker tracking |
| `word_count` | int | Importance scoring | Score calculation |
| `breadcrumb` | list | Navigation context | Response display |
| `depth` | int | Importance scoring | Score calculation |
| `children_count` | int | Hub detection | Importance scoring |
| `labels` | list[str] | Topic filtering | Label-based search |
| `has_code` | bool | Content filtering | Code search |
| `code_languages` | list[str] | Language filtering | Language-specific search |
| `has_images` | bool | Content filtering | Image search |
| `importance_score` | float | Result ranking | Search reranking |
| `version` | int | Change detection | Incremental sync |
| `content_hash` | str | Change detection | Incremental sync |
