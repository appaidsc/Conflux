# Product Requirements Document (PRD)
# Confluence MCP Server

**Project:** Enterprise Confluence AI - MCP Integration  
**Version:** 1.0  
**Date:** February 2026  
**Status:** Ready for Development

---

## 1. Overview

### 1.1 Product Vision

Build an **MCP (Model Context Protocol) server** that enables AI assistants to search, retrieve, and answer questions from Confluence documentation with **always-fresh content**.

### 1.2 Key Innovation: Skeletal RAG

| Aspect | Traditional RAG | Our Approach |
|--------|----------------|--------------|
| Indexed Content | Full documents | Titles + summaries only |
| Content Freshness | Stale after indexing | **Always current (live fetch)** |
| Vector DB Size | Large (GBs) | Small (MBs) |

---

## 2. Success Metrics

| Metric | Current State | Target | Priority |
|--------|---------------|--------|----------|
| Doc search time | 10-15 min | **< 3 seconds** | 🔴 P0 |
| Answer accuracy | N/A | **> 90% grounding score** | 🔴 P0 |
| Content freshness | Stale | **Real-time** | 🔴 P0 |
| Supported AI clients | 0 | **4+ clients** | 🟡 P1 |
| Daily time saved per dev | 0 | **30-60 min** | 🟡 P1 |

---

## 3. User Personas

### 3.1 Primary: Software Developer

**Goals:**
- Find documentation quickly without leaving IDE
- Get accurate answers with source citations
- Search code examples across all docs

**Pain Points:**
- Keyword search misses relevant pages
- Context switching between IDE and browser
- Stale cached answers

### 3.2 Secondary: New Team Member

**Goals:**
- Discover relevant documentation for onboarding
- Understand project architecture quickly

**Pain Points:**
- Doesn't know where to look
- Information scattered across multiple pages

---

## 4. Functional Requirements

### 4.1 Core Features (MVP - Phase 1)

| ID | Feature | Description | Priority |
|----|---------|-------------|----------|
| F1 | **Semantic Search** | Search pages by meaning, not just keywords | 🔴 P0 |
| F2 | **Page Retrieval** | Fetch full page content with children | 🔴 P0 |
| F3 | **Q&A with RAG** | Answer questions with source citations | 🔴 P0 |
| F4 | **Page Navigation** | Browse page hierarchy/tree | 🟡 P1 |

### 4.2 Enhanced Features (Phase 2)

| ID | Feature | Description | Priority |
|----|---------|-------------|----------|
| F5 | **Code Search** | Find pages containing code blocks | 🟡 P1 |
| F6 | **Image Search** | Find pages with diagrams | 🟢 P2 |
| F7 | **Related Pages** | Suggest similar/linked pages | 🟢 P2 |
| F8 | **Page Comparison** | Diff two pages | 🔵 P3 |

### 4.3 MCP Tools Specification

```
┌─────────────────────────────────────────────────────────────────┐
│                    MVP TOOLS (Phase 1)                          │
├─────────────────────────────────────────────────────────────────┤
│  1. search_pages(query, limit, space_filter)                    │
│     → Semantic search over indexed pages                        │
│     → Returns: page_id, title, snippet, score                   │
│                                                                 │
│  2. get_page_with_children(page_id, max_depth)                  │
│     → Fetch full content from live Confluence                   │
│     → Returns: markdown content + child pages                   │
│                                                                 │
│  3. ask_question(question, mode)                                │
│     → Full RAG pipeline: search → fetch → synthesize            │
│     → Returns: answer + citations + confidence                  │
│                                                                 │
│  4. get_page_tree(space_key)                                    │
│     → Get page hierarchy for navigation                         │
│     → Returns: tree structure                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Non-Functional Requirements

### 5.1 Performance

| Metric | Requirement |
|--------|-------------|
| Search latency | < 500ms |
| Page fetch | < 2s |
| Q&A response | < 5s |
| Concurrent users | 10+ |

### 5.2 Reliability

| Metric | Requirement |
|--------|-------------|
| Uptime | 99.5% |
| Error rate | < 1% |
| Graceful degradation | Fallback to cached content |

### 5.3 Security

| Requirement | Implementation |
|-------------|----------------|
| Authentication | Confluence PAT (Personal Access Token) |
| Data at rest | Qdrant local storage |
| Data in transit | HTTPS/TLS |
| No external APIs | All local (LM Studio) |

---

## 6. Technical Architecture

### 6.1 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       AI CLIENTS                                │
│   Claude Desktop | GitHub Copilot | LM Studio | VS Code        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ MCP Protocol (STDIO/SSE)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MCP SERVER                                  │
│   FastMCP (Python) | Port 8080                                  │
├─────────────────┬─────────────────┬─────────────────────────────┤
│ TOOLS           │ SERVICES        │ PROVIDERS                   │
│ search_pages    │ SearchService   │ Confluence API              │
│ get_page        │ FetchService    │ Qdrant                      │
│ ask_question    │ RAGService      │ LM Studio (LLM)             │
│ get_tree        │ EmbedService    │ GPU Embeddings              │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

### 6.2 Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **MCP Framework** | FastMCP | Lightweight, Python-native |
| **Vector DB** | Qdrant | Best hybrid search support |
| **Text Embeddings** | bge-base-en-v1.5 (768 dim) | SOTA quality, GPU accelerated |
| **Sparse Embeddings** | SPLADE | Keyword + semantic hybrid |
| **Code Embeddings** | CodeBERT (768 dim) | Specialized for code |
| **Image Embeddings** | CLIP (512 dim) | Diagram search |
| **Reranker** | FlashRank MiniLM-L-12 | Fast, local |
| **LLM** | LM Studio (local) | No API costs, private |
| **GPU** | NVIDIA RTX 2000 Ada | 8GB VRAM, CUDA 13.1 |

### 6.3 Data Pipeline

```
Confluence → Crawler → Parser → Embedder → Qdrant
                                    │
                                    └── Dense (768 dim)
                                    └── Sparse (SPLADE)
                                    └── Code (768 dim)
                                    └── Image (512 dim)
```

---

## 7. GPU Requirements

### 7.1 Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 2000 Ada |
| VRAM | 8 GB |
| CUDA | 13.1 (installed) |

### 7.2 Models & VRAM Usage

| Model | Purpose | VRAM |
|-------|---------|------|
| bge-base-en-v1.5 | Text embeddings | ~400 MB |
| Splade_PP_en_v1 | Sparse embeddings | ~500 MB |
| codebert-base | Code embeddings | ~450 MB |
| clip-vit-base | Image embeddings | ~550 MB |
| MiniLM-L-12-v2 | Reranker | ~100 MB |
| **Total** | | **~2.0 GB** ✅ |

> All models fit in 8GB VRAM with room for batching

---

## 8. Dependencies

### 8.1 Python Packages

```txt
# Core MCP
fastmcp>=0.3.0

# Vector Database
qdrant-client>=1.7.0

# Embeddings (GPU)
fastembed>=0.2.0
flashrank>=0.2.0
transformers>=4.30.0
sentence-transformers>=2.2.0

# GPU Support
torch>=2.1.0 (cu124)
onnxruntime-gpu>=1.16.0

# Utilities
httpx>=0.25.0
beautifulsoup4>=4.12.0
pydantic-settings>=2.0.0
Pillow>=10.0.0
```

### 8.2 Infrastructure

| Component | Required |
|-----------|----------|
| Qdrant | Docker container, port 6333 |
| LM Studio | Local, port 1234 |
| MongoDB | Optional (metadata backup) |

---

## 9. Milestones & Timeline

### Phase 1: MVP (Weeks 1-3)

| Week | Deliverable |
|------|-------------|
| 1 | GPU setup, model downloads, Qdrant collection |
| 2 | search_pages, get_page_with_children tools |
| 3 | ask_question tool, Claude Desktop integration |

**Exit Criteria:**
- [ ] Search returns relevant results in < 500ms
- [ ] Q&A answers have > 85% grounding score
- [ ] Works with Claude Desktop

### Phase 2a: Architectural Improvements (Weeks 4-5)

| Week | Deliverable |
|------|-------------|
| 4 | Query intent routing, semantic chunking |
| 5 | Content hashing sync, constrained grounding |

**Exit Criteria:**
- [ ] Simple queries (factual) return in <500ms
- [ ] Large pages chunked with hierarchy preserved
- [ ] Sync uses content hash (no race conditions)

### Phase 2b: Enhanced Features (Weeks 6-7)

| Week | Deliverable |
|------|-------------|
| 6 | Code search, related pages |
| 7 | Enhanced error handling, adaptive rate limiting |

**Exit Criteria:**
- [ ] Code blocks searchable
- [ ] Degraded health state supported

### Phase 3: Production (Weeks 8-9)

| Week | Deliverable |
|------|-------------|
| 8 | Testing, implicit feedback loop |
| 9 | Documentation, deployment |

**Exit Criteria:**
- [ ] 99.5% uptime over 1 week
- [ ] All tests passing
- [ ] Feedback signals collected

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Confluence API rate limiting | High | Batch requests, caching |
| GPU out of memory | Medium | Reduce batch size, queue requests |
| LLM quality issues | Medium | Grounding validation, fallback prompts |
| Model download blocked | Low | Manual download, local cache |

---

## 11. Out of Scope

| Item | Reason |
|------|--------|
| LangChain/LangGraph | MCP is simpler, more control |
| Cloud LLM APIs | Local LM Studio preferred |
| Real-time sync | Incremental sync is sufficient |
| Full agentic loops | Orchestrated RAG is better for Q&A |
| Mobile app | Desktop-first |

---

## 12. Acceptance Criteria

### Must Have (P0)
- [ ] Semantic search works with < 500ms latency
- [ ] Q&A answers have citations
- [ ] Content is always fresh (live fetch)
- [ ] Works with Claude Desktop

### Should Have (P1)
- [ ] Code block search
- [ ] Page tree navigation
- [ ] 4x speedup with GPU

### Nice to Have (P2)
- [ ] Image/diagram search
- [ ] Page comparison
- [ ] Response streaming

---

## 13. Related Documents

| Document | Purpose |
|----------|---------|
| [MCP_PDR.md](./MCP_PDR.md) | Technical design details |
| [GPU_SETUP_GUIDE.md](./GPU_SETUP_GUIDE.md) | GPU installation steps |
| [QDRANT_SCHEMA.md](./QDRANT_SCHEMA.md) | Vector database schema |

---

## 14. Approvals

| Role | Name | Date | Status |
|------|------|------|--------|
| Product Owner | | | ⏳ Pending |
| Tech Lead | | | ⏳ Pending |
| Security | | | ⏳ Pending |

---

*Document Version: 1.1*  
*Last Updated: February 2026*  
*Changes: Updated timeline to 9 weeks, added Phase 2a architectural improvements (query routing, chunking, grounding). See [MCP_PDR.md Section 16](./MCP_PDR.md#16-planned-improvements) for technical details.*
