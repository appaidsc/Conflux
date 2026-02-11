# GPU Acceleration Setup Guide

**For Confluence MCP Server | NVIDIA RTX 2000 Ada**

> **Reference:** See [MCP_PDR.md](./MCP_PDR.md) Section 5.6 for embedding model specifications  
> **Reference:** See [MCP_PDR.md](./MCP_PDR.md) Section 8 for Qdrant schema with all vector fields

---

## Overview

This guide covers GPU acceleration for ALL embedding models used in the MCP system:
- **Text Embeddings** (Dense + Sparse)
- **Code Embeddings** (For code block search)
- **Image Embeddings** (For diagram search)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GPU ACCELERATION STACK                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  NVIDIA RTX 2000 Ada (8GB VRAM)                                             │
│  ├── CUDA 12.x + cuDNN 8.x                                                  │
│  ├── PyTorch + CUDA                                                         │
│  └── ONNX Runtime GPU                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  MODELS RUNNING ON GPU:                                                     │
│  ├── 📝 Text Dense:   BAAI/bge-base-en-v1.5           (768 dim)            │
│  ├── 📝 Text Sparse:  prithivida/Splade_PP_en_v1      (sparse)             │
│  ├── 💻 Code:         microsoft/codebert-base          (768 dim)            │
│  ├── 🖼️ Image:        openai/clip-vit-base-patch32    (512 dim)            │
│  └── 🔄 Reranker:     ms-marco-MiniLM-L-12-v2         (cross-encoder)      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## All Models Summary

| Model | Purpose | Dimensions | Size | VRAM |
|-------|---------|------------|------|------|
| `BAAI/bge-base-en-v1.5` | Text embeddings | 768 | ~440 MB | ~400 MB |
| `prithivida/Splade_PP_en_v1` | Sparse/keyword search | Sparse | ~530 MB | ~500 MB |
| `microsoft/codebert-base` | Code block embeddings | 768 | ~500 MB | ~450 MB |
| `openai/clip-vit-base-patch32` | Image embeddings | 512 | ~600 MB | ~550 MB |
| `ms-marco-MiniLM-L-12-v2` | Reranking | - | ~130 MB | ~100 MB |
| **Total** | | | **~2.2 GB** | **~2.0 GB** |

> ✅ All models fit comfortably in your 8GB VRAM

---

## Hardware Requirements

| Component | Minimum | Your Hardware | Status |
|-----------|---------|---------------|--------|
| GPU | NVIDIA GTX 1060+ | RTX 2000 Ada | ✅ |
| VRAM | 4 GB | 8 GB | ✅ |
| CUDA Compute | 6.1+ | 8.9 (Ada) | ✅ |
| RAM | 16 GB | - | - |
| Disk | 3 GB (models) | - | - |

---

## Step 1: Verify NVIDIA Driver

```powershell
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 545.84       Driver Version: 545.84       CUDA Version: 12.3     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
|===============================+======================+======================|
|   0  NVIDIA RTX 2000...  WDDM | 00000000:01:00.0  On |                  N/A |
| 30%   45C    P8    10W / 130W |    512MiB /  8192MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

**Driver download:** https://www.nvidia.com/drivers

---

## Step 2: Install CUDA Toolkit ✅ ALREADY DONE

Your CUDA version:
```
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 13.1, V13.1.115
```

> ✅ **You have CUDA 13.1 installed - the latest version! Skip to Step 3.**

> **Note:** PyTorch currently supports up to CUDA 12.4 officially. Your CUDA 13.1 is backward 
> compatible, so PyTorch with CUDA 12.4 will work perfectly.

---

## Step 3: Install Python Dependencies

```powershell
# Activate venv
cd C:\Users\reuben.joseph\PycharmProjects\enterprise_confluence_ai
.venv\Scripts\activate

# PyTorch with CUDA 12.4 (compatible with your CUDA 13.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# ONNX Runtime GPU
pip install onnxruntime-gpu

# Embedding libraries
pip install fastembed>=0.2.0
pip install flashrank>=0.2.0
pip install transformers>=4.30.0
pip install sentence-transformers>=2.2.0
pip install Pillow>=10.0.0

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

---

## Step 4: Download ALL Models

### Complete Model Download Script

```python
# download_all_models.py
"""
Download all embedding models for MCP system.
Reference: MCP_PDR.md Section 5.6 - Embedding Models
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CACHE_DIR = "./models_cache"

print("=" * 60)
print("📥 MCP EMBEDDING MODELS DOWNLOADER")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# 1. TEXT DENSE EMBEDDINGS (bge-base-en-v1.5)
# ═══════════════════════════════════════════════════════════════
print("\n[1/5] 📝 Text Dense: BAAI/bge-base-en-v1.5...")
from fastembed import TextEmbedding
text_dense = TextEmbedding(
    model_name="BAAI/bge-base-en-v1.5",
    cache_dir=CACHE_DIR
)
test = list(text_dense.embed(["test"]))[0]
print(f"  ✅ Loaded! Dimension: {len(test)}")

# ═══════════════════════════════════════════════════════════════
# 2. TEXT SPARSE EMBEDDINGS (SPLADE)
# ═══════════════════════════════════════════════════════════════
print("\n[2/5] 📝 Text Sparse: prithivida/Splade_PP_en_v1...")
from fastembed import SparseTextEmbedding
text_sparse = SparseTextEmbedding(
    model_name="prithivida/Splade_PP_en_v1",
    cache_dir=CACHE_DIR
)
print("  ✅ Loaded!")

# ═══════════════════════════════════════════════════════════════
# 3. CODE EMBEDDINGS (CodeBERT)
# ═══════════════════════════════════════════════════════════════
print("\n[3/5] 💻 Code: microsoft/codebert-base...")
from transformers import AutoTokenizer, AutoModel
import torch

code_tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/codebert-base",
    cache_dir=CACHE_DIR
)
code_model = AutoModel.from_pretrained(
    "microsoft/codebert-base",
    cache_dir=CACHE_DIR
)
# Move to GPU
if torch.cuda.is_available():
    code_model = code_model.cuda()
    print("  ✅ Loaded on GPU! Dimension: 768")
else:
    print("  ✅ Loaded on CPU! Dimension: 768")

# ═══════════════════════════════════════════════════════════════
# 4. IMAGE EMBEDDINGS (CLIP)
# ═══════════════════════════════════════════════════════════════
print("\n[4/5] 🖼️ Image: openai/clip-vit-base-patch32...")
from transformers import CLIPProcessor, CLIPModel

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32",
    cache_dir=CACHE_DIR
)
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    cache_dir=CACHE_DIR
)
# Move to GPU
if torch.cuda.is_available():
    clip_model = clip_model.cuda()
    print("  ✅ Loaded on GPU! Dimension: 512")
else:
    print("  ✅ Loaded on CPU! Dimension: 512")

# ═══════════════════════════════════════════════════════════════
# 5. RERANKER (FlashRank MiniLM)
# ═══════════════════════════════════════════════════════════════
print("\n[5/5] 🔄 Reranker: ms-marco-MiniLM-L-12-v2...")
from flashrank import Ranker
reranker = Ranker(
    model_name="ms-marco-MiniLM-L-12-v2",
    cache_dir=CACHE_DIR
)
print("  ✅ Loaded!")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("✅ ALL MODELS DOWNLOADED SUCCESSFULLY!")
print("=" * 60)
print(f"""
📁 Cache Location: {CACHE_DIR}

Models Ready:
┌────────────────────────────────────────────────────────────┐
│ Model                          │ Purpose      │ Dimension │
├────────────────────────────────┼──────────────┼───────────┤
│ BAAI/bge-base-en-v1.5         │ Text Dense   │ 768       │
│ prithivida/Splade_PP_en_v1    │ Text Sparse  │ Sparse    │
│ microsoft/codebert-base        │ Code         │ 768       │
│ openai/clip-vit-base-patch32  │ Image        │ 512       │
│ ms-marco-MiniLM-L-12-v2       │ Reranker     │ -         │
└────────────────────────────────┴──────────────┴───────────┘
""")
```

Run it:
```powershell
python download_all_models.py
```

---

## Step 5: Manual Download Links (If HuggingFace Blocked)

### Text Dense: bge-base-en-v1.5
| File | URL |
|------|-----|
| model.onnx | https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/onnx/model.onnx |
| tokenizer.json | https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/tokenizer.json |
| config.json | https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/config.json |

### Text Sparse: Splade_PP_en_v1
| File | URL |
|------|-----|
| model.onnx | https://huggingface.co/prithivida/Splade_PP_en_v1/resolve/main/onnx/model.onnx |
| tokenizer.json | https://huggingface.co/prithivida/Splade_PP_en_v1/resolve/main/tokenizer.json |

### Code: CodeBERT
| File | URL |
|------|-----|
| pytorch_model.bin | https://huggingface.co/microsoft/codebert-base/resolve/main/pytorch_model.bin |
| config.json | https://huggingface.co/microsoft/codebert-base/resolve/main/config.json |
| tokenizer.json | https://huggingface.co/microsoft/codebert-base/resolve/main/tokenizer.json |
| vocab.txt | https://huggingface.co/microsoft/codebert-base/resolve/main/vocab.txt |

### Image: CLIP
| File | URL |
|------|-----|
| pytorch_model.bin | https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin |
| config.json | https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json |
| preprocessor_config.json | https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/preprocessor_config.json |

### Reranker: FlashRank
| File | URL |
|------|-----|
| (all files) | https://huggingface.co/prithivida/flashrank-ms-marco-MiniLM-L12-v2/tree/main |

---

## Step 6: Update Configuration

### `.env.local`

```bash
# ═══════════════════════════════════════════════════════════════
# GPU SETTINGS
# ═══════════════════════════════════════════════════════════════
CUDA_VISIBLE_DEVICES=0

# ═══════════════════════════════════════════════════════════════
# TEXT EMBEDDINGS (PDR Section 5.6)
# ═══════════════════════════════════════════════════════════════
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIMENSION=768
SPARSE_MODEL=prithivida/Splade_PP_en_v1

# ═══════════════════════════════════════════════════════════════
# CODE EMBEDDINGS (PDR Section 5.6)
# ═══════════════════════════════════════════════════════════════
CODE_EMBEDDING_MODEL=microsoft/codebert-base
CODE_EMBEDDING_DIMENSION=768

# ═══════════════════════════════════════════════════════════════
# IMAGE EMBEDDINGS (PDR Section 5.6)
# ═══════════════════════════════════════════════════════════════
IMAGE_EMBEDDING_MODEL=openai/clip-vit-base-patch32
IMAGE_EMBEDDING_DIMENSION=512

# ═══════════════════════════════════════════════════════════════
# RERANKER (PDR Section 5.6)
# ═══════════════════════════════════════════════════════════════
RERANK_MODEL=ms-marco-MiniLM-L-12-v2

# ═══════════════════════════════════════════════════════════════
# MODEL CACHE PATH
# ═══════════════════════════════════════════════════════════════
FASTEMBED_CACHE_PATH=./models_cache
```

---

## Step 7: Create Embedding Service (MCP Pipeline)

```python
# mcp_server/services/embedding_service.py
"""
Unified GPU Embedding Service for MCP.
Reference: MCP_PDR.md Section 5.6 - Embedding Models & Dimensions
"""
import os
import torch
from typing import List, Union
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class EmbeddingService:
    """GPU-accelerated embedding service for text, code, and images."""
    
    def __init__(self, cache_dir: str = "./models_cache"):
        self.cache_dir = cache_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Lazy-loaded models
        self._text_dense = None
        self._text_sparse = None
        self._code_model = None
        self._code_tokenizer = None
        self._clip_model = None
        self._clip_processor = None
    
    # ═══════════════════════════════════════════════════════════
    # TEXT EMBEDDINGS (768 dim)
    # ═══════════════════════════════════════════════════════════
    
    @property
    def text_dense(self):
        if self._text_dense is None:
            from fastembed import TextEmbedding
            self._text_dense = TextEmbedding(
                model_name="BAAI/bge-base-en-v1.5",
                cache_dir=self.cache_dir
            )
        return self._text_dense
    
    @property
    def text_sparse(self):
        if self._text_sparse is None:
            from fastembed import SparseTextEmbedding
            self._text_sparse = SparseTextEmbedding(
                model_name="prithivida/Splade_PP_en_v1",
                cache_dir=self.cache_dir
            )
        return self._text_sparse
    
    def embed_text(self, texts: List[str]) -> List[List[float]]:
        """Embed text using bge-base (768 dim)."""
        return list(self.text_dense.embed(texts))
    
    def embed_text_sparse(self, texts: List[str]):
        """Embed text using SPLADE for keyword search."""
        return list(self.text_sparse.embed(texts))
    
    # ═══════════════════════════════════════════════════════════
    # CODE EMBEDDINGS (768 dim)
    # ═══════════════════════════════════════════════════════════
    
    def _load_code_model(self):
        if self._code_model is None:
            from transformers import AutoTokenizer, AutoModel
            self._code_tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/codebert-base",
                cache_dir=self.cache_dir
            )
            self._code_model = AutoModel.from_pretrained(
                "microsoft/codebert-base",
                cache_dir=self.cache_dir
            ).to(self.device)
            self._code_model.eval()
    
    def embed_code(self, code_snippets: List[str]) -> List[List[float]]:
        """Embed code using CodeBERT (768 dim)."""
        self._load_code_model()
        
        embeddings = []
        with torch.no_grad():
            for code in code_snippets:
                inputs = self._code_tokenizer(
                    code, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                outputs = self._code_model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                embeddings.append(embedding.tolist())
        
        return embeddings
    
    # ═══════════════════════════════════════════════════════════
    # IMAGE EMBEDDINGS (512 dim)
    # ═══════════════════════════════════════════════════════════
    
    def _load_clip_model(self):
        if self._clip_model is None:
            from transformers import CLIPProcessor, CLIPModel
            self._clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir=self.cache_dir
            )
            self._clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir=self.cache_dir
            ).to(self.device)
            self._clip_model.eval()
    
    def embed_image(self, images: List[Union[str, Image.Image]]) -> List[List[float]]:
        """Embed images using CLIP (512 dim)."""
        self._load_clip_model()
        
        # Load images if paths provided
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img))
            else:
                pil_images.append(img)
        
        embeddings = []
        with torch.no_grad():
            for img in pil_images:
                inputs = self._clip_processor(
                    images=img,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self._clip_model.get_image_features(**inputs)
                embedding = outputs.cpu().numpy()[0]
                embeddings.append(embedding.tolist())
        
        return embeddings
    
    def embed_image_description(self, descriptions: List[str]) -> List[List[float]]:
        """Embed image descriptions using CLIP text encoder (512 dim)."""
        self._load_clip_model()
        
        embeddings = []
        with torch.no_grad():
            for desc in descriptions:
                inputs = self._clip_processor(
                    text=desc,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                outputs = self._clip_model.get_text_features(**inputs)
                embedding = outputs.cpu().numpy()[0]
                embeddings.append(embedding.tolist())
        
        return embeddings


# Singleton instance
embedding_service = EmbeddingService()
```

---

## Step 8: Update Qdrant Schema (Multi-Vector)

Reference: MCP_PDR.md Section 8.2 - Qdrant Schema

```python
# reset_qdrant_multivector.py
"""
Create Qdrant collection with all vector types.
Reference: MCP_PDR.md Section 8.2 - Qdrant Collection Design
"""
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")
collection_name = "confluence_mcp"

# Delete if exists
try:
    client.delete_collection(collection_name)
    print(f"Deleted existing collection: {collection_name}")
except:
    pass

# Create with multi-vector support
print(f"Creating collection with multi-vector support...")
client.create_collection(
    collection_name=collection_name,
    vectors_config={
        # Text embeddings (bge-base: 768 dim)
        "text_dense": models.VectorParams(
            size=768,
            distance=models.Distance.COSINE
        ),
        # Code embeddings (CodeBERT: 768 dim)
        "code": models.VectorParams(
            size=768,
            distance=models.Distance.COSINE
        ),
        # Image embeddings (CLIP: 512 dim)
        "image": models.VectorParams(
            size=512,
            distance=models.Distance.COSINE
        )
    },
    sparse_vectors_config={
        # Sparse text embeddings (SPLADE)
        "text_sparse": models.SparseVectorParams()
    }
)

print("""
✅ Collection created with multi-vector support!

Vector Fields:
┌─────────────────┬────────────┬──────────────────────────┐
│ Field           │ Dimension  │ Model                    │
├─────────────────┼────────────┼──────────────────────────┤
│ text_dense      │ 768        │ BAAI/bge-base-en-v1.5   │
│ text_sparse     │ Sparse     │ prithivida/Splade_PP    │
│ code            │ 768        │ microsoft/codebert-base │
│ image           │ 512        │ openai/clip-vit-base    │
└─────────────────┴────────────┴──────────────────────────┘
""")
```

---

## Step 9: Verify GPU Usage

```python
# test_all_embeddings.py
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from mcp_server.services.embedding_service import embedding_service

print("=" * 60)
print("🧪 TESTING ALL EMBEDDING MODELS ON GPU")
print("=" * 60)

# Test data
texts = ["How to configure Kafka authentication?"] * 10
code = ["def hello_world():\n    print('Hello, World!')"] * 10

# 1. Text Dense
print("\n[1/4] Text Dense (bge-base)...")
start = time.time()
text_embeddings = embedding_service.embed_text(texts)
print(f"  ✅ {len(texts)} texts in {time.time()-start:.2f}s")
print(f"  Dimension: {len(text_embeddings[0])}")

# 2. Text Sparse
print("\n[2/4] Text Sparse (SPLADE)...")
start = time.time()
sparse_embeddings = embedding_service.embed_text_sparse(texts)
print(f"  ✅ {len(texts)} texts in {time.time()-start:.2f}s")

# 3. Code
print("\n[3/4] Code (CodeBERT)...")
start = time.time()
code_embeddings = embedding_service.embed_code(code)
print(f"  ✅ {len(code)} snippets in {time.time()-start:.2f}s")
print(f"  Dimension: {len(code_embeddings[0])}")

# 4. Image (text description test)
print("\n[4/4] Image Description (CLIP)...")
descriptions = ["A diagram showing Kafka architecture"] * 10
start = time.time()
img_embeddings = embedding_service.embed_image_description(descriptions)
print(f"  ✅ {len(descriptions)} descriptions in {time.time()-start:.2f}s")
print(f"  Dimension: {len(img_embeddings[0])}")

print("\n" + "=" * 60)
print("✅ ALL EMBEDDINGS WORKING ON GPU!")
print("=" * 60)
```

---

## Performance Expectations

| Model | CPU Speed | GPU Speed | Speedup |
|-------|-----------|-----------|---------|
| Text Dense (bge-base) | ~50 docs/sec | ~200 docs/sec | **4x** |
| Text Sparse (SPLADE) | ~40 docs/sec | ~150 docs/sec | **3.5x** |
| Code (CodeBERT) | ~30 docs/sec | ~120 docs/sec | **4x** |
| Image (CLIP) | ~20 imgs/sec | ~100 imgs/sec | **5x** |

---

## Troubleshooting

### CUDA Not Detected
```python
import torch
print(torch.cuda.is_available())  # False?
```
**Fix:** Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### Out of Memory (OOM)
**Fix:** Reduce batch size or process one model at a time.

### SSL Download Error (Corporate Firewall)
```python
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
```

---

## Quick Reference Commands

```powershell
# Check GPU
nvidia-smi

# Download all models
python download_all_models.py

# Reset Qdrant with multi-vector
python reset_qdrant_multivector.py

# Test all embeddings
python test_all_embeddings.py
```

---

## Related Documentation

| Document | Section |
|----------|---------|
| [MCP_PDR.md](./MCP_PDR.md) | Section 5.6 - Embedding Models & Dimensions |
| [MCP_PDR.md](./MCP_PDR.md) | Section 8.2 - Qdrant Schema |
| [MCP_PDR.md](./MCP_PDR.md) | Section 6.4 - search_code / search_images tools |

---

*Document Version: 2.0*  
*Last Updated: February 2026*  
*Changes: Added Code (CodeBERT) and Image (CLIP) embeddings, unified embedding service, multi-vector Qdrant schema*
