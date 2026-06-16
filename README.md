# Efficient RAG Pipeline — Built From Scratch

A modular Retrieval-Augmented Generation (RAG) pipeline built step-by-step from first principles. Every component — chunking, embedding, retrieval (dense, sparse, hybrid), generation, and evaluation — is implemented from scratch to deeply understand the internals of modern RAG systems.

> **No LangChain. No LlamaIndex. No black boxes.** Just clean Python, real algorithms, and full control over every stage of the pipeline.

---

## Project Structure

```
efficient-rag-pipeline/
├── .gitignore
├── README.md
├── requirements.txt
├── run_pipeline.py                  # Interactive CLI runner
├── run_experiment.py                # Batch experiment runner for benchmarking
├── evaluate_retrieval.py            # Retrieval evaluation suite (Hit Rate, MRR)
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py                  # Core RAGPipeline orchestrator (all 4 retrieval modes)
│   │
│   ├── chunking/                    # Document chunking strategies
│   │   ├── __init__.py
│   │   ├── utils.py                 # Shared utilities: cosine_similarity, sentence_splitter
│   │   ├── naive.py                 # Fixed-size character-based chunking with overlap
│   │   ├── sentence.py              # Sentence-boundary aware chunking
│   │   └── semantic.py              # Semantic similarity-based chunking via embeddings
│   │
│   ├── embeddings/                  # Text embedding layer
│   │   ├── __init__.py
│   │   └── encoder.py               # SentenceTransformer encoder wrapper
│   │
│   ├── retrieval/                   # Search & retrieval strategies
│   │   ├── __init__.py
│   │   ├── dense.py                 # Brute-force cosine similarity retriever
│   │   ├── dense_faiss.py           # FAISS-accelerated approximate nearest neighbor search
│   │   ├── sparse.py                # BM25 (Okapi) sparse retriever — built from scratch
│   │   └── hybrid.py                # Reciprocal Rank Fusion (RRF) combining dense + sparse
│   │
│   └── generation/                  # LLM answer generation
│       ├── __init__.py
│       └── generator.py             # RAG prompt template & Ollama LLM interface
│
└── evaluate_retrieval.py            # Hit Rate @ K & MRR @ K benchmark across all modes
```

---

## Components

### 🔪 Chunking Strategies

| Strategy | File | Description |
|----------|------|-------------|
| **Naive** | `src/chunking/naive.py` | Fixed-size character splitting with configurable chunk size and overlap |
| **Sentence** | `src/chunking/sentence.py` | Respects sentence boundaries — groups sentences into coherent chunks |
| **Semantic** | `src/chunking/semantic.py` | Groups sentences by embedding similarity — splits where meaning shifts |

### 🧮 Embeddings

| Component | File | Description |
|-----------|------|-------------|
| **Encoder** | `src/embeddings/encoder.py` | Wrapper around `sentence-transformers` (default: `all-MiniLM-L6-v2`, 384-dim) for encoding documents and queries |

### 🔍 Retrieval Strategies

| Strategy | File | Description |
|----------|------|-------------|
| **Dense (Brute-Force)** | `src/retrieval/dense.py` | Cosine similarity search over all chunk embeddings — includes neighbor expansion and context building |
| **Dense (FAISS)** | `src/retrieval/dense_faiss.py` | Hardware-accelerated ANN search via Facebook's FAISS library — `IndexFlatIP` with L2-normalized vectors |
| **Sparse (BM25)** | `src/retrieval/sparse.py` | Okapi BM25 ranking function built from scratch — pure Python, no sklearn, lexical keyword matching |
| **Hybrid (RRF)** | `src/retrieval/hybrid.py` | Reciprocal Rank Fusion combining FAISS dense + BM25 sparse results for best-of-both-worlds retrieval |

### 💬 Generation

| Component | File | Description |
|-----------|------|-------------|
| **Generator** | `src/generation/generator.py` | RAG prompt template with strict grounding rules + Ollama LLM interface (default: `phi3`) |

### 🔗 Pipeline

| Component | File | Description |
|-----------|------|-------------|
| **RAGPipeline** | `src/pipeline.py` | End-to-end orchestrator: chunking → embedding → retrieval → context building → generation. Supports all 4 retrieval modes via config |
| **Interactive Runner** | `run_pipeline.py` | CLI tool for running the pipeline on custom documents and queries |
| **Experiment Runner** | `run_experiment.py` | Batch benchmarking across chunking strategies and retrieval modes |

### 📊 Evaluation

| Component | File | Description |
|-----------|------|-------------|
| **Retrieval Evaluator** | `evaluate_retrieval.py` | Benchmarks all 4 retrieval modes with Hit Rate @ K and Mean Reciprocal Rank (MRR @ K) on a test suite |

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Soumya080/efficient-rag-pipeline.git
cd efficient-rag-pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Make sure Ollama is running (for generation)
ollama serve
ollama pull phi3

# 4. Run the interactive pipeline
python run_pipeline.py

# 5. Run experiments across all retrieval modes
python run_experiment.py

# 6. Run retrieval evaluation benchmark
python evaluate_retrieval.py
```

---

## Retrieval Modes

The pipeline supports 4 retrieval modes, selectable via the `retrieval_mode` parameter:

```python
from src.pipeline import RAGPipeline

# Dense brute-force cosine similarity
pipeline = RAGPipeline(chunking_strategy="semantic", retrieval_mode="dense")

# FAISS-accelerated dense search
pipeline = RAGPipeline(chunking_strategy="semantic", retrieval_mode="faiss")

# BM25 sparse keyword search
pipeline = RAGPipeline(chunking_strategy="semantic", retrieval_mode="sparse")

# Hybrid: FAISS + BM25 with Reciprocal Rank Fusion
pipeline = RAGPipeline(chunking_strategy="semantic", retrieval_mode="hybrid")

# Run the pipeline
result = pipeline.run(
    document="Your document text...",
    query="Your question?",
    top_k=3,
    use_llm=True
)
print(result["answer"])
```

---

## Key Algorithms Implemented

| Algorithm | Where | What It Does |
|-----------|-------|-------------|
| **Cosine Similarity** | `src/chunking/utils.py` | Measures semantic similarity between embedding vectors |
| **Semantic Chunking** | `src/chunking/semantic.py` | Splits documents where consecutive sentence similarity drops below threshold |
| **FAISS IndexFlatIP** | `src/retrieval/dense_faiss.py` | Inner product search on L2-normalized vectors for fast ANN retrieval |
| **Okapi BM25** | `src/retrieval/sparse.py` | Classic TF-IDF ranking with term frequency saturation and document length normalization |
| **Reciprocal Rank Fusion** | `src/retrieval/hybrid.py` | Combines ranked lists from multiple retrievers: `RRF(d) = Σ 1/(k + rank(d))` |
| **Neighbor Expansion** | `src/retrieval/dense.py` | Expands retrieved chunks with adjacent chunks for better context coverage |

---

## Requirements

```
sentence-transformers
numpy
ollama
faiss-cpu
```

---

## Development Progress

| Day | Feature | Status |
|-----|---------|--------|
| 1 | Initial Setup & Naive Chunker | ✅ |
| 2 | Sentence-Based Chunking | ✅ |
| 3 | Semantic Chunking | ✅ |
| 4 | Embedding Encoder | ✅ |
| 5 | Dense Cosine Retriever | ✅ |
| 6 | Generation & RAG Prompts | ✅ |
| 7 | Core Pipeline & Runner | ✅ |
| 8 | FAISS Acceleration | ✅ |
| 9 | BM25 Sparse Search | ✅ |
| 10 | Hybrid Search (RRF) | ✅ |
| 11 | Full Pipeline Integration | ✅ |
| 12 | Retrieval Evaluation Suite | ✅ |

---

## What's Next

- [ ] Cross-encoder reranking for improved precision
- [ ] Redundancy filtering (cosine deduplication)
- [ ] Query-aware context compression
- [ ] Cost-aware adaptive retrieval (accuracy-per-token optimization)
- [ ] Contextual bandit for automatic k-selection
- [ ] Evaluation on NQ, HotpotQA, and PopQA benchmarks

---

## Built By

**Soumya** — Independent AI/ML researcher building retrieval-augmented generation systems from first principles.

- GitHub: [@Soumya080](https://github.com/Soumya080)
