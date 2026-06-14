# Efficient RAG Pipeline — Built From Scratch

A modular Retrieval-Augmented Generation (RAG) pipeline built step-by-step from scratch. Each component is implemented from first principles to deeply understand the internals of modern RAG systems.

## Project Structure

```
efficient-rag-pipeline/
├── .gitignore
├── README.md
├── requirements.txt
├── run_pipeline.py              # CLI runner for the full RAG pipeline
├── run_experiment.py            # Experiment runner for benchmarking & analysis
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py              # Core RAGPipeline orchestrator
│   │
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── utils.py             # Shared utilities: sentence_splitter
│   │   ├── naive.py             # Fixed-size character chunking
│   │   ├── sentence.py          # Sentence-boundary aware chunking
│   │   └── semantic.py          # Semantic similarity-based chunking
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── encoder.py           # SentenceTransformer encoder wrapper
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── dense.py             # Dense cosine similarity retriever
│   │   └── dense_faiss.py       # FAISS-accelerated vector search
│   │
│   └── generation/
│       ├── __init__.py
│       └── generator.py         # RAG prompt template & Ollama interface
│
└── future_updates/              # Staged code for upcoming features
    └── PLAN.md
```

## Components

### Chunking Strategies
- **Naive Chunker** — Fixed-size character-based splitting with configurable overlap
- **Sentence Chunker** — Respects sentence boundaries for more coherent chunks
- **Semantic Chunker** — Groups sentences by semantic similarity using embeddings

### Embeddings
- **Encoder** — Wrapper around SentenceTransformer models for document & query encoding

### Retrieval
- **Dense Retriever** — Cosine similarity search over embedded chunks
- **FAISS Retriever** — Hardware-accelerated approximate nearest neighbor search via FAISS

### Generation
- **Generator** — Prompt templating and LLM interface via Ollama for answer generation

### Pipeline
- **RAGPipeline** — End-to-end orchestrator that chains chunking → embedding → retrieval → generation
- **Runner Scripts** — `run_pipeline.py` for interactive use, `run_experiment.py` for batch experiments

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the RAG pipeline interactively
python run_pipeline.py

# Run experiments
python run_experiment.py
```

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
| 9 | BM25 Sparse Search | 🔜 |
| 10 | Hybrid Search (RRF) | 🔜 |
| 11 | Full Pipeline Integration | 🔜 |
| 12 | Retrieval Evaluation Suite | 🔜 |
