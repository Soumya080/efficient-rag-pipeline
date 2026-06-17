# RAG Pipeline Future Updates & Commit Schedule

This folder contains the code files and git commit commands to run over the next few days to show real, high-impact research progress.

---

## 📅 Overview of the 12-Day Plan

| Day / Date | Task | Commit Message | Files to Add / Modify |
|---|---|---|---|
| **Day 1 (May 20)** | **Initial Project Setup & Naive Chunker** | `chore: initial repository setup with fixed-size naive chunker` | `requirements.txt`, `README.md`, `src/__init__.py`, `src/chunking/__init__.py`, `src/chunking/utils.py`, `src/chunking/naive.py` |
| **Day 2 (May 21)** | **Sentence-Based Chunking** | `feat(chunking): implement sentence-based splitting and chunking` | Copy `future_updates/day_02/sentence.py` -> `src/chunking/sentence.py` |
| **Day 3 (May 22)** | **Semantic Chunking** | `feat(chunking): implement semantic similarity-based chunker` | Copy `future_updates/day_03/semantic.py` -> `src/chunking/semantic.py` |
| **Day 4 (May 23)** | **Embedding Encoder** | `feat(embeddings): implement SentenceTransformer encoder wrapper` | Create `src/embeddings/`, Copy `future_updates/day_04/encoder.py` -> `src/embeddings/encoder.py` (Add empty `src/embeddings/__init__.py`) |
| **Day 5 (May 24)** | **Dense Cosine Retriever** | `feat(retrieval): implement dense cosine similarity retriever` | Create `src/retrieval/`, Copy `future_updates/day_05/dense.py` -> `src/retrieval/dense.py` (Add empty `src/retrieval/__init__.py`) |
| **Day 6 (May 25)** | **Generation & RAG Prompts** | `feat(generation): build RAG prompt template and Ollama interface` | Create `src/generation/`, Copy `future_updates/day_06/generator.py` -> `src/generation/generator.py` (Add empty `src/generation/__init__.py`) |
| **Day 7 (May 26)** | **Core Pipeline & Runner** | `feat(pipeline): create core RAGPipeline orchestrator and runner` | Copy `future_updates/day_07/pipeline.py` -> `src/pipeline.py`, Copy `future_updates/day_07/run_pipeline.py` -> `run_pipeline.py`, Copy `future_updates/day_07/run_experiment.py` -> `run_experiment.py` |
| **Day 8 (May 27)** | **FAISS Acceleration** | `feat(retrieval): integrate FAISS index for accelerated vector search` | Copy `future_updates/day_08/dense_faiss.py` -> `src/retrieval/dense_faiss.py` |
| **Day 9 (May 28)** | **BM25 Lexical Sparse Search** | `feat(retrieval): implement BM25 sparse search from scratch` | Copy `future_updates/day_09/sparse.py` -> `src/retrieval/sparse.py` |
| **Day 10 (May 29)** | **Hybrid Search (RRF)** | `feat(retrieval): implement hybrid search with Reciprocal Rank Fusion` | Copy `future_updates/day_10/hybrid.py` -> `src/retrieval/hybrid.py` |
| **Day 11 (May 30)** | **Complete Pipeline Integration** | `refactor(pipeline): support dense, faiss, sparse, and hybrid modes` | Copy `future_updates/day_11/pipeline.py` -> `src/pipeline.py`, Copy `future_updates/day_11/requirements.txt` -> `requirements.txt` |
| **Day 12 (May 31)** | **Retrieval Evaluation Suite** | `feat(eval): add evaluation framework for Hit Rate and MRR` | Copy `future_updates/day_12/evaluate_retrieval.py` -> `evaluate_retrieval.py` |

---

## 🛠️ Step-by-Step Instructions

### Day 1 (May 20 - Today)
Initialize your local Git repository and make your first push:
```bash
git init
git add README.md requirements.txt src/
git commit -m "chore: initial repository setup with fixed-size naive chunker"
```
*(Now link to GitHub and push - see the main guide response for details).*

### Day 2 (May 21)
1. Copy `future_updates/day_02/sentence.py` to `src/chunking/sentence.py`
2. Add to Git and commit:
   ```bash
   git add src/chunking/sentence.py
   git commit -m "feat(chunking): implement sentence-based splitting and chunking"
   git push origin main
   ```

### Day 3 (May 22)
1. Copy `future_updates/day_03/semantic.py` to `src/chunking/semantic.py`
2. Add to Git and commit:
   ```bash
   git add src/chunking/semantic.py
   git commit -m "feat(chunking): implement semantic similarity-based chunker"
   git push origin main
   ```

### Day 4 (May 23)
1. Create a new folder: `src/embeddings`
2. Create an empty file: `src/embeddings/__init__.py`
3. Copy `future_updates/day_04/encoder.py` to `src/embeddings/encoder.py`
4. Update `requirements.txt` to include `sentence-transformers`:
   ```
   numpy
   sentence-transformers
   ```
5. Add to Git and commit:
   ```bash
   git add src/embeddings/ requirements.txt
   git commit -m "feat(embeddings): implement SentenceTransformer encoder wrapper"
   git push origin main
   ```

### Day 5 (May 24)
1. Create a new folder: `src/retrieval`
2. Create an empty file: `src/retrieval/__init__.py`
3. Copy `future_updates/day_05/dense.py` to `src/retrieval/dense.py`
4. Add to Git and commit:
   ```bash
   git add src/retrieval/
   git commit -m "feat(retrieval): implement dense cosine similarity retriever"
   git push origin main
   ```

### Day 6 (May 25)
1. Create a new folder: `src/generation`
2. Create an empty file: `src/generation/__init__.py`
3. Copy `future_updates/day_06/generator.py` to `src/generation/generator.py`
4. Update `requirements.txt` to include `ollama`:
   ```
   numpy
   sentence-transformers
   ollama
   ```
5. Add to Git and commit:
   ```bash
   git add src/generation/ requirements.txt
   git commit -m "feat(generation): build RAG prompt template and Ollama interface"
   git push origin main
   ```

### Day 7 (May 26)
1. Copy `future_updates/day_07/pipeline.py` to `src/pipeline.py`
2. Copy `future_updates/day_07/run_pipeline.py` to `run_pipeline.py`
3. Copy `future_updates/day_07/run_experiment.py` to `run_experiment.py`
4. Add to Git and commit:
   ```bash
   git add src/pipeline.py run_pipeline.py run_experiment.py
   git commit -m "feat(pipeline): create core RAGPipeline orchestrator and runner"
   git push origin main
   ```

### Day 8 (May 27)
1. Copy `future_updates/day_08/dense_faiss.py` to `src/retrieval/dense_faiss.py`
2. Add to Git and commit:
   ```bash
   git add src/retrieval/dense_faiss.py
   git commit -m "feat(retrieval): integrate FAISS index for accelerated vector search"
   git push origin main
   ```

### Day 9 (May 28)
1. Copy `future_updates/day_09/sparse.py` to `src/retrieval/sparse.py`
2. Add to Git and commit:
   ```bash
   git add src/retrieval/sparse.py
   git commit -m "feat(retrieval): implement BM25 sparse search from scratch"
   git push origin main
   ```

### Day 10 (May 29)
1. Copy `future_updates/day_10/hybrid.py` to `src/retrieval/hybrid.py`
2. Add to Git and commit:
   ```bash
   git add src/retrieval/hybrid.py
   git commit -m "feat(retrieval): implement hybrid search with Reciprocal Rank Fusion"
   git push origin main
   ```

### Day 11 (May 30)
1. Copy `future_updates/day_11/pipeline.py` to `src/pipeline.py`
2. Copy `future_updates/day_11/requirements.txt` to `requirements.txt`
3. Add to Git and commit:
   ```bash
   git add src/pipeline.py requirements.txt
   git commit -m "refactor(pipeline): support dense, faiss, sparse, and hybrid modes"
   git push origin main
   ```

### Day 12 (May 31)
1. Copy `future_updates/day_12/evaluate_retrieval.py` to `evaluate_retrieval.py`
2. Add to Git and commit:
   ```bash
   git add evaluate_retrieval.py
   git commit -m "feat(eval): add evaluation framework for Hit Rate and MRR"
   git push origin main
   ```
