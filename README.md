# Efficient RAG Pipeline — Built From Scratch

A modular RAG pipeline built step-by-step from scratch.

## Project Structure (Current State: Day 1)

At this stage, we have set up the core structure and our first naive chunker:

```
efficient-rag-pipeline/
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   └── chunking/
│       ├── __init__.py
│       ├── utils.py             # Shared: sentence_splitter
│       └── naive.py             # Fixed-size character chunking
|       └── sentence.py
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the naive chunker test
python -m src.chunking.naive
```
