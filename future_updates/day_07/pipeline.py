"""
RAG Pipeline — Configurable & Experiment-Ready

This is the core pipeline class. You configure WHICH chunking strategy
and WHICH retrieval strategy to use, then run it.

For experiments, you swap strategies and compare results.

Usage:
    from src.pipeline import RAGPipeline

    pipeline = RAGPipeline(
        chunking_strategy="semantic",   # or "naive", "sentence"
        retrieval_strategy="dense",     # only "dense" for now
    )

    result = pipeline.run(document, query)
"""

import time

from src.embeddings.encoder import EmbeddingEncoder
from src.chunking import naive, sentence, semantic
from src.retrieval.dense import retrieve, expand_neighbors, build_context
from src.generation.generator import build_rag_prompt, generate_answer


# ============================================================
# MAP: strategy name → chunking module
# ============================================================
# When you add a new chunking strategy, just add it here.

CHUNKING_STRATEGIES = {
    "naive": naive,
    "sentence": sentence,
    "semantic": semantic,
}

# When you add new retrieval strategies (bm25, hybrid, faiss),
# add them here the same way.


class RAGPipeline:
    """
    Configurable RAG pipeline.

    Lets you swap chunking and retrieval strategies for experiments.
    Tracks timing and stats for each run.
    """

    def __init__(self, chunking_strategy="semantic", model_name="all-MiniLM-L6-v2"):
        """
        Initialize the pipeline.

        Args:
            chunking_strategy: "naive", "sentence", or "semantic"
            model_name: Embedding model name
        """
        # validate strategy
        if chunking_strategy not in CHUNKING_STRATEGIES:
            raise ValueError(
                f"Unknown chunking strategy: '{chunking_strategy}'. "
                f"Choose from: {list(CHUNKING_STRATEGIES.keys())}"
            )

        self.chunking_strategy = chunking_strategy
        self.chunker = CHUNKING_STRATEGIES[chunking_strategy]
        self.model_name = model_name

        # load encoder
        print(f"\n[Pipeline] Loading encoder: {model_name}")
        self.encoder = EmbeddingEncoder(model_name)

        print(f"[Pipeline] Chunking strategy: {chunking_strategy}")
        print(f"[Pipeline] Ready.\n")

    def chunk_document(self, document, **chunk_kwargs):
        """
        Chunk a document using the selected strategy.

        Args:
            document: Input text
            **chunk_kwargs: Strategy-specific params
                naive:    chunk_size, overlap
                sentence: chunk_size, overlap
                semantic: threshold, max_chunk_size, alpha

        Returns:
            List of chunk dicts
        """
        # semantic chunker needs encoder; naive/sentence don't
        if self.chunking_strategy == "semantic":
            chunks = self.chunker.chunk(
                document,
                encoder=self.encoder,
                **chunk_kwargs
            )
        else:
            chunks = self.chunker.chunk(document, **chunk_kwargs)

        return chunks

    def run(self, document, query, top_k=3, min_similarity=0.2,
            use_expansion=True, use_llm=False, llm_model="phi3",
            chunk_kwargs=None, verbose=True):
        """
        Run the full RAG pipeline.

        Steps:
            1. Chunk document  (selected strategy)
            2. Embed chunks    (encoder)
            3. Retrieve        (dense cosine similarity)
            4. Expand          (neighbor chunks, optional)
            5. Build context   (concatenate)
            6. Generate        (Ollama, optional)

        Args:
            document: Input text document
            query: User's question
            top_k: Number of chunks to retrieve
            min_similarity: Minimum similarity threshold
            use_expansion: Whether to expand with neighbor chunks
            use_llm: Whether to call Ollama for generation
            llm_model: Ollama model name
            chunk_kwargs: Dict of params for the chunking strategy
            verbose: Print progress

        Returns:
            Dict with all pipeline outputs + timing stats
        """
        if chunk_kwargs is None:
            chunk_kwargs = {}

        stats = {}
        t_start = time.time()

        # -----------------------------------------
        # STEP 1: Chunk document
        # -----------------------------------------
        t = time.time()
        chunks = self.chunk_document(document, **chunk_kwargs)
        stats["chunking_time"] = time.time() - t
        stats["num_chunks"] = len(chunks)

        if verbose:
            print(f"[Step 1] Chunking ({self.chunking_strategy}): "
                  f"{len(chunks)} chunks in {stats['chunking_time']:.3f}s")

        # -----------------------------------------
        # STEP 2: Embed chunks
        # -----------------------------------------
        t = time.time()
        chunks = self.encoder.generate_chunk_embeddings(chunks)
        stats["embedding_time"] = time.time() - t

        if verbose:
            print(f"[Step 2] Embedding: {len(chunks)} chunks in "
                  f"{stats['embedding_time']:.3f}s")

        # -----------------------------------------
        # STEP 3: Retrieve
        # -----------------------------------------
        t = time.time()
        retrieved = retrieve(
            query=query,
            chunk_list=chunks,
            encoder=self.encoder,
            top_k=top_k,
            min_similarity=min_similarity
        )
        stats["retrieval_time"] = time.time() - t
        stats["num_retrieved"] = len(retrieved)

        if verbose:
            print(f"[Step 3] Retrieval: {len(retrieved)} chunks in "
                  f"{stats['retrieval_time']:.3f}s")

        # -----------------------------------------
        # STEP 4: Expand neighbors (optional)
        # -----------------------------------------
        if use_expansion and len(retrieved) > 0:
            expanded = expand_neighbors(retrieved, chunks)
            stats["num_expanded"] = len(expanded)
        else:
            expanded = retrieved
            stats["num_expanded"] = len(retrieved)

        if verbose:
            print(f"[Step 4] Expansion: {stats['num_expanded']} chunks "
                  f"(expansion={'ON' if use_expansion else 'OFF'})")

        # -----------------------------------------
        # STEP 5: Build context
        # -----------------------------------------
        context = build_context(expanded)
        stats["context_length"] = len(context)
        stats["context_tokens_approx"] = len(context.split())

        if verbose:
            print(f"[Step 5] Context: {len(context)} chars, "
                  f"~{stats['context_tokens_approx']} words")

        # -----------------------------------------
        # STEP 6: Generate (optional)
        # -----------------------------------------
        prompt = build_rag_prompt(query, context)
        answer = None

        if use_llm:
            t = time.time()
            answer = generate_answer(prompt, model_name=llm_model)
            stats["generation_time"] = time.time() - t

            if verbose:
                print(f"[Step 6] Generation: done in "
                      f"{stats['generation_time']:.3f}s")
        else:
            if verbose:
                print(f"[Step 6] Generation: skipped (use_llm=False)")

        stats["total_time"] = time.time() - t_start

        # -----------------------------------------
        # Build result
        # -----------------------------------------
        result = {
            "query": query,
            "chunking_strategy": self.chunking_strategy,
            "chunk_kwargs": chunk_kwargs,
            "chunks": chunks,
            "retrieved": retrieved,
            "expanded": expanded,
            "context": context,
            "prompt": prompt,
            "answer": answer,
            "stats": stats,
        }

        if verbose:
            print(f"\n[Done] Total: {stats['total_time']:.3f}s")
            self._print_retrieved(query, retrieved)

        return result

    def _print_retrieved(self, query, results):
        """Pretty-print retrieval results."""
        print("\n" + "=" * 70)
        print(f"QUERY: '{query}'")
        print(f"Retrieved: {len(results)} chunks")
        print("=" * 70)

        for i, res in enumerate(results, start=1):
            score = res.get("score", 0.0)
            content = res.get("content", "").strip()
            print(f"\n[{i}] SIMILARITY: {score:.4f}")
            print(f"    {content[:120]}{'...' if len(content) > 120 else ''}")

        print("=" * 70)
