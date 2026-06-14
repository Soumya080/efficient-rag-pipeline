"""
Experiment Runner — Compare chunking strategies side-by-side.

Runs the same queries across all chunking strategies and saves
a comparison table with timing and retrieval stats.

Usage:
    python run_experiment.py
    python run_experiment.py --save

This is where you'll add future experiments:
    - Compare retrieval strategies (dense vs bm25 vs hybrid)
    - Compare filtering strategies
    - Ablation studies
"""

import sys
import json
import os
from datetime import datetime

from src.pipeline import RAGPipeline


# ============================================================
# EXPERIMENT CONFIG
# ============================================================

DOCUMENT = """
Neural networks process information.
Deep learning uses neural networks extensively.
AI systems automate decision making.
Technology changes modern industries.
Software tools improve business productivity.
Companies use analytics platforms daily.
Football teams analyze player statistics.
Embeddings represent semantic meaning.
Vector databases store embeddings efficiently.
Cosine similarity compares semantic vectors.
RAG systems retrieve context using embeddings.
Football tournaments attract millions of fans.
Quantum mechanics studies particles.
Cats are common pets.
Pets require regular care.
Care improves human emotional health.
Health insurance is expensive.
Insurance companies manage financial risk.
Banks provide financial services.
"""

QUERIES = [
    "What is deep learning and AI?",
    "How do embeddings work in RAG systems?",
    "Tell me about football.",
    "What is quantum mechanics?",
    "How do companies use technology?",
]

STRATEGIES = ["naive", "sentence", "semantic"]

CHUNK_CONFIGS = {
    "naive": {"chunk_size": 150, "overlap": 30},
    "sentence": {"chunk_size": 200, "overlap": 1},
    "semantic": {"threshold": 0.3, "max_chunk_size": 200, "alpha": 0.5},
}


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment(save_results=False):
    """
    Run all strategies on all queries and compare.

    Returns:
        List of result dicts (one per strategy-query pair)
    """
    all_results = []

    for strategy in STRATEGIES:
        print(f"\n{'#' * 70}")
        print(f"  STRATEGY: {strategy}")
        print(f"{'#' * 70}")

        pipeline = RAGPipeline(chunking_strategy=strategy)
        chunk_kwargs = CHUNK_CONFIGS.get(strategy, {})

        for query in QUERIES:
            print(f"\n--- Query: '{query}' ---")

            result = pipeline.run(
                document=DOCUMENT,
                query=query,
                top_k=3,
                use_expansion=True,
                use_llm=False,
                chunk_kwargs=chunk_kwargs,
                verbose=False,
            )

            # collect summary
            summary = {
                "strategy": strategy,
                "query": query,
                "num_chunks": result["stats"]["num_chunks"],
                "num_retrieved": result["stats"]["num_retrieved"],
                "num_expanded": result["stats"]["num_expanded"],
                "context_length": result["stats"]["context_length"],
                "context_words": result["stats"]["context_tokens_approx"],
                "chunking_time": round(result["stats"]["chunking_time"], 4),
                "embedding_time": round(result["stats"]["embedding_time"], 4),
                "retrieval_time": round(result["stats"]["retrieval_time"], 4),
                "total_time": round(result["stats"]["total_time"], 4),
                "top_score": round(result["retrieved"][0]["score"], 4)
                    if result["retrieved"] else 0,
            }

            all_results.append(summary)

            print(f"  Chunks: {summary['num_chunks']} | "
                  f"Retrieved: {summary['num_retrieved']} | "
                  f"Context: {summary['context_words']} words | "
                  f"Top score: {summary['top_score']}")

    # ---- COMPARISON TABLE ----
    print_comparison_table(all_results)

    # ---- SAVE ----
    if save_results:
        save_experiment(all_results)

    return all_results


def print_comparison_table(results):
    """Print a clean comparison table."""

    print(f"\n\n{'=' * 90}")
    print("EXPERIMENT RESULTS — CHUNKING STRATEGY COMPARISON")
    print(f"{'=' * 90}")

    # header
    print(f"\n{'Strategy':<12} {'Query':<40} {'Chunks':<8} "
          f"{'Top Score':<10} {'Context':<10} {'Time(s)':<8}")
    print("-" * 90)

    for r in results:
        query_short = r["query"][:37] + "..." if len(r["query"]) > 37 else r["query"]
        print(f"{r['strategy']:<12} {query_short:<40} "
              f"{r['num_chunks']:<8} {r['top_score']:<10.4f} "
              f"{r['context_words']:<10} {r['total_time']:<8.4f}")

    # ---- STRATEGY AVERAGES ----
    print(f"\n\n{'=' * 60}")
    print("STRATEGY AVERAGES")
    print(f"{'=' * 60}")
    print(f"\n{'Strategy':<12} {'Avg Chunks':<12} {'Avg Score':<12} "
          f"{'Avg Context':<14} {'Avg Time':<10}")
    print("-" * 60)

    for strategy in STRATEGIES:
        s_results = [r for r in results if r["strategy"] == strategy]
        avg_chunks = sum(r["num_chunks"] for r in s_results) / len(s_results)
        avg_score = sum(r["top_score"] for r in s_results) / len(s_results)
        avg_context = sum(r["context_words"] for r in s_results) / len(s_results)
        avg_time = sum(r["total_time"] for r in s_results) / len(s_results)

        print(f"{strategy:<12} {avg_chunks:<12.1f} {avg_score:<12.4f} "
              f"{avg_context:<14.1f} {avg_time:<10.4f}")


def save_experiment(results):
    """Save experiment results to JSON."""

    os.makedirs("experiments/results", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiments/results/chunking_comparison_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump({
            "experiment": "chunking_strategy_comparison",
            "timestamp": timestamp,
            "strategies": STRATEGIES,
            "num_queries": len(QUERIES),
            "results": results,
        }, f, indent=2)

    print(f"\n[Saved] Results written to: {filename}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    save = "--save" in sys.argv
    run_experiment(save_results=save)
