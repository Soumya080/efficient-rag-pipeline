"""
Quick Demo — Run the RAG pipeline once.

Usage:
    python run_pipeline.py
    python run_pipeline.py --strategy semantic
    python run_pipeline.py --strategy naive --query "What is AI?"
    python run_pipeline.py --strategy sentence --no-generate
"""

import sys
from src.pipeline import RAGPipeline


# ============================================================
# SAMPLE DOCUMENT (same text from notebooks)
# ============================================================

SAMPLE_DOCUMENT = """
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


def main():
    # ---- parse args ----
    strategy = "semantic"
    query = "What is deep learning and AI?"
    use_llm = True

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--strategy" and i + 1 < len(args):
            strategy = args[i + 1]
        elif arg == "--query" and i + 1 < len(args):
            query = args[i + 1]
        elif arg == "--no-generate":
            use_llm = False

    # ---- build pipeline ----
    pipeline = RAGPipeline(chunking_strategy=strategy)

    # ---- run ----
    print(f"\nRunning with strategy='{strategy}', query='{query}'\n")

    result = pipeline.run(
        document=SAMPLE_DOCUMENT,
        query=query,
        top_k=3,
        use_expansion=True,
        use_llm=use_llm,
    )

    # ---- show answer ----
    if result["answer"]:
        print("\n" + "=" * 70)
        print("ANSWER:")
        print("=" * 70)
        print(result["answer"])

    # ---- show stats ----
    print("\n" + "-" * 70)
    print("STATS:")
    for k, v in result["stats"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
