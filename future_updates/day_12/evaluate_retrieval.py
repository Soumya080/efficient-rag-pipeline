"""
Retrieval Evaluation Framework

Benchmarks and compares retrieval quality across:
- Brute-Force Cosine Dense (dense)
- FAISS Dense (faiss)
- BM25 Sparse (sparse)
- Hybrid (dense + sparse fusion)

Calculates evaluation metrics:
- Hit Rate @ K: Was the ground-truth context retrieved?
- Mean Reciprocal Rank (MRR @ K): How high did the correct context rank?
"""

from src.pipeline import RAGPipeline

# ============================================================
# EVALUATION DATASET
# List of documents, queries, and their expected relevant contents.
# ============================================================
EVAL_DOCUMENT = """
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

# Format: (query, keyword_contained_in_ground_truth)
TEST_CASES = [
    ("What is deep learning and AI?", "AI systems automate decision making."),
    ("Tell me about football and sports.", "Football teams analyze player statistics."),
    ("How do vector databases store embeddings?", "Vector databases store embeddings efficiently."),
    ("Tell me about cats and pets.", "Cats are common pets."),
    ("Explain quantum mechanics.", "Quantum mechanics studies particles."),
]

RETRIEVAL_MODES = ["dense", "faiss", "sparse", "hybrid"]

def evaluate_mode(mode, top_k=3):
    """
    Run evaluation on the test suite using a specific retrieval mode.
    """
    pipeline = RAGPipeline(chunking_strategy="semantic", retrieval_mode=mode)
    
    hits = 0
    mrr_sum = 0.0
    total = len(TEST_CASES)
    
    for query, ground_truth in TEST_CASES:
        # Run pipeline in current retrieval mode (Skip LLM and expansion to check raw retrieval indices)
        result = pipeline.run(
            document=EVAL_DOCUMENT,
            query=query,
            top_k=top_k,
            use_expansion=False,
            use_llm=False,
            verbose=False
        )
        
        retrieved_chunks = result["retrieved"]
        
        # Check if ground_truth sentence exists in retrieved contents
        found = False
        rank = 0
        for i, hit in enumerate(retrieved_chunks, start=1):
            if ground_truth in hit["content"]:
                found = True
                rank = i
                break
                
        if found:
            hits += 1
            mrr_sum += 1.0 / rank
            
    hit_rate = hits / total
    mrr = mrr_sum / total
    
    return hit_rate, mrr

def main():
    print("=" * 60)
    print("RUNNING RETRIEVAL EVALUATION BENCHMARK")
    print("=" * 60)
    
    results = {}
    for mode in RETRIEVAL_MODES:
        print(f"\nEvaluating mode: '{mode}'...")
        hit_rate, mrr = evaluate_mode(mode, top_k=3)
        results[mode] = {"Hit Rate@3": hit_rate, "MRR@3": mrr}
        
    print("\n" + "=" * 60)
    print("FINAL EVALUATION METRICS")
    print("=" * 60)
    print(f"{'Retrieval Mode':<18} | {'Hit Rate @ 3':<15} | {'MRR @ 3':<10}")
    print("-" * 60)
    for mode, metrics in results.items():
        print(f"{mode:<18} | {metrics['Hit Rate@3']:<15.2%} | {metrics['MRR@3']:<10.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
