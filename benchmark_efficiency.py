import os
import argparse
from tqdm import tqdm
from src.pipeline import RAGPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/knowledge_base/")
    args = parser.parse_args()

    questions = [
        "What is the self-attention mechanism used in?",
        "What optimization algorithms are used in deep learning?",
        "What does FAISS stand for?",
        "How does BM25 improve upon TF-IDF?",
        "What is Q-learning?"
    ]

    modes = ["dense", "faiss", "sparse", "hybrid"]

    print(f"Loading pipeline and indexing {args.data}...")
    pipeline = RAGPipeline(chunking_strategy="semantic", retrieval_mode="hybrid")
    if os.path.exists(args.data):
        pipeline.ingest(args.data, verbose=False)
    else:
        print("Data directory not found. Please ensure data/knowledge_base/ exists.")
        return

    print("\n" + "="*60)
    print("           COST-EFFICIENCY BENCHMARK           ")
    print("="*60)
    print(f"{'Mode':<10} | {'Tokens/Query':<15} | {'Retrieval Time (ms)':<20}")
    print("-" * 60)

    for mode in modes:
        pipeline.retrieval_mode = mode
        avg_tokens = 0
        avg_time = 0
        
        for q in questions:
            res = pipeline.query(q, top_k=3, use_llm=False, verbose=False)
            # Estimate tokens as approx chars / 4
            context_len = len(" ".join(res['context']))
            avg_tokens += context_len / 4
            avg_time += res['stats'].get('retrieval_time', 0)
            
        avg_tokens /= len(questions)
        avg_time = (avg_time / len(questions)) * 1000
        
        print(f"{mode:<10} | {avg_tokens:<15.1f} | {avg_time:<20.2f}")
    print("="*60)

if __name__ == "__main__":
    main()
