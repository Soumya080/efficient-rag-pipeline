import os
import time
import argparse
from tqdm import tqdm
from src.pipeline import RAGPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/knowledge_base/")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    # We use a single question repeated to test latency without caching
    question = "What is the self-attention mechanism used in deep learning and NLP?"

    modes = ["dense", "faiss", "sparse", "hybrid"]

    print(f"Loading pipeline and indexing {args.data}...")
    pipeline = RAGPipeline(chunking_strategy="semantic", retrieval_mode="hybrid")
    if os.path.exists(args.data):
        pipeline.ingest(args.data, verbose=False)
    else:
        print("Data directory not found. Please ensure data/knowledge_base/ exists.")
        return

    print("\n" + "="*50)
    print("           RETRIEVAL LATENCY BENCHMARK           ")
    print("="*50)
    print(f"{'Mode':<10} | {'Latency (ms/query)':<20} | {'Queries/sec':<15}")
    print("-" * 50)

    for mode in modes:
        pipeline.retrieval_mode = mode
        
        # Warmup
        pipeline.query(question, top_k=3, use_llm=False, verbose=False)
        
        total_time = 0
        for _ in range(args.iterations):
            t_start = time.time()
            pipeline.query(question, top_k=3, use_llm=False, verbose=False)
            total_time += (time.time() - t_start)
            
        avg_latency_ms = (total_time / args.iterations) * 1000
        qps = args.iterations / total_time if total_time > 0 else 0
        
        print(f"{mode:<10} | {avg_latency_ms:<20.2f} | {qps:<15.2f}")
    print("="*50)

if __name__ == "__main__":
    main()
