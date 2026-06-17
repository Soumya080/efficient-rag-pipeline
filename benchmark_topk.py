import os
import argparse
from tqdm import tqdm
from src.pipeline import RAGPipeline

def exact_match(prediction, truth):
    return 1 if truth.lower() in prediction.lower() or prediction.lower() in truth.lower() else 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/knowledge_base/")
    args = parser.parse_args()

    questions = [
        "What is the self-attention mechanism used in?",
        "What optimization algorithms are used in deep learning?",
        "What does FAISS stand for?",
        "How does BM25 improve upon TF-IDF?",
        "What is Q-learning?",
        "What does NLP stand for?",
        "What is algorithmic bias?"
    ]
    answers = [
        "Transformer architecture",
        "Adam and SGD",
        "Facebook AI Similarity Search",
        "term frequency saturation and document length normalization",
        "value-based RL algorithm",
        "Natural Language Processing",
        "inherit and amplify human prejudices"
    ]

    modes = ["dense", "faiss", "sparse", "hybrid"]
    top_ks = [1, 3, 5, 10]

    print(f"Loading pipeline and indexing {args.data}...")
    pipeline = RAGPipeline(chunking_strategy="semantic", retrieval_mode="hybrid")
    if os.path.exists(args.data):
        pipeline.ingest(args.data, verbose=False)
    else:
        print("Data directory not found. Please ensure data/knowledge_base/ exists.")
        return

    print("\n" + "="*50)
    print("           TOP-K SENSITIVITY BENCHMARK           ")
    print("="*50)
    print(f"{'Mode':<10} | {'k=1':<8} | {'k=3':<8} | {'k=5':<8} | {'k=10':<8}")
    print("-" * 50)

    for mode in modes:
        pipeline.retrieval_mode = mode
        row = f"{mode:<10} | "
        for k in top_ks:
            hit_rate = 0
            for q, a in zip(questions, answers):
                res = pipeline.query(q, top_k=k, use_llm=False, verbose=False)
                context_str = " ".join([c.get('content', '') for c in res['retrieved']])
                if exact_match(context_str, a):
                    hit_rate += 1
            hit_rate /= len(questions)
            row += f"{hit_rate:.2f}     | "
        print(row)
    print("="*50)

if __name__ == "__main__":
    main()
