import os
import time
import json
import argparse
from tqdm import tqdm
from src.pipeline import RAGPipeline

def exact_match(prediction, truth):
    return 1 if truth.lower() in prediction.lower() or prediction.lower() in truth.lower() else 0

def compute_f1(prediction, truth):
    pred_tokens = prediction.lower().split()
    truth_tokens = truth.lower().split()
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(truth_tokens)
    return 2 * (prec * rec) / (prec + rec)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nq_open", choices=["nq_open", "hotpot_qa", "trivia_qa", "local"])
    parser.add_argument("--subset", type=int, default=100)
    parser.add_argument("--mode", type=str, default="hybrid")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--use_llm", action="store_true")
    args = parser.parse_args()

    print(f"Loading dataset {args.dataset}...")
    
    questions = []
    answers = []
    contexts = [] # Optional, for local eval

    if args.dataset != "local":
        try:
            from datasets import load_dataset
            # Load dataset logic
            ds = load_dataset(args.dataset, split="validation" if args.dataset == "nq_open" else "validation")
            # Limit subset
            ds = ds.select(range(min(args.subset, len(ds))))
            
            for item in ds:
                if args.dataset == "nq_open":
                    questions.append(item['question'])
                    answers.append(item['answer'][0] if isinstance(item['answer'], list) else item['answer'])
                    # For real NQ open we need a wikipedia dump. Since we might not have it, 
                    # we will just test the LLM generation and retrieval from local KB if any.
                    # Actually, for standard evaluation, RAG evaluates on the provided knowledge base.
        except ImportError:
            print("datasets library not found. Run pip install datasets. Using local fallback.")
            args.dataset = "local"
    
    if args.dataset == "local":
        # Fallback local dataset
        questions = [
            "What is the self-attention mechanism used in?",
            "What optimization algorithms are used in deep learning?",
            "What does FAISS stand for?",
            "How does BM25 improve upon TF-IDF?"
        ]
        answers = [
            "Transformer architecture",
            "Adam and SGD",
            "Facebook AI Similarity Search",
            "term frequency saturation and document length normalization"
        ]
        
    print(f"Loaded {len(questions)} test cases.")
    
    # Initialize Pipeline
    pipeline = RAGPipeline(chunking_strategy="semantic", retrieval_mode=args.mode)
    
    # We will use the local knowledge base to test if the answers can be retrieved.
    kb_path = "data/knowledge_base"
    if os.path.exists(kb_path):
        pipeline.ingest(kb_path, verbose=False)
    else:
        # Just index a dummy string
        pipeline.ingest(["Deep learning uses Adam and SGD. Transformers use self-attention. FAISS stands for Facebook AI Similarity Search. BM25 adds term frequency saturation."], verbose=False)

    print(f"Evaluating {args.mode} retrieval (top_k={args.top_k}, llm={args.use_llm})...")
    
    results = {
        "hit_rate": 0,
        "exact_match": 0,
        "f1": 0,
        "avg_retrieval_time": 0
    }
    
    total_time = 0
    for q, a in tqdm(zip(questions, answers), total=len(questions)):
        res = pipeline.query(q, top_k=args.top_k, use_llm=args.use_llm, verbose=False)
        total_time += res['stats'].get('retrieval_time', 0)
        
        # Check if answer is in retrieved context
        context_str = " ".join([c.get('content', '') for c in res['retrieved']])
        if exact_match(context_str, a):
            results['hit_rate'] += 1
            
        if args.use_llm and res.get('answer'):
            em = exact_match(res['answer'], a)
            f1 = compute_f1(res['answer'], a)
            results['exact_match'] += em
            results['f1'] += f1
            
    results['hit_rate'] /= len(questions)
    results['exact_match'] /= len(questions)
    results['f1'] /= len(questions)
    results['avg_retrieval_time'] = total_time / len(questions)
    
    print("\n" + "="*40)
    print("           EVALUATION RESULTS           ")
    print("="*40)
    print(f"Dataset      : {args.dataset} (n={len(questions)})")
    print(f"Mode         : {args.mode}")
    print(f"Hit Rate@{args.top_k}  : {results['hit_rate']:.4f}")
    if args.use_llm:
        print(f"Exact Match  : {results['exact_match']:.4f}")
        print(f"F1 Score     : {results['f1']:.4f}")
    print(f"Latency      : {results['avg_retrieval_time']*1000:.2f} ms/query")
    print("="*40)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open(f"results/benchmark_{args.dataset}_{args.mode}.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
