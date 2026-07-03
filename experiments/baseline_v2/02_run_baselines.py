"""
02_run_baselines.py — Baseline v2: Run BM25, Dense, and Hybrid on real Wikipedia passages.

Identical experiment design to v1 but using the real-passage corpus from v2 data prep.
This is the production-grade experiment for the paper.

Usage:
    cd D:\\LLM-RESEARCH-LAB\\efficient-rag-pipeline
    python experiments/baseline_v2/02_run_baselines.py
    python experiments/baseline_v2/02_run_baselines.py --dataset nq --max_questions 10
    python experiments/baseline_v2/02_run_baselines.py --dataset both --top_k 1 3 5 10 20
"""

import json
import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

from src.embeddings.encoder import EmbeddingEncoder
from src.retrieval.sparse import BM25Retriever
from src.retrieval.dense_faiss import FAISSRetriever
from src.retrieval.hybrid import reciprocal_rank_fusion

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "experiments/baseline_v2/data"
RESULTS_DIR = "experiments/baseline_v2/results"
TOP_K_VALUES = [1, 3, 5, 10, 20]   # Extended k values for thorough analysis
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ============================================================
# METRICS — Standard QA retrieval metrics (paper-grade)
# ============================================================
def normalize_answer(text):
    """Lowercase, strip articles and punctuation for fair comparison."""
    import re
    text = text.lower().strip()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

def exact_match_score(prediction, ground_truths):
    """Standard Exact Match: prediction must exactly equal a ground truth (normalized)."""
    pred_norm = normalize_answer(prediction)
    for gt in ground_truths:
        if normalize_answer(gt) == pred_norm:
            return 1.0
    return 0.0

def substring_match_score(prediction, ground_truths):
    """Substring containment: any ground truth appears in prediction."""
    pred_norm = normalize_answer(prediction)
    for gt in ground_truths:
        if normalize_answer(gt) in pred_norm:
            return 1.0
    return 0.0

def f1_score(prediction, ground_truths):
    """Token-level F1 between prediction and best-matching ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        common = set(pred_tokens) & set(gt_tokens)
        if not common:
            continue
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(gt_tokens) if gt_tokens else 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            best_f1 = max(best_f1, f1)
    return best_f1

def check_hit(retrieved_contents, answers):
    """Check if any answer appears in any retrieved passage (standard Hit@K)."""
    for content in retrieved_contents:
        content_norm = normalize_answer(content)
        for ans in answers:
            if normalize_answer(ans) in content_norm:
                return True
    return False

def compute_mrr(retrieved_contents, answers):
    """Mean Reciprocal Rank — rank of first relevant result."""
    for rank, content in enumerate(retrieved_contents, start=1):
        content_norm = normalize_answer(content)
        for ans in answers:
            if normalize_answer(ans) in content_norm:
                return 1.0 / rank
    return 0.0

def compute_recall_at_k(retrieved_contents, answers, k):
    """Recall@K — fraction of unique answers found in top-k passages."""
    if not answers:
        return 0.0
    context = " ".join(retrieved_contents[:k])
    context_norm = normalize_answer(context)
    found = sum(1 for ans in set(answers) if normalize_answer(ans) in context_norm)
    return found / len(set(answers))


# ============================================================
# RETRIEVER RUNNERS
# ============================================================
def build_chunk_list(corpus_texts):
    """Convert passage strings to chunk dicts matching pipeline format."""
    chunks = []
    for i, text in enumerate(corpus_texts):
        chunks.append({
            "content": text,
            "chunk_id": f"chunk_{i}",
            "chunk_index": i,
        })
    return chunks

def run_single_query_all_retrievers(question, corpus_texts, encoder, top_k=20):
    """
    Run BM25, Dense, and Hybrid on a single question against a corpus.
    Returns per-retriever results.
    """
    chunks = build_chunk_list(corpus_texts)

    # Embed all chunks (batched for speed)
    texts = [c["content"] for c in chunks]
    embeddings = encoder.encode_batch(texts)
    for i, chunk in enumerate(chunks):
        chunk["chunk_embedding"] = embeddings[i]

    # Embed query
    query_embedding = encoder.encode(question)

    results = {}

    # --- BM25 ---
    t0 = time.time()
    bm25 = BM25Retriever()
    bm25.fit(chunks)
    bm25_results = bm25.retrieve(question, top_k=top_k)
    bm25_time = time.time() - t0
    results["bm25"] = {
        "retrieved": [r["content"] for r in bm25_results],
        "scores": [r["score"] for r in bm25_results],
        "latency_ms": bm25_time * 1000,
    }

    # --- Dense (FAISS) ---
    t0 = time.time()
    faiss_ret = FAISSRetriever(encoder.dimension)
    faiss_ret.add_chunks(chunks)
    faiss_results = faiss_ret.retrieve(query_embedding, top_k=top_k)
    faiss_time = time.time() - t0
    results["dense"] = {
        "retrieved": [r["content"] for r in faiss_results],
        "scores": [r["score"] for r in faiss_results],
        "latency_ms": faiss_time * 1000,
    }

    # --- Hybrid (RRF) ---
    t0 = time.time()
    dense_candidates = faiss_ret.retrieve(query_embedding, top_k=max(20, top_k * 2))
    sparse_candidates = bm25.retrieve(question, top_k=max(20, top_k * 2))
    hybrid_results = reciprocal_rank_fusion(dense_candidates, sparse_candidates, top_n=top_k)
    hybrid_time = time.time() - t0
    results["hybrid"] = {
        "retrieved": [r["content"] for r in hybrid_results],
        "scores": [r["score"] for r in hybrid_results],
        "latency_ms": hybrid_time * 1000,
    }

    return results


def run_baseline_experiment(dataset_name, data, encoder, top_k_values):
    """Run all retrievers on all questions, collect per-query results."""
    print(f"\n{'='*60}")
    print(f"RUNNING BASELINE v2: {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"  Questions: {len(data)}")
    print(f"  Top-K values: {top_k_values}")
    print(f"  Retrievers: BM25, Dense (FAISS), Hybrid (RRF)")
    print(f"  Corpus type: Real Wikipedia passages")
    print()

    all_results = []
    gold_type_counts = {"real_wikipedia": 0, "constructed": 0}

    for i, item in enumerate(tqdm(data, desc=f"Evaluating {dataset_name}")):
        question = item["question"]
        answers = item["answers"]
        corpus = item["corpus"]
        gold_type = item.get("gold_type", "unknown")
        gold_type_counts[gold_type] = gold_type_counts.get(gold_type, 0) + 1

        max_k = max(top_k_values)
        retriever_results = run_single_query_all_retrievers(
            question, corpus, encoder, top_k=max_k
        )

        query_result = {
            "query_id": i,
            "question": question,
            "answers": answers,
            "corpus_size": len(corpus),
            "gold_type": gold_type,
            "retrievers": {},
        }

        for ret_name in ["bm25", "dense", "hybrid"]:
            ret_data = retriever_results[ret_name]
            query_result["retrievers"][ret_name] = {
                "latency_ms": ret_data["latency_ms"],
                "metrics_at_k": {},
            }

            for k in top_k_values:
                top_k_contents = ret_data["retrieved"][:k]

                hit = check_hit(top_k_contents, answers)
                mrr = compute_mrr(ret_data["retrieved"][:k], answers)
                recall = compute_recall_at_k(ret_data["retrieved"], answers, k)

                # F1/EM on concatenated retrieved text
                concat_text = " ".join(top_k_contents)
                f1 = f1_score(concat_text, answers)
                em = substring_match_score(concat_text, answers)

                query_result["retrievers"][ret_name]["metrics_at_k"][str(k)] = {
                    "hit": hit,
                    "mrr": mrr,
                    "recall": recall,
                    "f1": f1,
                    "em": em,
                }

        all_results.append(query_result)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(data)} queries...")

    print(f"\n  Gold passage types: {gold_type_counts}")
    return all_results


def print_summary(results, dataset_name, k=5):
    """Print a summary table of results at a specific k."""
    print(f"\n{'='*75}")
    print(f"RESULTS SUMMARY: {dataset_name.upper()} (top_k={k})")
    print(f"{'='*75}")

    retrievers = ["bm25", "dense", "hybrid"]
    metrics = ["hit", "mrr", "recall", "f1", "em"]

    header = f"{'Retriever':<12}"
    for m in metrics:
        header += f" | {m.upper():>8}"
    header += f" | {'Latency':>10}"
    print(header)
    print("-" * 75)

    for ret in retrievers:
        row = f"{ret:<12}"
        for m in metrics:
            values = [r["retrievers"][ret]["metrics_at_k"][str(k)][m] for r in results]
            mean_val = np.mean(values)
            if m == "hit":
                row += f" | {mean_val:>8.1%}"
            else:
                row += f" | {mean_val:>8.4f}"

        latencies = [r["retrievers"][ret]["latency_ms"] for r in results]
        row += f" | {np.mean(latencies):>8.2f}ms"
        print(row)

    print("-" * 75)

    # Quick oracle preview
    oracle_hits = []
    for r in results:
        any_hit = any(
            r["retrievers"][ret]["metrics_at_k"][str(k)]["hit"]
            for ret in retrievers
        )
        oracle_hits.append(int(any_hit))

    best_fixed_hit = max(
        np.mean([r["retrievers"][ret]["metrics_at_k"][str(k)]["hit"] for r in results])
        for ret in retrievers
    )
    oracle_hit = np.mean(oracle_hits)
    headroom = oracle_hit - best_fixed_hit

    print(f"\n  Oracle Hit@{k}: {oracle_hit:.1%} | Best Fixed: {best_fixed_hit:.1%} | "
          f"HEADROOM: {headroom:.1%} ({headroom/max(0.001,best_fixed_hit)*100:.1f}% relative)")


def main():
    parser = argparse.ArgumentParser(description="Run baseline v2 retrieval experiments")
    parser.add_argument("--dataset", type=str, default="both", choices=["nq", "triviaqa", "both"])
    parser.add_argument("--top_k", type=int, nargs="+", default=TOP_K_VALUES)
    parser.add_argument("--max_questions", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading embedding model...")
    encoder = EmbeddingEncoder(EMBEDDING_MODEL)

    datasets_to_run = []
    if args.dataset in ["nq", "both"]:
        nq_path = os.path.join(DATA_DIR, "nq_prepared.json")
        if os.path.exists(nq_path):
            with open(nq_path, "r") as f:
                nq_data = json.load(f)
            if args.max_questions:
                nq_data = nq_data[:args.max_questions]
            datasets_to_run.append(("nq", nq_data))
        else:
            print(f"WARNING: {nq_path} not found. Run 01_prepare_data.py first.")

    if args.dataset in ["triviaqa", "both"]:
        tqa_path = os.path.join(DATA_DIR, "triviaqa_prepared.json")
        if os.path.exists(tqa_path):
            with open(tqa_path, "r") as f:
                tqa_data = json.load(f)
            if args.max_questions:
                tqa_data = tqa_data[:args.max_questions]
            datasets_to_run.append(("triviaqa", tqa_data))
        else:
            print(f"WARNING: {tqa_path} not found. Run 01_prepare_data.py first.")

    if not datasets_to_run:
        print("No datasets found. Run 01_prepare_data.py first.")
        return

    for ds_name, ds_data in datasets_to_run:
        t_start = time.time()
        results = run_baseline_experiment(ds_name, ds_data, encoder, args.top_k)
        total_time = time.time() - t_start

        output_path = os.path.join(RESULTS_DIR, f"{ds_name}_per_query_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n>>> Saved per-query results: {output_path}")

        for k in args.top_k:
            print_summary(results, ds_name, k=k)

        print(f"\nTotal time for {ds_name}: {total_time:.1f}s")

    print(f"\n>>> ALL BASELINES v2 COMPLETE")
    print(f"  Next step: python experiments/baseline_v2/03_analyze_oracle.py")


if __name__ == "__main__":
    main()
