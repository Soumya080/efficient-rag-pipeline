"""
03_analyze_oracle.py — Oracle analysis to determine if retriever routing has headroom.

THE MOST IMPORTANT SCRIPT IN YOUR RESEARCH.

If oracle_gap > 5%: You have a paper. Proceed with RL routing.
If oracle_gap 2-5%: Marginal. Consider combining with cost/latency analysis.
If oracle_gap < 2%: No headroom. Pivot to a different idea.

Usage:
    cd D:\LLM-RESEARCH-LAB\efficient-rag-pipeline
    python experiments/baseline_v1/03_analyze_oracle.py
"""

import json
import os
import numpy as np
from collections import Counter

RESULTS_DIR = "experiments/baseline_v1/results"
FIGURES_DIR = "experiments/baseline_v1/results/figures"
RETRIEVERS = ["bm25", "dense", "hybrid"]


def load_results(dataset_name):
    path = os.path.join(RESULTS_DIR, f"{dataset_name}_per_query_results.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def compute_oracle(results, k="5", metric="hit"):
    """
    For each query, find which retriever performed best.
    Oracle always picks the best retriever per query.
    
    Returns:
        oracle_score: mean score when always picking the best retriever
        best_fixed: (name, score) of the best single fixed retriever
        per_query_best: list of which retriever was best for each query
        headroom: oracle_score - best_fixed_score
    """
    oracle_scores = []
    per_query_best = []
    per_retriever_scores = {r: [] for r in RETRIEVERS}
    
    for item in results:
        best_score = -1
        best_retriever = None
        
        for ret_name in RETRIEVERS:
            score = item["retrievers"][ret_name]["metrics_at_k"][str(k)][metric]
            per_retriever_scores[ret_name].append(score)
            
            if score > best_score:
                best_score = score
                best_retriever = ret_name
            elif score == best_score and best_retriever is not None:
                # Tie-break: prefer cheaper retriever
                cost_order = {"bm25": 0, "dense": 1, "hybrid": 2}
                if cost_order.get(ret_name, 99) < cost_order.get(best_retriever, 99):
                    best_retriever = ret_name
        
        oracle_scores.append(best_score)
        per_query_best.append(best_retriever)
    
    oracle_mean = np.mean(oracle_scores)
    
    # Find best fixed retriever
    fixed_means = {r: np.mean(scores) for r, scores in per_retriever_scores.items()}
    best_fixed_name = max(fixed_means, key=fixed_means.get)
    best_fixed_score = fixed_means[best_fixed_name]
    
    headroom = oracle_mean - best_fixed_score
    
    return {
        "oracle_mean": oracle_mean,
        "best_fixed_name": best_fixed_name,
        "best_fixed_score": best_fixed_score,
        "headroom_absolute": headroom,
        "headroom_relative_pct": (headroom / best_fixed_score * 100) if best_fixed_score > 0 else 0,
        "per_query_best_counts": dict(Counter(per_query_best)),
        "per_retriever_means": fixed_means,
        "per_retriever_scores": {r: scores for r, scores in per_retriever_scores.items()},
    }


def analyze_disagreement(results, k="5", metric="hit"):
    """
    How often do retrievers disagree on which query they answer correctly?
    High disagreement = more routing opportunity.
    """
    agree_count = 0
    disagree_count = 0
    
    all_correct = 0   # All retrievers got it right
    none_correct = 0  # No retriever got it right
    some_correct = 0  # Some retrievers got it right, others didn't
    
    for item in results:
        scores = {}
        for ret_name in RETRIEVERS:
            scores[ret_name] = item["retrievers"][ret_name]["metrics_at_k"][str(k)][metric]
        
        unique_scores = set(scores.values())
        if len(unique_scores) == 1:
            agree_count += 1
            if list(unique_scores)[0] > 0:
                all_correct += 1
            else:
                none_correct += 1
        else:
            disagree_count += 1
            some_correct += 1
    
    total = len(results)
    return {
        "total_queries": total,
        "all_agree": agree_count,
        "all_agree_pct": agree_count / total * 100,
        "disagree": disagree_count,
        "disagree_pct": disagree_count / total * 100,
        "all_correct": all_correct,
        "all_correct_pct": all_correct / total * 100,
        "none_correct": none_correct,
        "none_correct_pct": none_correct / total * 100,
        "some_correct": some_correct,
        "some_correct_pct": some_correct / total * 100,
    }


def generate_plots(oracle_data, dataset_name):
    """Generate visualization plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        os.makedirs(FIGURES_DIR, exist_ok=True)
        
        # Plot 1: Retriever comparison bar chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart of means
        names = list(oracle_data["per_retriever_means"].keys()) + ["ORACLE"]
        values = list(oracle_data["per_retriever_means"].values()) + [oracle_data["oracle_mean"]]
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        bars = axes[0].bar(names, values, color=colors[:len(names)])
        axes[0].set_ylabel("Hit Rate @ K")
        axes[0].set_title(f"{dataset_name.upper()}: Retriever Comparison")
        axes[0].set_ylim(0, 1.1)
        
        for bar, val in zip(bars, values):
            axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', fontsize=10)
        
        # Pie chart of per-query best retriever
        counts = oracle_data["per_query_best_counts"]
        axes[1].pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%',
                    colors=colors[:len(counts)])
        axes[1].set_title(f"{dataset_name.upper()}: Which Retriever Wins Per Query?")
        
        plt.tight_layout()
        plot_path = os.path.join(FIGURES_DIR, f"{dataset_name}_retriever_comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved plot: {plot_path}")
        
    except ImportError:
        print("  Warning: matplotlib not available, skipping plots")


def main():
    print("="*70)
    print("ORACLE ANALYSIS — Does Retriever Routing Have Headroom?")
    print("="*70)
    
    all_analysis = {}
    
    for ds_name in ["nq", "triviaqa"]:
        results = load_results(ds_name)
        if results is None:
            print(f"\nSkipping {ds_name} (no results found)")
            continue
        
        print(f"\n{'='*70}")
        print(f"DATASET: {ds_name.upper()} ({len(results)} queries)")
        print(f"{'='*70}")
        
        k_values = ["1", "3", "5", "10"]
        available_ks = []
        # Check which k values are available
        if results:
            available_ks = list(results[0]["retrievers"]["bm25"]["metrics_at_k"].keys())
        
        for k in available_ks:
            print(f"\n--- Analysis at K={k} ---")
            
            # Oracle analysis
            oracle = compute_oracle(results, k=k, metric="hit")
            
            print(f"\n  Per-Retriever Hit Rates:")
            for ret, score in oracle["per_retriever_means"].items():
                print(f"    {ret:<10}: {score:.4f} ({score*100:.1f}%)")
            
            print(f"\n  Oracle (always pick best): {oracle['oracle_mean']:.4f} ({oracle['oracle_mean']*100:.1f}%)")
            print(f"  Best Fixed Retriever:     {oracle['best_fixed_name']} = {oracle['best_fixed_score']:.4f}")
            
            print(f"\n  ┌─────────────────────────────────────────────┐")
            print(f"  │  HEADROOM = {oracle['headroom_absolute']:.4f} ({oracle['headroom_relative_pct']:.1f}% relative)  │")
            print(f"  └─────────────────────────────────────────────┘")
            
            if oracle['headroom_relative_pct'] > 5:
                print(f"  → ✓ SIGNIFICANT HEADROOM — Routing policy is justified!")
            elif oracle['headroom_relative_pct'] > 2:
                print(f"  → ~ MARGINAL HEADROOM — Consider cost/latency angle too")
            else:
                print(f"  → ✗ MINIMAL HEADROOM — Routing may not be worth it")
            
            # Per-query best retriever distribution
            print(f"\n  Per-Query Best Retriever Distribution:")
            for ret, count in oracle["per_query_best_counts"].items():
                pct = count / len(results) * 100
                bar = "█" * int(pct / 2)
                print(f"    {ret:<10}: {count:>4} queries ({pct:>5.1f}%) {bar}")
            
            # Disagreement analysis
            disagree = analyze_disagreement(results, k=k, metric="hit")
            print(f"\n  Retriever Agreement Analysis:")
            print(f"    All agree (all correct): {disagree['all_correct']:>4} ({disagree['all_correct_pct']:.1f}%)")
            print(f"    All agree (none correct):{disagree['none_correct']:>4} ({disagree['none_correct_pct']:.1f}%)")
            print(f"    Disagree (routing helps): {disagree['some_correct']:>4} ({disagree['some_correct_pct']:.1f}%)")
            
            # Store for saving
            if ds_name not in all_analysis:
                all_analysis[ds_name] = {}
            all_analysis[ds_name][f"k_{k}"] = {
                "oracle": {
                    "oracle_mean": oracle["oracle_mean"],
                    "best_fixed_name": oracle["best_fixed_name"],
                    "best_fixed_score": oracle["best_fixed_score"],
                    "headroom_absolute": oracle["headroom_absolute"],
                    "headroom_relative_pct": oracle["headroom_relative_pct"],
                    "per_query_best_counts": oracle["per_query_best_counts"],
                    "per_retriever_means": oracle["per_retriever_means"],
                },
                "disagreement": disagree,
            }
        
        # Generate plots for default k
        default_k = "5" if "5" in available_ks else available_ks[-1] if available_ks else "3"
        oracle_for_plot = compute_oracle(results, k=default_k, metric="hit")
        generate_plots(oracle_for_plot, ds_name)
    
    # Save full analysis
    analysis_path = os.path.join(RESULTS_DIR, "oracle_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(all_analysis, f, indent=2)
    print(f"\n✓ Saved oracle analysis: {analysis_path}")
    
    # ============================================================
    # FINAL VERDICT
    # ============================================================
    print("\n" + "="*70)
    print("FINAL VERDICT: SHOULD YOU PURSUE RETRIEVER ROUTING?")
    print("="*70)
    
    for ds_name, ds_analysis in all_analysis.items():
        # Use k=5 as the primary evaluation point
        k_key = "k_5" if "k_5" in ds_analysis else list(ds_analysis.keys())[-1]
        headroom = ds_analysis[k_key]["oracle"]["headroom_relative_pct"]
        disagree_pct = ds_analysis[k_key]["disagreement"]["disagree_pct"]
        
        print(f"\n  {ds_name.upper()}:")
        print(f"    Routing headroom: {headroom:.1f}%")
        print(f"    Disagreement rate: {disagree_pct:.1f}%")
        
        if headroom > 5 and disagree_pct > 15:
            print(f"    VERDICT: ✓ STRONG SIGNAL — Proceed with routing paper!")
        elif headroom > 2 or disagree_pct > 10:
            print(f"    VERDICT: ~ MODERATE SIGNAL — Add cost/latency angle")
        else:
            print(f"    VERDICT: ✗ WEAK SIGNAL — Consider pivoting")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("  If STRONG/MODERATE: Start building RL routing policy")
    print("  If WEAK: Pivot to Idea #7 (Adaptive Hybrid Weights)")
    print("  Share these results with me for detailed next-step planning")
    print("="*70)


if __name__ == "__main__":
    main()
