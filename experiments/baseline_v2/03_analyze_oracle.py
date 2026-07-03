"""
03_analyze_oracle.py — Baseline v2: Oracle analysis with comprehensive paper-grade output.

Analyzes per-query results from 02_run_baselines.py to compute:
1. Per-retriever Hit@K, MRR, Recall at multiple K values
2. Oracle headroom (routing potential)
3. Per-query best retriever distribution
4. Disagreement analysis
5. Head-to-head comparisons (BM25 vs Dense)
6. Visualization figures for the paper

Usage:
    python experiments/baseline_v2/03_analyze_oracle.py
"""

import json
import os
import sys
import numpy as np

# Add project root
sys.path.insert(0, os.path.abspath("."))

RESULTS_DIR = "experiments/baseline_v2/results"
FIGURES_DIR = "experiments/baseline_v2/results/figures"

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available, skipping plots")


def load_results(dataset_name):
    path = os.path.join(RESULTS_DIR, f"{dataset_name}_per_query_results.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def analyze_dataset(results, dataset_name, k_values):
    """Full oracle analysis at all k values."""
    retrievers = ["bm25", "dense", "hybrid"]
    n = len(results)

    print(f"\n{'='*70}")
    print(f"DATASET: {dataset_name.upper()} ({n} queries)")
    print(f"{'='*70}")

    analysis = {}

    for k in k_values:
        k_str = str(k)
        print(f"\n--- Analysis at K={k} ---\n")

        # Compute per-retriever means
        per_ret_hits = {}
        per_ret_mrrs = {}
        per_ret_recalls = {}

        for ret in retrievers:
            hits = [r["retrievers"][ret]["metrics_at_k"].get(k_str, {}).get("hit", False)
                    for r in results]
            mrrs = [r["retrievers"][ret]["metrics_at_k"].get(k_str, {}).get("mrr", 0.0)
                    for r in results]
            recalls = [r["retrievers"][ret]["metrics_at_k"].get(k_str, {}).get("recall", 0.0)
                       for r in results]

            per_ret_hits[ret] = np.mean([int(h) for h in hits])
            per_ret_mrrs[ret] = np.mean(mrrs)
            per_ret_recalls[ret] = np.mean(recalls)

            print(f"  {ret:10s}: Hit={per_ret_hits[ret]:.1%}  MRR={per_ret_mrrs[ret]:.4f}  "
                  f"Recall={per_ret_recalls[ret]:.4f}")

        # Oracle computation
        oracle_hits = []
        per_query_best = []
        best_counts = {}

        for r in results:
            best_hit = 0
            best_ret = "tie"
            best_mrr = -1

            for ret in retrievers:
                h = int(r["retrievers"][ret]["metrics_at_k"].get(k_str, {}).get("hit", False))
                m = r["retrievers"][ret]["metrics_at_k"].get(k_str, {}).get("mrr", 0.0)

                if h > best_hit or (h == best_hit and m > best_mrr):
                    best_hit = h
                    best_ret = ret
                    best_mrr = m

            oracle_hits.append(best_hit)
            per_query_best.append(best_ret)
            best_counts[best_ret] = best_counts.get(best_ret, 0) + 1

        oracle_rate = np.mean(oracle_hits)
        best_fixed = max(per_ret_hits.values())
        best_fixed_name = max(per_ret_hits, key=per_ret_hits.get)
        headroom_abs = oracle_rate - best_fixed
        headroom_rel = headroom_abs / max(best_fixed, 0.001) * 100

        print(f"\n  Oracle Hit@{k}: {oracle_rate:.1%}")
        print(f"  Best Fixed:    {best_fixed_name} = {best_fixed:.1%}")
        print(f"\n  ┌─────────────────────────────────────────────┐")
        print(f"  │  HEADROOM = {headroom_abs:.4f} ({headroom_rel:.1f}% relative)  │")
        print(f"  └─────────────────────────────────────────────┘")

        if headroom_rel >= 5:
            print(f"  → ✓ STRONG HEADROOM — Routing is justified!")
        elif headroom_rel >= 2:
            print(f"  → ~ MARGINAL HEADROOM — Consider cost/latency angle too")
        else:
            print(f"  → ✗ MINIMAL HEADROOM — Routing may not be worth it")

        # Per-query best distribution
        print(f"\n  Per-Query Best Retriever Distribution:")
        for ret in sorted(best_counts, key=best_counts.get, reverse=True):
            bar = "█" * int(best_counts[ret] / n * 50)
            print(f"    {ret:10s}: {best_counts[ret]:>4d} queries ({best_counts[ret]/n*100:>5.1f}%) {bar}")

        # Disagreement analysis
        all_correct = 0
        none_correct = 0
        disagree = 0

        for r in results:
            hits_per_ret = [
                int(r["retrievers"][ret]["metrics_at_k"].get(k_str, {}).get("hit", False))
                for ret in retrievers
            ]
            if all(h == 1 for h in hits_per_ret):
                all_correct += 1
            elif all(h == 0 for h in hits_per_ret):
                none_correct += 1
            else:
                disagree += 1

        print(f"\n  Retriever Agreement Analysis:")
        print(f"    All agree (all correct):   {all_correct} ({all_correct/n*100:.1f}%)")
        print(f"    All agree (none correct):  {none_correct} ({none_correct/n*100:.1f}%)")
        print(f"    Disagree (routing helps):  {disagree} ({disagree/n*100:.1f}%)")

        # Head-to-head: BM25 vs Dense
        bm25_wins = 0
        dense_wins = 0
        ties = 0
        for r in results:
            bm25_h = int(r["retrievers"]["bm25"]["metrics_at_k"].get(k_str, {}).get("hit", False))
            dense_h = int(r["retrievers"]["dense"]["metrics_at_k"].get(k_str, {}).get("hit", False))
            if bm25_h > dense_h:
                bm25_wins += 1
            elif dense_h > bm25_h:
                dense_wins += 1
            else:
                ties += 1

        print(f"\n  Head-to-Head (BM25 vs Dense):")
        print(f"    BM25 wins:  {bm25_wins} ({bm25_wins/n*100:.1f}%)")
        print(f"    Dense wins: {dense_wins} ({dense_wins/n*100:.1f}%)")
        print(f"    Ties:       {ties} ({ties/n*100:.1f}%)")

        # Gold type analysis (unique to v2)
        gold_types = {}
        for r in results:
            gt = r.get("gold_type", "unknown")
            if gt not in gold_types:
                gold_types[gt] = {"total": 0}
                for ret in retrievers:
                    gold_types[gt][ret] = []

            gold_types[gt]["total"] += 1
            for ret in retrievers:
                h = int(r["retrievers"][ret]["metrics_at_k"].get(k_str, {}).get("hit", False))
                gold_types[gt][ret].append(h)

        if len(gold_types) > 1:
            print(f"\n  Performance by Gold Passage Type:")
            for gt, data in gold_types.items():
                print(f"    {gt} ({data['total']} queries):")
                for ret in retrievers:
                    if data[ret]:
                        hit = np.mean(data[ret])
                        print(f"      {ret}: {hit:.1%}")

        analysis[f"k_{k}"] = {
            "oracle": {
                "oracle_mean": round(oracle_rate, 4),
                "best_fixed_name": best_fixed_name,
                "best_fixed_score": round(best_fixed, 4),
                "headroom_absolute": round(headroom_abs, 4),
                "headroom_relative_pct": round(headroom_rel, 2),
                "per_query_best_counts": best_counts,
                "per_retriever_means": {r: round(v, 4) for r, v in per_ret_hits.items()},
                "per_retriever_mrrs": {r: round(v, 4) for r, v in per_ret_mrrs.items()},
            },
            "disagreement": {
                "total_queries": n,
                "all_correct": all_correct,
                "all_correct_pct": round(all_correct/n*100, 1),
                "none_correct": none_correct,
                "none_correct_pct": round(none_correct/n*100, 1),
                "disagree": disagree,
                "disagree_pct": round(disagree/n*100, 1),
                "bm25_exclusive_wins": bm25_wins,
                "dense_exclusive_wins": dense_wins,
            }
        }

    return analysis


def plot_comparison(analysis, dataset_name):
    """Generate comparison plots for the paper."""
    if not HAS_MPL:
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)

    k_values = sorted([int(k.split("_")[1]) for k in analysis.keys()])
    retrievers = ["bm25", "dense", "hybrid"]
    colors = {"bm25": "#e74c3c", "dense": "#3498db", "hybrid": "#2ecc71"}

    # Plot 1: Hit@K comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Hit rates
    ax = axes[0]
    for ret in retrievers:
        hits = [analysis[f"k_{k}"]["oracle"]["per_retriever_means"][ret] for k in k_values]
        ax.plot(k_values, hits, 'o-', color=colors[ret], label=ret.upper(), linewidth=2, markersize=6)

    oracle_hits = [analysis[f"k_{k}"]["oracle"]["oracle_mean"] for k in k_values]
    ax.plot(k_values, oracle_hits, 's--', color='#9b59b6', label='Oracle', linewidth=2, markersize=6)

    ax.set_xlabel('K (Top-K)', fontsize=12)
    ax.set_ylabel('Hit Rate', fontsize=12)
    ax.set_title(f'{dataset_name.upper()} — Hit@K', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Headroom
    ax = axes[1]
    headrooms = [analysis[f"k_{k}"]["oracle"]["headroom_relative_pct"] for k in k_values]
    disagree = [analysis[f"k_{k}"]["disagreement"]["disagree_pct"] for k in k_values]

    ax.bar(range(len(k_values)), headrooms, color='#e74c3c', alpha=0.7, label='Headroom %')
    ax.set_xlabel('K', fontsize=12)
    ax.set_ylabel('Relative Headroom (%)', fontsize=12, color='#e74c3c')
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_title(f'{dataset_name.upper()} — Oracle Headroom & Disagreement', fontsize=13)

    ax2 = ax.twinx()
    ax2.plot(range(len(k_values)), disagree, 'o-', color='#3498db', linewidth=2, label='Disagree %')
    ax2.set_ylabel('Disagreement Rate (%)', fontsize=12, color='#3498db')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"{dataset_name}_v2_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved plot: {path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    all_analysis = {}
    k_values = [1, 3, 5, 10, 20]

    for ds in ["nq", "triviaqa"]:
        results = load_results(ds)
        if results is None:
            print(f"No results for {ds}. Run 02_run_baselines.py first.")
            continue

        analysis = analyze_dataset(results, ds, k_values)
        all_analysis[ds] = analysis
        plot_comparison(analysis, ds)

    # Save combined analysis
    if all_analysis:
        output_path = os.path.join(RESULTS_DIR, "oracle_analysis.json")
        with open(output_path, "w") as f:
            json.dump(all_analysis, f, indent=2)
        print(f"\n✓ Saved oracle analysis: {output_path}")

    # Final verdict
    if all_analysis:
        print(f"\n{'='*70}")
        print(f"FINAL VERDICT: SHOULD YOU PURSUE RETRIEVER ROUTING?")
        print(f"{'='*70}")

        for ds, analysis in all_analysis.items():
            k5 = analysis.get("k_5", analysis.get("k_3", {}))
            headroom = k5.get("oracle", {}).get("headroom_relative_pct", 0)
            disagree = k5.get("disagreement", {}).get("disagree_pct", 0)

            print(f"\n  {ds.upper()}:")
            print(f"    Routing headroom at K=5: {headroom:.1f}%")
            print(f"    Disagreement rate:       {disagree:.1f}%")

            if headroom >= 5:
                print(f"    VERDICT: ✓ STRONG SIGNAL — Proceed with RL routing paper!")
            elif headroom >= 2:
                print(f"    VERDICT: ~ MODERATE — Add cost/latency angle")
            else:
                print(f"    VERDICT: ✗ WEAK — Consider pivoting")


if __name__ == "__main__":
    main()
