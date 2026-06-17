"""
Hybrid Retrieval (Reciprocal Rank Fusion)

Combines dense (vector-based) and sparse (keyword-based) search results
using Reciprocal Rank Fusion (RRF) algorithm.
"""

def reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_n=3):
    """
    Ranks documents by combining their ordinal rankings from dense and sparse retrieval.
    
    Formula:
        RRF_Score(d ∈ D) = Σ (1 / (k + r_m(d)))
        where r_m(d) is the rank of doc d in retriever system m.
        
    Args:
        dense_results: List of retrieved chunks from dense retriever
        sparse_results: List of retrieved chunks from sparse retriever
        k: Constant parameters to penalize low-ranked outputs (default: 60)
        top_n: Final combined chunks to return
        
    Returns:
        List of fused search results sorted by combined RRF score.
    """
    rrf_scores = {}
    content_map = {}

    def add_rankings(results):
        for rank, hit in enumerate(results, start=1):
            chunk_id = hit["chunk_id"]
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
                content_map[chunk_id] = hit
            
            # Reciprocal rank math
            rrf_scores[chunk_id] += 1.0 / (k + rank)

    add_rankings(dense_results)
    add_rankings(sparse_results)

    # Sort candidates by combined score
    fused_sorted = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    final_results = []
    for chunk_id, fused_score in fused_sorted[:top_n]:
        # Duplicate hit information and set fused rank score
        hit = content_map[chunk_id].copy()
        hit["score"] = fused_score
        final_results.append(hit)

    return final_results
