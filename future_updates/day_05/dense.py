"""
Dense Retriever — cosine similarity based.

Retrieves top-k chunks by computing cosine similarity between
the query embedding and all chunk embeddings.

Also includes neighbor expansion and context building.



"""

def retrieve(query, chunk_list, encoder, top_k=3, min_similarity=0.2):
    """
    Retrieve top-k most relevant chunks for a query.

    User's original: retriever_v2()

    Args:
        query: User's question string
        chunk_list: List of chunk dicts (must have 'chunk_embedding')
        encoder: EmbeddingEncoder instance
        top_k: Number of top chunks to return
        min_similarity: Minimum cosine similarity to include

    Returns:
        List of result dicts with content, score, chunk_id, chunk_index
    """
    # encode query
    query_embedding = encoder.encode(query)

    # score all chunks
    retrieval_scores = []

    for i, chunk in enumerate(chunk_list):
        similarity = cosine_similarity(
            query_embedding,
            chunk["chunk_embedding"]
        )

        # similarity filter
        if similarity >= min_similarity:
            retrieval_scores.append({
                "content": chunk["content"],
                "score": similarity,
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"]
            })

    # sort descending by score
    retrieval_scores.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # deduplicate by chunk_index
    unique_scores = []
    seen_indices = set()

    for item in retrieval_scores:
        idx = item["chunk_index"]
        if idx not in seen_indices:
            seen_indices.add(idx)
            unique_scores.append(item)

    # top-k
    top_chunks = unique_scores[:top_k]

    if len(top_chunks) == 0:
        print("\nNo relevant chunks found.\n")
        return []

    return top_chunks


def expand_neighbors(top_chunks, chunk_list):
    """
    Expand retrieved chunks with their prev/next neighbors.

    User's original: expand_neighbors()

    Args:
        top_chunks: List of retrieved chunk results
        chunk_list: Full list of all chunks

    Returns:
        List of chunks including neighbors, sorted by chunk_index
    """
    expanded_chunks = []
    seen_indices = set()

    for chunk in top_chunks:
        current_index = chunk["chunk_index"]

        # previous chunk
        if current_index - 1 >= 0:
            prev_chunk = chunk_list[current_index - 1]
            if prev_chunk["chunk_index"] not in seen_indices:
                expanded_chunks.append(prev_chunk)
                seen_indices.add(prev_chunk["chunk_index"])

        # current chunk
        current_chunk = chunk_list[current_index]
        if current_chunk["chunk_index"] not in seen_indices:
            expanded_chunks.append(current_chunk)
            seen_indices.add(current_chunk["chunk_index"])

        # next chunk
        if current_index + 1 < len(chunk_list):
            next_chunk = chunk_list[current_index + 1]
            if next_chunk["chunk_index"] not in seen_indices:
                expanded_chunks.append(next_chunk)
                seen_indices.add(next_chunk["chunk_index"])

    # preserve original document order
    expanded_chunks.sort(
        key=lambda x: x["chunk_index"]
    )

    return expanded_chunks


def build_context(retrieved_chunks):
    """
    Build a context string from retrieved chunks.

    User's original: build_context()

    Args:
        retrieved_chunks: List of chunk dicts

    Returns:
        Single context string
    """
    if len(retrieved_chunks) == 0:
        return "No relevant context found."

    retrieved_chunks.sort(
        key=lambda x: x["chunk_index"]
    )

    context = ""
    for chunk in retrieved_chunks:
        context += chunk["content"]
        context += "\n\n"

    return context
