"""
Semantic Chunking

Groups sentences based on embedding similarity using a running centroid.
Sentences that are semantically similar stay in the same chunk.
When similarity drops below threshold, a new chunk starts.



NOTE: This chunker requires an embedding encoder (unlike naive/sentence).
      Pass the encoder when calling chunk().
"""

import uuid
from src.chunking.utils import sentence_splitter, cosine_similarity


def chunk(text, encoder, threshold=0.3, max_chunk_size=200, alpha=0.5):
    """
    Semantic chunking using embedding similarity with running centroid.

    Pipeline:
        1. Split text into sentences  (sentence_splitter)
        2. Encode each sentence        (encoder.encode)
        3. Group by similarity          (cosine_similarity to centroid)

    Args:
        text: Input text to chunk
        encoder: EmbeddingEncoder instance (needed for embeddings)
        threshold: Minimum cosine similarity to stay in same chunk
        max_chunk_size: Maximum character length per chunk
        alpha: Weight for centroid update (0.5 = equal weight old/new)

    Returns:
        List of chunk dicts with chunk_id, content, chunk_index,
        sentence_count, chunk_length, avg_similarity
    """
    # Step 1: split into sentences
    sentences = sentence_splitter(text)

    # Step 2: build sentence objects with embeddings
    sentence_objects = []
    for sentence in sentences:
        embedding = encoder.encode(sentence)
        sentence_objects.append({
            "sentence": sentence,
            "embedding": embedding
        })

    # Step 3: semantic grouping
    chunk_list = []
    current_chunk = []
    embedding_list = []
    similarity_scores = []
    current_length = 0
    centroid = None

    for i in range(len(sentence_objects)):
        current_sentence = sentence_objects[i]["sentence"]
        current_embedding = sentence_objects[i]["embedding"]
        sentence_length = len(current_sentence)

        # first sentence always starts a new chunk
        if len(current_chunk) == 0:
            current_chunk.append(current_sentence)
            embedding_list.append(current_embedding)
            centroid = current_embedding
            current_length += sentence_length
            continue

        # compute similarity to chunk centroid using cosine_simliarity 
        similarity = cosine_similarity(centroid, current_embedding)

        # if similar enough AND fits in size limit, add to current chunk / both needs to statisfy 
        if similarity >= threshold and current_length + sentence_length <= max_chunk_size:
            current_chunk.append(current_sentence)
            embedding_list.append(current_embedding)

            # update centroid with exponential moving average
            centroid = (alpha * centroid + (1 - alpha) * current_embedding)  # centroid calcultor

            similarity_scores.append(similarity)
            current_length += sentence_length

        # otherwise, finalize current chunk and start new 
        else:
            chunk_content = " ".join(current_chunk)
            chunk_data = {
                "chunk_id": str(uuid.uuid4()),
                "content": chunk_content,
                "chunk_index": len(chunk_list),
                "sentence_count": len(current_chunk),
                "chunk_length": len(chunk_content),
                "avg_similarity": (
                    sum(similarity_scores) / len(similarity_scores)
                    if len(similarity_scores) > 0
                    else 0
                ),
            }
            chunk_list.append(chunk_data)

            # reset for new chunk
            current_chunk = [current_sentence]
            embedding_list = [current_embedding]
            centroid = current_embedding
            current_length = sentence_length
            similarity_scores = []

    # finalize last chunk
    if len(current_chunk) > 0:
        chunk_content = " ".join(current_chunk)
        chunk_data = {
            "chunk_id": str(uuid.uuid4()),
            "content": chunk_content,
            "chunk_index": len(chunk_list),
            "sentence_count": len(current_chunk),
            "chunk_length": len(chunk_content),
            "avg_similarity": (
                sum(similarity_scores) / len(similarity_scores)
                if len(similarity_scores) > 0
                else 0
            ),
        }
        chunk_list.append(chunk_data)

    return chunk_list

 
# STANDALONE TEST
if __name__ == "__main__":

    from src.embeddings.encoder import EmbeddingEncoder

    test_text = "Neural networks process information. Deep learning uses neural networks extensively. AI systems automate decision making. Football teams analyze player statistics. Football tournaments attract millions of fans. Quantum mechanics studies particles."

    print("=" * 60)
    print("SEMANTIC CHUNKING TEST")
    print("=" * 60)

    enc = EmbeddingEncoder()
    results = chunk(test_text, encoder=enc, threshold=0.3, max_chunk_size=200)

    for i, c in enumerate(results):
        print(f"\nChunk {i}:")
        print(f"  Sentences:  {c['sentence_count']}")
        print(f"  Length:     {c['chunk_length']}")
        print(f"  Avg Sim:   {c['avg_similarity']:.3f}")
        print(f"  Text:       \"{c['content']}\"")

    print(f"\nTotal chunks: {len(results)}")
