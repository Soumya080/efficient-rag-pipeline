"""
FAISS Flat Index Retriever

Uses FAISS to index and search dense embeddings. Normalizes embeddings
for Cosine Similarity metrics.
"""

import faiss
import numpy as np

class FAISSRetriever:
    """
    Fast vector retrieval using FAISS IndexFlatIP (Inner Product).
    Equivalent to Cosine Similarity when vectors are L2 normalized.
    """
    def __init__(self, dimension):
        """
        Initialize the FAISS Index.
        
        Args:
            dimension: Dimension of the embedding vectors (e.g. 384 for all-MiniLM-L6-v2)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []

    def add_chunks(self, chunk_list):
        """
        L2 normalize chunk embeddings and add them to the index.
        
        Args:
            chunk_list: List of chunk dicts containing 'chunk_embedding'
        """
        self.chunks = chunk_list
        embeddings = np.array([c["chunk_embedding"] for c in chunk_list]).astype('float32')
        
        # Cosine similarity requires normalization
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def retrieve(self, query_embedding, top_k=3):
        """
        Retrieve top-k closest chunks using Normalized Inner Product.
        
        Args:
            query_embedding: Numpy array embedding vector of the query
            top_k: Number of results to return
            
        Returns:
            List of results matching retriever interface
        """
        if self.index.ntotal == 0:
            return []

        q_emb = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(q_emb)
        
        distances, indices = self.index.search(q_emb, top_k)
        
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks) and idx >= 0:
                chunk = self.chunks[idx]
                results.append({
                    "content": chunk["content"],
                    "score": float(score),
                    "chunk_id": chunk["chunk_id"],
                    "chunk_index": chunk["chunk_index"]
                })
        return results
