"""
BM25 (Okapi) Sparse Retriever - Built From Scratch

Calculates TF-IDF-based document matching scores.
Optimized for keyword relevance (lexical matching).
"""

import math
from collections import Counter

class BM25Retriever:
    """
    Lexical sparse search using the Okapi BM25 ranking function.
    No scikit-learn, no外部 dependencies. Pure Python/Numpy formula.
    """
    def __init__(self, k1=1.5, b=0.75):
        """
        Args:
            k1: Term frequency scaling factor (higher = limits term saturation)
            b: Document length normalization scaling (0 = none, 1 = full)
        """
        self.k1 = k1
        self.b = b
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.corpus = []
        
    def _tokenize(self, text):
        """Simple lowercase word tokenization and punctuation stripping."""
        cleaned = "".join([c.lower() if c.isalnum() or c.isspace() else " " for c in text])
        return cleaned.split()

    def fit(self, chunk_list):
        """
        Compute corpus statistics (TF, IDF, average document length).
        
        Args:
            chunk_list: List of chunk dicts
        """
        self.corpus = chunk_list
        self.doc_len = []
        self.doc_freqs = []
        nd = len(chunk_list)
        
        df = Counter()
        for chunk in chunk_list:
            tokens = self._tokenize(chunk["content"])
            self.doc_len.append(len(tokens))
            self.doc_freqs.append(Counter(tokens))
            for token in set(tokens):
                df[token] += 1
        
        self.avgdl = sum(self.doc_len) / nd if nd > 0 else 0
        
        # Calculate IDF using standard Okapi BM25 formulation
        for word, freq in df.items():
            self.idf[word] = math.log((nd - freq + 0.5) / (freq + 0.5) + 1.0)

    def retrieve(self, query, top_k=3):
        """
        Score and rank chunks for lexical similarity against a query string.
        
        Args:
            query: Input question/query text
            top_k: Number of top results to return
        """
        if not self.corpus:
            return []

        query_tokens = self._tokenize(query)
        scores = []
        
        for idx in range(len(self.corpus)):
            score = 0.0
            doc_len = self.doc_len[idx]
            freqs = self.doc_freqs[idx]
            
            for token in query_tokens:
                if token not in freqs:
                    continue
                tf = freqs[token]
                idf = self.idf.get(token, 0.0)
                
                # Okapi BM25 formula
                numerator = idf * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += numerator / denominator
            
            scores.append((score, idx))
        
        # Sort descending by BM25 score
        scores.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, idx in scores[:top_k]:
            if score > 0:
                chunk = self.corpus[idx]
                results.append({
                    "content": chunk["content"],
                    "score": score,
                    "chunk_id": chunk["chunk_id"],
                    "chunk_index": chunk["chunk_index"]
                })
        return results
