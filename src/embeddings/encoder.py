"""
Embedding encoder for RAG pipeline.

Wraps sentence-transformers model for encoding text into vectors.

"""

from sentence_transformers import SentenceTransformer


class EmbeddingEncoder:
    """
    Wrapper around SentenceTransformer for consistent embedding generation.

    The user's original code loaded model globally and called model.encode().
    This wraps that pattern into a reusable class.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Load the sentence transformer model.

        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2)
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded: {model_name}")
        print(f"Embedding dimension: {self.dimension}")

    def encode(self, text):
        """
        Encode a single text string into an embedding vector.

        Args:
            text: Input string

        Returns:
            numpy array of shape (dimension,)
        """
        return self.model.encode(text)

    def encode_batch(self, texts):
        """
        Encode multiple texts at once (more efficient than one-by-one).

        Args:
            texts: List of strings

        Returns:
            numpy array of shape (len(texts), dimension)
        """
        return self.model.encode(texts)

    def generate_chunk_embeddings(self, chunk_list):
        """
        Add embeddings to chunk dicts.

        Original code from 01_RETRIVER.ipynb:
            def generate_chunk_embeddings(chunk_list):
                for chunk in chunk_list:
                    chunk_embedding = model.encode(chunk["content"])
                    chunk["chunk_embedding"] = chunk_embedding
                return chunk_list

        Args:
            chunk_list: List of chunk dicts (must have 'content' key)

        Returns:
            Same list with 'chunk_embedding' added to each chunk
        """
        for chunk in chunk_list:
            chunk["chunk_embedding"] = self.encode(chunk["content"])
        return chunk_list
