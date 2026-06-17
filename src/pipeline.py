"""
RAG Pipeline — Fully Integrated Version with Hybrid Retrieval

Integrates:
- Chunking: Naive, Sentence, Semantic
- Retrieval: Dense (Brute-Force), Dense (FAISS), Sparse (BM25), Hybrid (FAISS + BM25)
"""

import time
import os
import glob

from src.embeddings.encoder import EmbeddingEncoder
from src.chunking import naive, sentence, semantic
from src.retrieval.dense import retrieve as brute_force_dense, expand_neighbors, build_context
from src.retrieval.dense_faiss import FAISSRetriever
from src.retrieval.sparse import BM25Retriever
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.generation.generator import build_rag_prompt, generate_answer

CHUNKING_STRATEGIES = {
    "naive": naive,
    "sentence": sentence,
    "semantic": semantic,
}

class RAGPipeline:
    def __init__(self, chunking_strategy="semantic", retrieval_mode="hybrid", model_name="all-MiniLM-L6-v2"):
        """
        Args:
            chunking_strategy: "naive", "sentence", or "semantic"
            retrieval_mode: "dense" (brute-force), "faiss" (fast dense), "sparse" (BM25), "hybrid" (RRF)
            model_name: Embedding encoder model
        """
        if chunking_strategy not in CHUNKING_STRATEGIES:
            raise ValueError(f"Unknown chunking: {chunking_strategy}")
            
        self.chunking_strategy = chunking_strategy
        self.chunker = CHUNKING_STRATEGIES[chunking_strategy]
        self.retrieval_mode = retrieval_mode
        self.model_name = model_name
        
        print(f"[Pipeline] Initializing Encoder: {model_name}...")
        self.encoder = EmbeddingEncoder(model_name)
        
        # Sparse module instantiation
        self.sparse_retriever = BM25Retriever()
        
        # Dense FAISS module instantiation
        self.faiss_retriever = FAISSRetriever(self.encoder.dimension)
        
        self.chunks = []
        
        print(f"[Pipeline] Setup Config: Chunker='{chunking_strategy}', Mode='{retrieval_mode}'\n")

    def ingest(self, source, chunk_kwargs=None, verbose=True):
        """
        Loads documents from strings, files, or folders.
        Chunks once, embeds once, builds FAISS + BM25 indexes.
        """
        if chunk_kwargs is None:
            chunk_kwargs = {}
            
        t_start = time.time()
        
        documents = []
        # Handle source type (string, file path, directory path)
        if isinstance(source, list):
            documents = source
        elif os.path.isdir(source):
            # Load all txt, md files in dir
            for ext in ("*.txt", "*.md"):
                for filepath in glob.glob(os.path.join(source, ext)):
                    with open(filepath, "r", encoding="utf-8") as f:
                        documents.append(f.read())
        elif os.path.isfile(source):
            with open(source, "r", encoding="utf-8") as f:
                documents.append(f.read())
        elif isinstance(source, str):
            documents.append(source)
        else:
            raise ValueError("source must be a string, list of strings, file path, or directory path")
            
        if verbose:
            print(f"[Pipeline] Loaded {len(documents)} documents.")
            
        # 1. Chunk all documents
        all_chunks = []
        for doc in documents:
            if self.chunking_strategy == "semantic":
                doc_chunks = self.chunker.chunk(doc, encoder=self.encoder, **chunk_kwargs)
            else:
                doc_chunks = self.chunker.chunk(doc, **chunk_kwargs)
            all_chunks.extend(doc_chunks)
            
        # Reassign global chunk index so expand_neighbors works across the entire knowledge base
        for idx, chunk_dict in enumerate(all_chunks):
            chunk_dict["chunk_index"] = idx
            
        if verbose:
            print(f"[Pipeline] Generated {len(all_chunks)} total chunks.")
            
        # 2. Embed all chunks
        t = time.time()
        self.chunks = self.encoder.generate_chunk_embeddings(all_chunks)
        embedding_time = time.time() - t
        
        # 3. Build Indexes
        t = time.time()
        # Reset retrievers
        self.sparse_retriever = BM25Retriever()
        self.faiss_retriever = FAISSRetriever(self.encoder.dimension)
        
        self.sparse_retriever.fit(self.chunks)
        self.faiss_retriever.add_chunks(self.chunks)
        index_time = time.time() - t
        
        if verbose:
            print(f"[Pipeline] Built indexes in {index_time:.4f}s (Embedding: {embedding_time:.4f}s)")
            
        return {"num_docs": len(documents), "num_chunks": len(self.chunks), "time": time.time() - t_start}

    def query(self, question, top_k=3, min_similarity=0.2, use_expansion=True, use_llm=True, llm_model="phi3", verbose=True):
        """
        Queries the pre-built index. No re-chunking, no re-embedding.
        Returns retrieved chunks + answer + scores + stats.
        """
        if not self.chunks:
            raise ValueError("No chunks indexed. Call ingest() first.")
            
        stats = {}
        t_start = time.time()
        
        # 1. Retrieve
        t = time.time()
        query_embedding = self.encoder.encode(question)
        
        if self.retrieval_mode == "dense":
            retrieved = brute_force_dense(question, self.chunks, self.encoder, top_k, min_similarity)
        elif self.retrieval_mode == "faiss":
            retrieved = self.faiss_retriever.retrieve(query_embedding, top_k)
        elif self.retrieval_mode == "sparse":
            retrieved = self.sparse_retriever.retrieve(question, top_k)
        elif self.retrieval_mode == "hybrid":
            dense_res = self.faiss_retriever.retrieve(query_embedding, top_k=max(5, top_k))
            sparse_res = self.sparse_retriever.retrieve(question, top_k=max(5, top_k))
            retrieved = reciprocal_rank_fusion(dense_res, sparse_res, top_n=top_k)
        else:
            raise ValueError(f"Unknown mode: {self.retrieval_mode}")
            
        stats["retrieval_time"] = time.time() - t
        stats["num_retrieved"] = len(retrieved)
        
        # 2. Context Expansion
        if use_expansion and len(retrieved) > 0:
            expanded = expand_neighbors(retrieved, self.chunks)
        else:
            expanded = retrieved
        stats["num_expanded"] = len(expanded)
        
        # 3. Build context
        context = build_context(expanded)
        stats["context_length"] = len(context)
        
        # 4. Optional generation
        answer = None
        if use_llm:
            t = time.time()
            prompt = build_rag_prompt(question, context)
            answer = generate_answer(prompt, model_name=llm_model)
            stats["generation_time"] = time.time() - t
            
        stats["total_time"] = time.time() - t_start
        
        result = {
            "query": question,
            "retrieved": retrieved,
            "expanded": expanded,
            "context": context,
            "answer": answer,
            "stats": stats
        }
        
        if verbose:
            print(f"[Done] Processed query in {stats['total_time']:.4f}s using '{self.retrieval_mode}' search.")
            
        return result

    def run(self, document, query, top_k=3, min_similarity=0.2, use_expansion=True, use_llm=False, llm_model="phi3", chunk_kwargs=None, verbose=True):
        if chunk_kwargs is None:
            chunk_kwargs = {}
            
        stats = {}
        t_start = time.time()
        
        # 1. Chunk Document
        t = time.time()
        if self.chunking_strategy == "semantic":
            chunks = self.chunker.chunk(document, encoder=self.encoder, **chunk_kwargs)
        else:
            chunks = self.chunker.chunk(document, **chunk_kwargs)
        stats["chunking_time"] = time.time() - t
        stats["num_chunks"] = len(chunks)
        
        # 2. Embed Chunks
        t = time.time()
        chunks = self.encoder.generate_chunk_embeddings(chunks)
        stats["embedding_time"] = time.time() - t
        
        # Build Index states
        self.sparse_retriever.fit(chunks)
        self.faiss_retriever.add_chunks(chunks)
        
        # 3. Retrieve
        t = time.time()
        query_embedding = self.encoder.encode(query)
        
        if self.retrieval_mode == "dense":
            retrieved = brute_force_dense(query, chunks, self.encoder, top_k, min_similarity)
        elif self.retrieval_mode == "faiss":
            retrieved = self.faiss_retriever.retrieve(query_embedding, top_k)
        elif self.retrieval_mode == "sparse":
            retrieved = self.sparse_retriever.retrieve(query, top_k)
        elif self.retrieval_mode == "hybrid":
            dense_res = self.faiss_retriever.retrieve(query_embedding, top_k=5)
            sparse_res = self.sparse_retriever.retrieve(query, top_k=5)
            retrieved = reciprocal_rank_fusion(dense_res, sparse_res, top_n=top_k)
        else:
            raise ValueError(f"Unknown mode: {self.retrieval_mode}")
            
        stats["retrieval_time"] = time.time() - t
        stats["num_retrieved"] = len(retrieved)
        
        # 4. Context Expansion
        if use_expansion and len(retrieved) > 0:
            expanded = expand_neighbors(retrieved, chunks)
        else:
            expanded = retrieved
        stats["num_expanded"] = len(expanded)
        
        # 5. Build context
        context = build_context(expanded)
        stats["context_length"] = len(context)
        
        # 6. Optional generation
        prompt = build_rag_prompt(query, context)
        answer = None
        if use_llm:
            t = time.time()
            answer = generate_answer(prompt, model_name=llm_model)
            stats["generation_time"] = time.time() - t
            
        stats["total_time"] = time.time() - t_start
        
        result = {
            "query": query,
            "chunks": chunks,
            "retrieved": retrieved,
            "expanded": expanded,
            "context": context,
            "answer": answer,
            "stats": stats
        }
        
        if verbose:
            print(f"[Done] Processed pipeline in {stats['total_time']:.4f}s using '{self.retrieval_mode}' search.")
            
        return result
