import os
import argparse
import time
from src.pipeline import RAGPipeline

def main():
    parser = argparse.ArgumentParser(description="Interactive RAG Pipeline")
    parser.add_argument("--data", type=str, default="data/knowledge_base/", help="Path to knowledge base directory")
    parser.add_argument("--chunking", type=str, default="semantic", help="Chunking strategy")
    parser.add_argument("--mode", type=str, default="hybrid", help="Retrieval mode")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Embedding model")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM answer generation")
    args = parser.parse_args()

    print(f"\n[RAG Pipeline] Loading documents from {args.data}...")
    
    pipeline = RAGPipeline(
        chunking_strategy=args.chunking,
        retrieval_mode=args.mode,
        model_name=args.model
    )
    
    # Ingest documents
    try:
        stats = pipeline.ingest(args.data, verbose=False)
        print(f"  → Found {stats['num_docs']} documents")
        print(f"  → Chunking strategy: {args.chunking}")
        print(f"  → Generated {stats['num_chunks']} chunks")
        print(f"\n[RAG Pipeline] Building indexes...")
        print(f"  → FAISS index: {stats['num_chunks']} vectors ({pipeline.encoder.dimension} dimensions)")
        print(f"  → BM25 corpus: {stats['num_chunks']} documents fitted")
        print(f"  → Total ingestion time: {stats['time']:.2f}s")
    except Exception as e:
        print(f"Error during ingestion: {e}")
        return

    print("\n" + "═"*51)
    print("  RAG Pipeline Ready — Type your questions below")
    print("  Type 'quit' to exit | 'stats' for pipeline info")
    print("═"*51 + "\n")

    while True:
        try:
            query = input("You: ")
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if query.lower() == 'stats':
                print(f"\nPipeline Stats:")
                print(f"  Documents Indexed: {stats['num_docs']}")
                print(f"  Total Chunks: {stats['num_chunks']}")
                print(f"  Retrieval Mode: {args.mode}")
                print(f"  Embedding Model: {args.model}\n")
                continue
            if not query.strip():
                continue
                
            result = pipeline.query(
                question=query, 
                top_k=3, 
                use_llm=not args.no_llm,
                verbose=False
            )
            
            print(f"\n───── Retrieved Chunks ({len(result['retrieved'])}) " + "─"*25)
            for i, chunk in enumerate(result['retrieved']):
                score_str = f"Score: {chunk.get('score', 0):.4f} | " if 'score' in chunk else ""
                text = chunk.get('content', '')
                if len(text) > 150:
                    text = text[:147] + "..."
                print(f"  [{i+1}] {score_str}Chunk #{chunk.get('chunk_id', 'N/A')}")
                print(f"      \"{text}\"\n")
                
            if result.get('answer'):
                print("───── Answer " + "─"*39)
                print(result['answer'])
                print()
                
            s = result['stats']
            tokens = len(result['context']) // 4  # rough estimate
            print("───── Stats " + "─"*40)
            print(f"  Retrieval time: {s.get('retrieval_time', 0):.4f}s | Mode: {args.mode} | Tokens: ~{tokens}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    main()
