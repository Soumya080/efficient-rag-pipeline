"""
Naive Fixed-Size Chunking

Splits text into chunks of fixed character length with overlap.

Original code from: 01_chunking.ipynb
"""

import uuid


def chunk(text, chunk_size=200, overlap=50):
    """
    Fixed-size character chunking with overlap.

    Splits text into chunks of `chunk_size` characters,
    sliding forward by (chunk_size - overlap) each step.

    Args:
        text: Input string to chunk
        chunk_size: Number of characters per chunk
        overlap: Number of overlapping characters between chunks

    Returns:
        List of chunk dicts with chunk_id, content, chunk_index, chunk_length
    """
    chunk_list = []

    if chunk_size > overlap:
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            content = text[start:end]

            chunk_data = {
                "chunk_id": str(uuid.uuid4()),
                "content": content,
                "chunk_index": len(chunk_list),
                "chunk_length": len(content),
                "start": start,
                "end": end
            }
            chunk_list.append(chunk_data)

            start = start + (chunk_size - overlap)

    return chunk_list


# -----------------------------------------------
# STANDALONE TEST
# -----------------------------------------------
if __name__ == "__main__":

    test_text = "Neural networks process information. Deep learning uses neural networks extensively. AI systems automate decision making. Technology changes modern industries."

    print("=" * 60)
    print("NAIVE CHUNKING TEST")
    print("=" * 60)

    results = chunk(test_text, chunk_size=60, overlap=15)

    for i, c in enumerate(results):
        print(f"\nChunk {i}:")
        print(f"  Length: {c['chunk_length']}")
        print(f"  Range:  [{c['start']}:{c['end']}]")
        print(f"  Text:   \"{c['content']}\"")

    print(f"\nTotal chunks: {len(results)}")
