"""
Sentence-Based Chunking

Splits text into sentences first, then groups sentences
into chunks up to a character limit with sentence-level overlap.

"""

import uuid
from src.chunking.utils import sentence_splitter


def _overlap_handler(current_chunk, overlap):
    """Return the last `overlap` sentences from current chunk for overlap."""
    if overlap == 0:
        return []
    return current_chunk[-overlap:]


def chunk(text, chunk_size=200, overlap=1):
    """
    Sentence-based chunking with sentence-level overlap.

    Splits text into sentences first, then groups sentences
    into chunks up to `chunk_size` characters. Overlap keeps
    the last N sentences from the previous chunk.

    Args:
        text: Input text to chunk
        chunk_size: Max character length per chunk
        overlap: Number of sentences to overlap between chunks

    Returns:
        List of chunk dicts with chunk_id, content, chunk_index,
        sentence_count, chunk_length
    """
    sentences = sentence_splitter(text)

    current_length = 0
    current_chunk = []
    chunk_list = []

    for sentence in sentences:
        sentence_length = len(sentence)

        # CASE 1: sentence fits in current chunk
        if (current_length + sentence_length + 1 <= chunk_size) or (current_length == 0):
            current_chunk.append(sentence)
            current_length += sentence_length + 1

        # CASE 2: chunk full — finalize and start new
        else:
            chunk_content = " ".join(current_chunk)
            chunk_data = {
                "chunk_id": str(uuid.uuid4()),
                "content": chunk_content,
                "chunk_index": len(chunk_list),
                "sentence_count": len(current_chunk),
                "chunk_length": len(chunk_content)
            }
            chunk_list.append(chunk_data)

            # overlap handling
            overlap_part = _overlap_handler(current_chunk, overlap)
            current_chunk = overlap_part.copy()
            current_chunk.append(sentence)

            current_length = 0
            for s in current_chunk:
                current_length += len(s) + 1

    # final remaining chunk
    if current_chunk:
        chunk_content = " ".join(current_chunk)
        chunk_data = {
            "chunk_id": str(uuid.uuid4()),
            "content": chunk_content,
            "chunk_index": len(chunk_list),
            "sentence_count": len(current_chunk),
            "chunk_length": len(chunk_content)
        }
        chunk_list.append(chunk_data)

    return chunk_list



# STANDALONE TEST
 
if __name__ == "__main__":

    test_text = "Neural networks process information. Deep learning uses neural networks extensively. AI systems automate decision making. Technology changes modern industries. Software tools improve business productivity. Companies use analytics platforms daily."

    print("=" * 60)
    print("SENTENCE CHUNKING TEST")
    print("=" * 60)

    results = chunk(test_text, chunk_size=100, overlap=1)

    for i, c in enumerate(results):
        print(f"\nChunk {i}:")
        print(f"  Sentences: {c['sentence_count']}")
        print(f"  Length:    {c['chunk_length']}")
        print(f"  Text:      \"{c['content']}\"")

    print(f"\nTotal chunks: {len(results)}")
