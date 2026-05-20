"""
Shared utilities used across all chunking strategies.

Contains:
  - sentence_splitter()   — hand-written sentence boundary detector
  - cosine_similarity()   — hand-written cosine similarity

Both from: 01_RETRIVER.ipynb, 03_SEMANTIC_CHUNKING_OWN.ipynb
"""

import numpy as np


def sentence_splitter(text):
    """
    Split text into sentences by detecting '.', '?', '!' boundaries.

    Hand-written splitter — no regex, no nltk.

    Args:
        text: Raw input text

    Returns:
        List of sentence strings
    """
    start = 0
    sentences = []

    i = 0
    while i < len(text):
        if text[i] in [".", "?", "!"]:
            sentence = text[start:i+1].strip()
            if len(sentence) > 0:
                sentences.append(sentence)
            start = i + 1
        i += 1

    # remaining text
    if start < len(text):
        remaining = text[start:].strip()
        if len(remaining) > 0:
            sentences.append(remaining)

    return sentences


def cosine_similarity(vec1, vec2):
    """
    Hand-written cosine similarity. No sklearn, no scipy.

    cos(A, B) = (A · B) / (||A|| * ||B||)

    Args:
        vec1: First embedding vector (numpy array)
        vec2: Second embedding vector (numpy array)

    Returns:
        Cosine similarity score (float)
    """
    dot_product = 0
    mag_A = 0
    mag_B = 0

    for i in range(len(vec1)):
        dot_product += vec1[i] * vec2[i]
        mag_A += vec1[i] ** 2
        mag_B += vec2[i] ** 2

    mag_A = np.sqrt(mag_A)
    mag_B = np.sqrt(mag_B)

    similarity = dot_product / (mag_A * mag_B)
    return similarity
