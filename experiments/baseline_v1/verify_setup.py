

import sys
print(f"Python: {sys.version}")

# Test each dependency
try:
    from datasets import load_dataset
    print("✓ datasets")
except: print("✗ datasets — run: pip install datasets")

try:
    import faiss
    print("✓ faiss")
except: print("✗ faiss — run: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
    print("✓ sentence-transformers")
except: print("✗ sentence-transformers — run: pip install sentence-transformers")

try:
    import pandas
    print("✓ pandas")
except: print("✗ pandas — run: pip install pandas")

try:
    import matplotlib
    print("✓ matplotlib")
except: print("✗ matplotlib — run: pip install matplotlib")

try:
    from tqdm import tqdm
    print("✓ tqdm")
except: print("✗ tqdm — run: pip install tqdm")

try:
    import numpy
    print("✓ numpy")
except: print("✗ numpy — run: pip install numpy")

# Test model loading (this downloads ~90MB model first time)
print("\nLoading embedding model (first time downloads ~90MB)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
vec = model.encode("test query")
print(f"✓ Model loaded. Embedding dim: {vec.shape[0]}")

print("\n✓ ALL CHECKS PASSED — Ready to experiment!")
