# debug_rag_loading.py
import json
import numpy as np
from pathlib import Path

print("🔍 Debugging RAG loading...")

# Check files
docs_file = Path("embeddings/documents.json")
emb_file = Path("embeddings/embeddings.npy")

print(f"\n1️⃣ Files exist:")
print(f"   documents.json: {docs_file.exists()} ({docs_file.stat().st_size/1024/1024:.1f}MB)")
print(f"   embeddings.npy: {emb_file.exists()} ({emb_file.stat().st_size/1024/1024:.1f}MB)")

# Try loading
print(f"\n2️⃣ Loading documents.json...")
try:
    with open(docs_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        docs = data.get('documents', [])
        print(f"   ✅ Format: dict")
        print(f"   ✅ Keys: {list(data.keys())}")
    else:
        docs = data
        print(f"   ✅ Format: list")
    
    print(f"   ✅ Documents loaded: {len(docs)}")
    print(f"   ✅ Sample doc: {docs[0][:100] if docs else 'None'}...")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    docs = []

# Try loading embeddings
print(f"\n3️⃣ Loading embeddings.npy...")
try:
    embeddings = np.load(emb_file)
    print(f"   ✅ Shape: {embeddings.shape}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Check what precomputed_rag.py expects
print(f"\n4️⃣ Checking precomputed_rag.py logic...")
print("   Looking at how it loads...")

from precomputed_rag import EnhancedPrecomputedRAGSystem

# Try to see what's happening
try:
    rag = EnhancedPrecomputedRAGSystem()
    print(f"   Documents loaded: {len(rag.documents) if hasattr(rag, 'documents') else 0}")
    print(f"   Has search method: {hasattr(rag, 'search')}")
except Exception as e:
    print(f"   ❌ Init error: {e}")
    import traceback
    traceback.print_exc()