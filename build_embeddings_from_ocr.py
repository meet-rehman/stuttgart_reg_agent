#!/usr/bin/env python3
"""
Add OCR outputs to existing embeddings
This ADDS to your 12,106 documents, doesn't replace them
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from datetime import datetime

print("="*70)
print("📊 ADD OCR TO EMBEDDINGS")
print("="*70)
print()

# Paths
ocr_dir = Path("data/processed/ocr_outputs/Landuse Plans")
embeddings_dir = Path("embeddings")
documents_file = embeddings_dir / "documents.json"
embeddings_file = embeddings_dir / "embeddings.npy"

# Check files exist
if not documents_file.exists():
    print("❌ embeddings/documents.json not found!")
    print("   Run build_embeddings_locally.py first")
    import sys
    sys.exit(1)

if not ocr_dir.exists():
    print("❌ OCR outputs not found!")
    print("   Run quick_ocr_stgt.py first")
    import sys
    sys.exit(1)

# Load existing embeddings
print("📂 Loading existing embeddings...")
with open(documents_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Handle both formats: dict or list
if isinstance(data, dict):
    existing_docs = data.get('documents', [])
    existing_meta = data.get('metadata', [])
elif isinstance(data, list):
    existing_docs = data
    existing_meta = [{'source': 'Unknown', 'chunk_id': i} for i in range(len(data))]
else:
    print("❌ Unknown documents.json format!")
    import sys
    sys.exit(1)

existing_embeddings = np.load(embeddings_file)

print(f"✅ Loaded {len(existing_docs)} existing documents")
print()

# Find OCR files
ocr_files = list(ocr_dir.glob("*_ocr.txt"))
print(f"📄 Found {len(ocr_files)} OCR files")

if not ocr_files:
    print("❌ No OCR files found!")
    import sys
    sys.exit(1)

for f in ocr_files[:5]:
    print(f"   - {f.name}")
if len(ocr_files) > 5:
    print(f"   ... and {len(ocr_files) - 5} more")
print()

# Process OCR files
new_docs = []
new_meta = []

for ocr_file in ocr_files:
    print(f"📖 Reading: {ocr_file.name}")
    
    try:
        with open(ocr_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into chunks (500 chars each)
        chunks = []
        for i in range(0, len(text), 500):
            chunk = text[i:i+500].strip()
            if len(chunk) > 50:  # Skip tiny chunks
                chunks.append(chunk)
        
        # Add to new docs
        for i, chunk in enumerate(chunks):
            new_docs.append(chunk)
            new_meta.append({
                'source': f"Landuse Plans/{ocr_file.stem.replace('_ocr', '')}",
                'chunk_id': i,
                'length': len(chunk),
                'document_type': 'Landuse Plan',
                'ocr_date': datetime.now().isoformat()
            })
        
        print(f"   ✅ Added {len(chunks)} chunks")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

print()
print(f"📊 New documents: {len(new_docs)}")

if not new_docs:
    print("❌ No new documents to add!")
    import sys
    sys.exit(1)

# Generate embeddings for new docs
print()
print("🤖 Generating embeddings for new documents...")
model = SentenceTransformer('all-MiniLM-L6-v2')
new_embeddings = model.encode(new_docs, show_progress_bar=True)
print(f"✅ Generated {new_embeddings.shape[0]} embeddings")

# Combine with existing
print()
print("🔗 Combining with existing embeddings...")
combined_docs = existing_docs + new_docs
combined_meta = existing_meta + new_meta
combined_embeddings = np.vstack([existing_embeddings, new_embeddings])

print(f"   Previous: {len(existing_docs)} documents")
print(f"   New:      {len(new_docs)} documents")
print(f"   Total:    {len(combined_docs)} documents")

# Backup old files
print()
print("💾 Creating backup...")
backup_dir = embeddings_dir / "_backup"
backup_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
import shutil
shutil.copy(documents_file, backup_dir / f"documents_{timestamp}.json")
shutil.copy(embeddings_file, backup_dir / f"embeddings_{timestamp}.npy")
print(f"✅ Backup saved to: {backup_dir}")

# Save combined
print()
print("💾 Saving combined embeddings...")

combined_data = {
    'documents': combined_docs,
    'metadata': combined_meta,
    'total_documents': len(combined_docs)
}

with open(documents_file, 'w', encoding='utf-8') as f:
    json.dump(combined_data, f, ensure_ascii=False)

np.save(embeddings_file, combined_embeddings)

print(f"✅ Saved to {embeddings_dir}")
print()

# Summary
print("="*70)
print("✅ SUCCESS!")
print("="*70)
print(f"📊 Previous: {len(existing_docs)} documents")
print(f"📊 Added: {len(new_docs)} documents")
print(f"📊 Total: {len(combined_docs)} documents")
print(f"📁 Backup: {backup_dir / f'documents_{timestamp}.json'}")
print()
print("Next: python test_rag_local.py")
print("="*70)