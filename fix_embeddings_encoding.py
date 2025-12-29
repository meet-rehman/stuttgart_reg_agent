# fix_embeddings_encoding.py
import json
import numpy as np
from datetime import datetime
from pathlib import Path

print("🔧 Fixing embeddings encoding...")

# Read with correct encoding
try:
    with open('embeddings/documents.json', 'r', encoding='utf-8', errors='ignore') as f:
        data = json.load(f)
    print(f"✅ Loaded with error handling")
except:
    # Try latin-1 if utf-8 fails
    with open('embeddings/documents.json', 'r', encoding='latin-1') as f:
        data = json.load(f)
    print(f"✅ Loaded with latin-1")

# Check format
if isinstance(data, dict):
    docs = data['documents']
    meta = data.get('metadata', [])
else:
    docs = data
    meta = []

print(f"📊 Documents: {len(docs)}")

# Clean and re-save with proper encoding
cleaned_docs = []
for doc in docs:
    if isinstance(doc, str):
        # Remove problematic characters
        clean = doc.encode('utf-8', errors='ignore').decode('utf-8')
        cleaned_docs.append(clean)
    else:
        cleaned_docs.append(doc)

# Backup
backup_path = Path('embeddings/_backup') / f'documents_broken_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
backup_path.parent.mkdir(exist_ok=True)
import shutil
shutil.copy('embeddings/documents.json', backup_path)
print(f"💾 Backup: {backup_path}")

# Save cleaned version
cleaned_data = {
    'documents': cleaned_docs,
    'metadata': meta,
    'total_documents': len(cleaned_docs)
}

with open('embeddings/documents.json', 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, ensure_ascii=False)

print(f"✅ Saved cleaned version: {len(cleaned_docs)} documents")
print("\nNow test: python test_direct_rag.py")