# fix_mixed_document_format.py
import json
import numpy as np
from datetime import datetime
from pathlib import Path

print("🔧 Fixing mixed document format...")

# Load current data
with open('embeddings/documents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

docs = data['documents']
meta = data.get('metadata', [])

print(f"📊 Total documents: {len(docs)}")

# Count formats
dict_count = sum(1 for d in docs if isinstance(d, dict))
str_count = sum(1 for d in docs if isinstance(d, str))

print(f"   Dict format: {dict_count}")
print(f"   String format: {str_count}")

# Convert all to dict format
fixed_docs = []
for i, doc in enumerate(docs):
    if isinstance(doc, dict):
        # Already correct format
        fixed_docs.append(doc)
    else:
        # Convert string to dict format
        # Get metadata if available
        doc_meta = meta[i] if i < len(meta) else {}
        
        fixed_doc = {
            'content': doc,
            'metadata': doc_meta if isinstance(doc_meta, dict) else {
                'source': str(doc_meta) if doc_meta else 'Unknown',
                'document_type': 'Landuse Plan',
                'chunk_id': i
            },
            'source': doc_meta.get('source', 'Landuse Plans') if isinstance(doc_meta, dict) else 'Landuse Plans',
            'citation': f"OCR Document {i}",
            'document_id': f"ocr_doc_{i}"
        }
        fixed_docs.append(fixed_doc)

print(f"\n✅ Fixed {str_count} string documents to dict format")

# Backup
backup_path = Path('embeddings/_backup') / f'documents_mixed_format_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
import shutil
shutil.copy('embeddings/documents.json', backup_path)
print(f"💾 Backup: {backup_path}")

# Save fixed version
fixed_data = {
    'documents': fixed_docs,
    'metadata': meta,
    'total_documents': len(fixed_docs)
}

with open('embeddings/documents.json', 'w', encoding='utf-8') as f:
    json.dump(fixed_data, f, ensure_ascii=False, indent=2)

print(f"✅ Saved: {len(fixed_docs)} documents (all dict format)")

# Verify
print("\n🔍 Verifying fix...")
with open('embeddings/documents.json', 'r', encoding='utf-8') as f:
    verify_data = json.load(f)

verify_docs = verify_data['documents']
all_dicts = all(isinstance(d, dict) for d in verify_docs)

if all_dicts:
    print("✅ All documents are now dict format!")
    
    # Check if OCR content is there
    has_stgt = any('stgt' in str(d.get('content', '')).lower() or '272' in str(d.get('content', '')) 
                   for d in verify_docs[-200:])
    
    if has_stgt:
        print("✅ Stgt 272 content found in documents!")
    else:
        print("⚠️  Stgt 272 content not found in last 200 docs")
else:
    print("❌ Still have mixed formats!")

print("\nTest now: python test_direct_rag.py")