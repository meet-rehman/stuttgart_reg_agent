from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import shutil

PROJECT_ROOT = Path(__file__).parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "Stuttgart_Nord" / "stgt_272_bebauungsplan"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"

print("="*80)
print("📚 ADD STGT 272 TO EMBEDDINGS")
print("="*80)

# Check if OCR output exists
ocr_file = PROCESSED_DIR / "stgt_272_ocr.json"

if not ocr_file.exists():
    print(f"❌ OCR output not found: {ocr_file}")
    print("💡 Run: python ocr_stgt272_only.py first!")
    exit(1)

# Load OCR data
print(f"\n📄 Loading OCR output...")
with open(ocr_file, 'r', encoding='utf-8') as f:
    ocr_data = json.load(f)

print(f"   Source: {ocr_data['source_file']}")
print(f"   Pages: {ocr_data['total_pages']}")

# Create documents with proper naming
new_docs = []
for page in ocr_data['pages']:
    text = page.get('text', '').strip()
    
    if text and len(text) > 50:  # Only substantial content
        doc = {
            'content': text,
            'metadata': {
                'source_file': 'stgt-272-bebauungsplan-2024-02.pdf',
                'district': 'Stuttgart Nord',
                'subarea': 'Nordbahnhof-Friedhofstrasse',
                'document_type': 'Bebauungsplan',
                'bebauungsplan_number': 'Stgt 272',  # Important!
                'page': page['page'],
                'char_count': page['char_count']
            },
            'source': f"stgt-272-bebauungsplan-2024-02.pdf, Page {page['page']}",
            'citation': f"Bebauungsplan Stgt 272: Nordbahnhof-Friedhofstrasse, Page {page['page']}",
            'document_id': f"stgt_272_bebauungsplan_p{page['page']}"
        }
        new_docs.append(doc)

print(f"\n✅ Created {len(new_docs)} document chunks")

if not new_docs:
    print("❌ No valid documents created! Check OCR quality.")
    exit(1)

# Load existing embeddings
print(f"\n💾 Loading existing embeddings...")
with open(EMBEDDINGS_DIR / "documents.json", 'r', encoding='utf-8') as f:
    existing_docs = json.load(f)

existing_embeddings = np.load(EMBEDDINGS_DIR / "embeddings.npy")

print(f"   Current documents: {len(existing_docs):,}")

# Check for duplicates
existing_ids = {doc.get('document_id') for doc in existing_docs}
new_docs_filtered = [doc for doc in new_docs if doc['document_id'] not in existing_ids]

if not new_docs_filtered:
    print("\n⚠️ All documents already in embeddings!")
    exit(0)

print(f"   New documents to add: {len(new_docs_filtered)}")

# Backup
print(f"\n💾 Creating backup...")
backup_dir = EMBEDDINGS_DIR / "_backup"
backup_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

shutil.copy2(EMBEDDINGS_DIR / "documents.json", backup_dir / f"documents_{timestamp}.json")
shutil.copy2(EMBEDDINGS_DIR / "embeddings.npy", backup_dir / f"embeddings_{timestamp}.npy")
print(f"   Backup saved: {timestamp}")

# Create embeddings
print(f"\n🤖 Creating embeddings for new documents...")
model = SentenceTransformer('all-MiniLM-L6-v2')

new_texts = [doc['content'] for doc in new_docs_filtered]
new_embeddings = model.encode(
    new_texts,
    show_progress_bar=True,
    convert_to_numpy=True,
    batch_size=32
)

# Merge
print(f"\n🔄 Merging with existing embeddings...")
merged_docs = existing_docs + new_docs_filtered
merged_embeddings = np.vstack([existing_embeddings, new_embeddings])

# Save
print(f"\n💾 Saving merged embeddings...")
with open(EMBEDDINGS_DIR / "documents.json", 'w', encoding='utf-8') as f:
    json.dump(merged_docs, f, ensure_ascii=False, indent=2)

np.save(EMBEDDINGS_DIR / "embeddings.npy", merged_embeddings)

# Update metadata
metadata = {
    'model_name': 'all-MiniLM-L6-v2',
    'embedding_dim': int(merged_embeddings.shape[1]),
    'document_count': len(merged_docs),
    'embedding_count': int(merged_embeddings.shape[0]),
    'created_at': datetime.now().isoformat(),
    'last_update': 'Added Stgt 272 Bebauungsplan with correct naming'
}

with open(EMBEDDINGS_DIR / "model_info.json", 'w') as f:
    json.dump(metadata, f, indent=2)

# Summary
print(f"\n{'='*80}")
print(f"✅ SUCCESS! STGT 272 ADDED TO EMBEDDINGS")
print(f"{'='*80}")
print(f"📊 Previous: {len(existing_docs):,} documents")
print(f"📊 Added: {len(new_docs_filtered)} Stgt 272 pages")
print(f"📊 Total: {len(merged_docs):,} documents")
print(f"📁 Backup: {backup_dir / f'documents_{timestamp}.json'}")
print(f"{'='*80}")
print(f"\n✨ You can now query:")
print(f"   • 'What is the GRZ for Stgt 272?'")
print(f"   • 'Stgt 272 height restrictions'")
print(f"   • 'Bebauungsplan 272 Nordbahnhof'")