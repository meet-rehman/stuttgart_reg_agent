import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
OCR_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"

def load_existing_embeddings():
    """Load current embeddings"""
    
    doc_file = EMBEDDINGS_DIR / "documents.json"
    emb_file = EMBEDDINGS_DIR / "embeddings.npy"
    
    if not doc_file.exists() or not emb_file.exists():
        print("⚠️  No existing embeddings found")
        return [], None
    
    print("📚 Loading existing embeddings...")
    
    with open(doc_file, 'r', encoding='utf-8') as f:
        existing_docs = json.load(f)
    
    existing_embeddings = np.load(emb_file)
    
    print(f"✅ Loaded {len(existing_docs):,} existing documents")
    
    return existing_docs, existing_embeddings

def load_ocr_documents(ocr_dir):
    """Load new OCR results"""
    
    documents = []
    ocr_files = list(ocr_dir.rglob("*_ocr.json"))
    ocr_files = [f for f in ocr_files if f.name != "_ocr_summary.json"]
    
    print(f"📚 Found {len(ocr_files)} new OCR files")
    
    for ocr_file in ocr_files:
        try:
            with open(ocr_file, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading {ocr_file.name}: {e}")
            continue
        
        source_file = ocr_data.get('source_file', ocr_file.stem)
        
        rel_path = ocr_file.relative_to(ocr_dir)
        parts = rel_path.parts
        district = parts[0] if len(parts) > 0 else "unknown"
        subarea = parts[1] if len(parts) > 1 else None
        
        # Determine document type
        doc_type = "Building Regulation"
        if "bebauungsplan" in source_file.lower():
            doc_type = "Bebauungsplan"
        elif "landuse" in source_file.lower():
            doc_type = "Land Use Plan"
        elif "statutes" in source_file.lower():
            doc_type = "Local Statute"
        
        for page in ocr_data.get('pages', []):
            text = page.get('text', '').strip()
            
            if text and len(text) > 50:
                doc = {
                    'content': text,
                    'metadata': {
                        'source_file': source_file,
                        'district': district,
                        'subarea': subarea,
                        'document_type': doc_type,
                        'page': page['page'],
                        'char_count': len(text)
                    },
                    'source': f"{source_file}, Page {page['page']}",
                    'citation': f"{doc_type}: {source_file}, Page {page['page']} ({district})",
                    'document_id': f"{source_file}_p{page['page']}"
                }
                documents.append(doc)
    
    return documents

def merge_and_save(existing_docs, existing_embeddings, new_docs, embeddings_dir):
    """Merge existing and new documents, create embeddings for new ones"""
    
    print(f"\n🔄 Merging documents...")
    
    # Check for duplicates by document_id
    existing_ids = {doc.get('document_id') for doc in existing_docs if doc.get('document_id')}
    new_docs_filtered = [doc for doc in new_docs if doc.get('document_id') not in existing_ids]
    
    duplicates = len(new_docs) - len(new_docs_filtered)
    if duplicates > 0:
        print(f"⚠️  Skipped {duplicates} duplicate documents")
    
    if not new_docs_filtered:
        print("✅ No new documents to add!")
        return
    
    print(f"📝 Adding {len(new_docs_filtered):,} new documents")
    
    # Create embeddings for new documents only
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
    merged_docs = existing_docs + new_docs_filtered
    merged_embeddings = np.vstack([existing_embeddings, new_embeddings])
    
    # Backup old embeddings
    backup_dir = embeddings_dir / "_backup"
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n💾 Creating backup...")
    import shutil
    shutil.copy2(
        embeddings_dir / "documents.json",
        backup_dir / f"documents_{timestamp}.json"
    )
    shutil.copy2(
        embeddings_dir / "embeddings.npy",
        backup_dir / f"embeddings_{timestamp}.npy"
    )
    
    # Save merged
    print(f"💾 Saving merged embeddings...")
    
    doc_file = embeddings_dir / "documents.json"
    with open(doc_file, 'w', encoding='utf-8') as f:
        json.dump(merged_docs, f, ensure_ascii=False, indent=2)
    
    emb_file = embeddings_dir / "embeddings.npy"
    np.save(emb_file, merged_embeddings)
    
    # Update metadata
    metadata = {
        'model_name': 'all-MiniLM-L6-v2',
        'embedding_dim': int(merged_embeddings.shape[1]),
        'document_count': len(merged_docs),
        'embedding_count': int(merged_embeddings.shape[0]),
        'created_at': datetime.now().isoformat(),
        'last_merge': timestamp
    }
    
    model_file = embeddings_dir / "model_info.json"
    with open(model_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Statistics
    from collections import Counter
    
    print(f"\n{'='*60}")
    print(f"✅ EMBEDDINGS MERGED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"📊 Previous documents: {len(existing_docs):,}")
    print(f"📊 New documents: {len(new_docs_filtered):,}")
    print(f"📊 Total documents: {len(merged_docs):,}")
    print(f"📊 Embeddings shape: {merged_embeddings.shape}")
    
    # By document type
    by_type = Counter(doc['metadata'].get('document_type', 'Unknown') for doc in merged_docs)
    print(f"\n📋 Documents by type:")
    for doc_type, count in by_type.most_common():
        print(f"   {doc_type}: {count:,} chunks")
    
    # By district
    by_district = Counter(doc['metadata'].get('district', 'Unknown') for doc in merged_docs)
    print(f"\n📂 Documents by district:")
    for district, count in by_district.most_common():
        print(f"   {district}: {count:,} chunks")
    
    print(f"\n💾 Backup saved to: {backup_dir}")
    print(f"{'='*60}")

def main():
    print("\n" + "="*60)
    print("🔄 MERGE NEW OCR WITH EXISTING EMBEDDINGS")
    print("="*60)
    
    # Check OCR output
    if not OCR_OUTPUT_DIR.exists():
        print(f"\n❌ OCR output not found: {OCR_OUTPUT_DIR}")
        print("💡 Run batch_ocr.py first!")
        return
    
    # Load existing
    existing_docs, existing_embeddings = load_existing_embeddings()
    
    # Load new OCR
    print(f"\n📁 Loading new OCR output from: {OCR_OUTPUT_DIR}")
    new_docs = load_ocr_documents(OCR_OUTPUT_DIR)
    
    if not new_docs:
        print("\n❌ No new documents found!")
        return
    
    print(f"✅ Loaded {len(new_docs):,} new document chunks")
    
    # Merge
    if existing_docs:
        merge_and_save(existing_docs, existing_embeddings, new_docs, EMBEDDINGS_DIR)
    else:
        print("\n⚠️  No existing embeddings, creating new ones...")
        # Just create new (same as build_embeddings_from_ocr.py)
    
    print("\n✨ Process complete!")

if __name__ == "__main__":
    main()