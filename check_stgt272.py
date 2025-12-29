import json
from pathlib import Path

EMBEDDINGS_DIR = Path("embeddings")

with open(EMBEDDINGS_DIR / "documents.json", 'r', encoding='utf-8') as f:
    documents = json.load(f)

print("="*80)
print("🔍 SEARCHING FOR STGT 272 IN EMBEDDINGS")
print("="*80)

stgt272_docs = [
    doc for doc in documents 
    if "272" in doc.get('source', '') or 
       "272" in doc.get('metadata', {}).get('source_file', '')
]

print(f"\n📚 Found {len(stgt272_docs)} documents containing '272'")

if stgt272_docs:
    print("\n📄 Sample documents:")
    for i, doc in enumerate(stgt272_docs[:5], 1):
        source = doc.get('source', 'Unknown')
        content_preview = doc.get('content', '')[:200]
        print(f"\n{i}. Source: {source}")
        print(f"   Content: {content_preview}...")
else:
    print("\n❌ NO DOCUMENTS FOUND WITH 'stgt-272' or '272'!")
    print("\n💡 This means:")
    print("   1. The OCR for stgt-272-bebauungsplan-2024-02.pdf failed")
    print("   2. Or it wasn't included in the merge")
    print("   3. Or the filename changed during processing")

# Check what Nordbahnhof docs we have
print("\n" + "="*80)
print("🔍 SEARCHING FOR NORDBAHNHOF DOCUMENTS")
print("="*80)

nordbahnhof_docs = [
    doc for doc in documents 
    if "nordbahnhof" in doc.get('source', '').lower() or 
       "nordbahnhof" in doc.get('content', '').lower()[:500]
]

print(f"\n📚 Found {len(nordbahnhof_docs)} documents mentioning 'Nordbahnhof'")

if nordbahnhof_docs:
    # Group by source
    from collections import Counter
    sources = Counter(doc.get('source', 'Unknown') for doc in nordbahnhof_docs)
    print("\n📋 Documents by source:")
    for source, count in sources.most_common(10):
        print(f"   {source}: {count} chunks")