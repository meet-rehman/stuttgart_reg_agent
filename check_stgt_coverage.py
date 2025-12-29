#!/usr/bin/env python3
"""
Check what Stgt/Bebauungsplan documents are in your embeddings
"""

from precomputed_rag import EnhancedPrecomputedRAGSystem

print("="*70)
print("🔍 CHECKING STGT/BEBAUUNGSPLAN COVERAGE")
print("="*70)
print()

rag = EnhancedPrecomputedRAGSystem()
stats = rag.get_stats()

print(f"📊 Total documents: {stats['total_documents']}")
print()

# Search for Stgt documents
print("="*70)
print("🔍 Searching for Stgt documents...")
print("="*70)
print()

search_terms = [
    "Stgt 272",
    "Stgt 283",
    "Nordbahnhof",
    "Friedhofstrasse",
    "Killesberg",
    "Maybachstraße",
    "Bebauungsplan",
    "GRZ Grundflächenzahl",
]

for term in search_terms:
    print(f"\n🔍 Searching for: '{term}'")
    print("-"*70)
    
    results = rag.search(term, top_k=3)
    
    if not results:
        print("   ❌ No results found")
        continue
    
    best_score = results[0].score
    
    if best_score > 0.7:
        print(f"   ✅ Found! (best score: {best_score:.3f})")
    elif best_score > 0.5:
        print(f"   ⚠️  Weak match (best score: {best_score:.3f})")
    else:
        print(f"   ❌ Poor match (best score: {best_score:.3f})")
    
    for i, result in enumerate(results[:3], 1):
        source_short = result.source[:60] + "..." if len(result.source) > 60 else result.source
        print(f"      [{i}] {source_short} (score: {result.score:.3f})")

print("\n" + "="*70)
print("📊 ANALYSIS")
print("="*70)
print()

# Check for specific documents
has_stgt_272 = any(r.score > 0.7 for r in rag.search("Stgt 272", top_k=5))
has_stgt_283 = any(r.score > 0.7 for r in rag.search("Stgt 283", top_k=5))
has_nordbahnhof = any(r.score > 0.7 for r in rag.search("Nordbahnhof", top_k=5))

print(f"Stgt 272 documents:     {'✅ Yes' if has_stgt_272 else '❌ No'}")
print(f"Stgt 283 documents:     {'✅ Yes' if has_stgt_283 else '❌ No'}")
print(f"Nordbahnhof area docs:  {'✅ Yes' if has_nordbahnhof else '❌ No'}")

print()
print("="*70)
print("💡 RECOMMENDATION")
print("="*70)
print()

if not (has_stgt_272 or has_stgt_283):
    print("❌ CRITICAL: Your embeddings do NOT contain Stgt 272 or Stgt 283 documents!")
    print()
    print("You need to:")
    print("   1. Check if you have these PDFs in data/raw/Landuse Plans/:")
    print("      - Stgt_272_Nordbahnhof.pdf")
    print("      - Stgt_283_Killesberg.pdf")
    print()
    print("   2. Run OCR on these PDFs:")
    print("      python batch_ocr.py")
    print()
    print("   3. Build embeddings from OCR output:")
    print("      python build_embeddings_from_ocr.py")
    print()
    print("   4. Re-test:")
    print("      python test_rag_local.py")
else:
    print("✅ Good! You have some relevant documents.")
    print("   The low scores might be due to:")
    print("   - Query wording differences")
    print("   - OCR quality issues")
    print("   - Need for more specific documents")

print()