# test_check_embeddings.py
from precomputed_rag import EnhancedPrecomputedRAGSystem

print("✅ RAG system imported successfully!")

# Initialize RAG
rag = EnhancedPrecomputedRAGSystem()

# Search for specific values
queries = [
    "GRZ 0.4",
    "HBA 283",
    "Gebäudehöhe 283.25",
    "Grundflächenzahl",
    "Stgt 272 Nordbahnhof",
    "EFH 261",
    "Höhe baulicher Anlagen"
]

print("\n" + "="*70)
print("🔍 CHECKING IF SPECIFIC VALUES ARE IN EMBEDDINGS")
print("="*70)

for q in queries:
    print(f"\n📌 Query: '{q}'")
    print("-" * 50)
    
    try:
        results = rag.search(q, top_k=3)
        
        if not results:
            print("   ❌ No results found")
            continue
        
        for i, r in enumerate(results, 1):
            score = r.score if hasattr(r, 'score') else 0
            content = r.content if hasattr(r, 'content') else str(r)
            source = r.source if hasattr(r, 'source') else 'Unknown'
            
            print(f"   Result {i}:")
            print(f"   Score: {score:.3f}")
            print(f"   Source: {source}")
            print(f"   Content: {content[:250]}...")
            print()
    
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("="*70)
print("\n💡 If scores are low (<0.5) or content doesn't have exact values,")
print("   your OCR didn't capture the tables/numeric labels properly.")