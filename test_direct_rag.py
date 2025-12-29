# test_direct_rag.py
from precomputed_rag import EnhancedPrecomputedRAGSystem

rag = EnhancedPrecomputedRAGSystem()

queries = [
    "HbA 280",
    "EFH 261",
    "Stgt 272 Höhe",
    "Nordbahnhof Gebäudehöhe",
]

for query in queries:
    print(f"\nQuery: {query}")
    results = rag.search(query, top_k=3)
    for r in results:
        print(f"  Score: {r.score:.3f} | {r.content[:100]}")