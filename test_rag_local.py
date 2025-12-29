#!/usr/bin/env python3
"""
Test RAG system with Nordbahnhof/Friedhofstrasse and Killesberg queries
Based on your specific document coverage
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

from precomputed_rag import EnhancedPrecomputedRAGSystem

print("="*70)
print("🧪 TESTING RAG SYSTEM - NORDBAHNHOF & FRIEDHOFSTRASSE")
print("="*70)
print()

print("🔧 Initializing RAG system...")
try:
    rag = EnhancedPrecomputedRAGSystem()
    
    # Try different method names to get document count
    if hasattr(rag, 'get_stats'):
        stats = rag.get_stats()
        doc_count = stats.get('total_documents', 'unknown')
    elif hasattr(rag, 'documents') and rag.documents:
        doc_count = len(rag.documents)
    elif hasattr(rag, 'embeddings') and rag.embeddings:
        doc_count = len(rag.embeddings)
    else:
        doc_count = "unknown (but system loaded)"
    
    print(f"✅ RAG system loaded with {doc_count} documents")
except Exception as e:
    print(f"❌ Failed to initialize RAG: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test queries organized by category
test_categories = {
    "Stgt 272 - Nordbahnhof/Friedhofstrasse": [
        "What is the GRZ for Stgt 272 Nordbahnhof?",
        "Grundflächenzahl WA 1 Stgt 272",
        "Height restrictions in Nordbahnhofstrasse Friedhofstrasse area",
        "HöA Höhe baulicher Anlagen Stgt 272",
        "What uses are permitted in WA zones Stgt 272?",
        "Was ist im WA 1 zulässig Stgt 272?",
        "Building coverage ratio Stgt 272",
        "Setback requirements Nordbahnhofstrasse",
        "Solar panel regulations Stgt 272",
    ],
    
    "Stgt 283 - Killesberg/Maybachstraße": [
        "Tell me about Killesberg Bebauungsplan Stgt 283",
        "Building regulations for Maybachstraße",
        "What are the requirements for Killesberg district?",
        "Stgt 283 zoning regulations",
    ],
    
    "General Stuttgart Regulations": [
        "Parking requirements for residential buildings Stuttgart",
        "Barrier-free building requirements DIN 18040",
        "Stuttgart energy and climate protection buildings",
        "Building permit process Stuttgart",
        "Accessibility requirements entrances",
        "Stellplätze Wohngebäude Stuttgart",
    ],
    
    "Bad Cannstatt District": [
        "Building regulations for Bad Cannstatt district",
        "Development plan Bad Cannstatt",
        "Local building statutes Bad Cannstatt",
    ]
}

# Track overall stats
total_queries = sum(len(queries) for queries in test_categories.values())
queries_with_results = 0
queries_with_good_scores = 0
query_num = 0

print("="*70)
print("📊 TESTING BY CATEGORY")
print("="*70)

for category, queries in test_categories.items():
    print(f"\n{'='*70}")
    print(f"📂 CATEGORY: {category}")
    print(f"{'='*70}")
    
    for query in queries:
        query_num += 1
        print(f"\n[{query_num}/{total_queries}] ❓ {query}")
        print("-"*70)
        
        try:
            results = rag.search(query, top_k=3)
            
            if not results:
                print("⚠️  No results found")
                continue
            
            queries_with_results += 1
            best_score = results[0].score if results else 0
            
            if best_score > 0.7:
                queries_with_good_scores += 1
                print(f"✅ Good match (score: {best_score:.3f})")
            elif best_score > 0.5:
                print(f"⚠️  Moderate match (score: {best_score:.3f})")
            else:
                print(f"❌ Weak match (score: {best_score:.3f})")
            
            # Show top result
            top_result = results[0]
            print(f"\n[Top Result]")
            print(f"  Source: {top_result.source}")
            print(f"  Score: {top_result.score:.3f}")
            
            # Show content preview
            content_preview = top_result.content[:200].replace('\n', ' ')
            print(f"  Content: {content_preview}...")
            
            # Show metadata if available
            if hasattr(top_result, 'metadata') and top_result.metadata:
                meta = top_result.metadata
                if 'district' in meta:
                    print(f"  District: {meta['district']}")
                if 'document_type' in meta:
                    print(f"  Type: {meta['document_type']}")
                if 'bebauungsplan_number' in meta:
                    print(f"  Bebauungsplan: {meta['bebauungsplan_number']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

print("\n" + "="*70)
print("📊 SUMMARY STATISTICS")
print("="*70)
print(f"Total queries tested:         {total_queries}")
print(f"Queries with results:         {queries_with_results} ({queries_with_results/total_queries*100:.1f}%)")
print(f"Queries with good scores:     {queries_with_good_scores} ({queries_with_good_scores/total_queries*100:.1f}%)")
print(f"Coverage rate:                {queries_with_results/total_queries*100:.1f}%")
print()

# Performance rating
if queries_with_good_scores / total_queries > 0.7:
    rating = "✅ EXCELLENT - System has good coverage of your data"
elif queries_with_good_scores / total_queries > 0.5:
    rating = "⚠️  GOOD - Most queries work, some areas need more data"
elif queries_with_good_scores / total_queries > 0.3:
    rating = "⚠️  MODERATE - Coverage gaps exist, consider adding more documents"
else:
    rating = "❌ POOR - Significant coverage gaps, need more documents"

print(f"Overall Rating: {rating}")
print()

print("="*70)
print("💡 RECOMMENDATIONS")
print("="*70)

if queries_with_good_scores < total_queries * 0.7:
    print("\n⚠️  Detected Coverage Gaps:")
    print("   - Consider adding more Bebauungsplan documents")
    print("   - Add specific district regulations")
    print("   - Include technical standards (DIN documents)")
    print("   - Verify OCR quality for existing documents")
else:
    print("\n✅ Good coverage! Your system can handle:")
    print("   - Area-specific queries (Nordbahnhof, Killesberg)")
    print("   - General Stuttgart regulations")
    print("   - Technical/accessibility requirements")

print()
print("="*70)
print("✅ RAG TESTING COMPLETE")
print("="*70)
print()
print("Next step: Start the API server with 'python multi_agent_app.py'")