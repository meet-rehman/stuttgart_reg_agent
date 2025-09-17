# document_analyzer.py - Analyze what documents your bot has processed

import json
import pickle
from pathlib import Path
import numpy as np
from collections import Counter

def analyze_document_processing():
    """Analyze what documents the RAG system has processed"""
    
    project_root = Path(".")
    memory_dir = project_root / "memory"
    
    # Check if cache files exist
    documents_cache = memory_dir / "documents_cache.json"
    embeddings_cache = memory_dir / "embeddings_cache.pkl"
    index_cache = memory_dir / "index_metadata.json"
    
    if not documents_cache.exists():
        print("❌ No document cache found. Run the server first to process PDFs.")
        return
    
    # Load document cache
    with open(documents_cache, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)
    
    documents = cache_data['documents']
    metadata = cache_data['metadata']
    
    # Load index metadata
    if index_cache.exists():
        with open(index_cache, 'r') as f:
            index_data = json.load(f)
    else:
        index_data = {}
    
    print("📊 DOCUMENT PROCESSING ANALYSIS")
    print("=" * 50)
    print(f"Total document chunks: {len(documents)}")
    print(f"Processing date: {index_data.get('created_at', 'Unknown')}")
    print()
    
    # Analyze by source file
    source_counts = Counter(meta['source'] for meta in metadata)
    print("📁 DOCUMENTS BY SOURCE FILE:")
    print("-" * 30)
    for source, count in source_counts.most_common():
        print(f"{source}: {count} chunks")
    print()
    
    # Show sample content from each document
    print("📄 SAMPLE CONTENT FROM EACH DOCUMENT:")
    print("-" * 40)
    
    sources_shown = set()
    for i, (doc, meta) in enumerate(zip(documents, metadata)):
        source = meta['source']
        if source not in sources_shown and len(sources_shown) < 5:  # Show max 5 documents
            sources_shown.add(source)
            print(f"\n🔹 {source} (Page {meta.get('page', 'Unknown')}):")
            print(f"   {doc[:200]}..." if len(doc) > 200 else f"   {doc}")
    
    if len(source_counts) > 5:
        print(f"\n... and {len(source_counts) - 5} more documents")
    
    # Show embedding info
    if embeddings_cache.exists():
        with open(embeddings_cache, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"\n🧠 EMBEDDINGS INFO:")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Memory usage: ~{embeddings.nbytes / 1024 / 1024:.1f} MB")
    
    # Test search functionality
    print(f"\n🔍 TESTING SEARCH FUNCTIONALITY:")
    print("-" * 35)
    
    # Simple similarity search simulation
    sample_queries = [
        "building height restrictions",
        "Stuttgart zoning regulations", 
        "accessibility requirements",
        "residential building codes"
    ]
    
    for query in sample_queries:
        # Find documents containing query words
        query_words = query.lower().split()
        matches = []
        
        for i, doc in enumerate(documents):
            doc_lower = doc.lower()
            if any(word in doc_lower for word in query_words):
                matches.append((i, metadata[i]['source']))
        
        print(f"Query: '{query}' -> {len(matches)} potential matches")
        if matches:
            # Show first match source
            print(f"   Best match from: {matches[0][1]}")
    
    print("\n✅ Analysis complete!")
    print(f"\nYour bot can answer questions about:")
    for source in source_counts.keys():
        print(f"   • {source}")

def test_rag_search_simulation():
    """Simulate how RAG search works with your documents"""
    memory_dir = Path("memory")
    documents_cache = memory_dir / "documents_cache.json"
    
    if not documents_cache.exists():
        print("❌ Run the server first to process documents")
        return
    
    with open(documents_cache, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)
    
    documents = cache_data['documents']
    metadata = cache_data['metadata']
    
    print("🔍 RAG SEARCH SIMULATION")
    print("=" * 30)
    
    test_query = "What are the height limits for residential buildings?"
    print(f"Test Query: '{test_query}'")
    print()
    
    # Simple keyword matching (actual system uses embeddings)
    keywords = ['height', 'limit', 'residential', 'building', 'floor']
    scores = []
    
    for i, doc in enumerate(documents):
        doc_lower = doc.lower()
        score = sum(1 for keyword in keywords if keyword in doc_lower)
        scores.append((score, i, metadata[i]['source']))
    
    # Sort by score (descending)
    scores.sort(reverse=True)
    
    print("Top 3 relevant document chunks:")
    for rank, (score, idx, source) in enumerate(scores[:3], 1):
        if score > 0:
            print(f"\n{rank}. From {source} (Score: {score})")
            print(f"   Content: {documents[idx][:300]}...")
    
    print(f"\n💡 The AI would use these chunks as context to answer your question!")

if __name__ == "__main__":
    print("Choose analysis type:")
    print("1. Document processing analysis")
    print("2. RAG search simulation")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        analyze_document_processing()
    elif choice == "2":
        test_rag_search_simulation()
    else:
        print("Invalid choice. Running document analysis...")
        analyze_document_processing()