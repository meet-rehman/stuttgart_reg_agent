from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

# FIX: Use the correct class name
from precomputed_rag import EnhancedPrecomputedRAGSystem

def test_rag():
    """Test the merged RAG system"""
    
    print("="*60)
    print("🧪 TESTING COMPREHENSIVE RAG SYSTEM")
    print("="*60)
    
    print("\n📚 Initializing RAG system...")
    rag = EnhancedPrecomputedRAGSystem()  # ← Fixed class name
    
    if not rag.is_ready:
        print("❌ RAG system not ready!")
        return
    
    # Check what we have
    print(f"✅ RAG loaded successfully")
    
    # Test queries for different document types
    test_queries = [
        # Bebauungsplan queries
        ("What is the GRZ for Stgt 272?", "Bebauungsplan - Stgt 272"),
        ("Height restrictions in Nordbahnhof area", "Nordbahnhof"),
        ("Tell me about Bürgerhospital Bebauungsplan", "Bürgerhospital"),
        
        # Barrier-free building
        ("Requirements for barrier-free building", "Accessibility"),
        ("Wheelchair accessibility standards", "Accessibility"),
        
        # General building regulations
        ("Parking requirements for residential buildings", "Building Code"),
        ("Fire safety regulations", "Building Safety"),
        
        # Federal/State law
        ("What does BauGB say about building permits?", "Federal Law"),
        ("Baden-Württemberg building regulations", "State Law"),
        
        # Local statutes
        ("Local statutes for Stuttgart", "Local Regulations"),
        ("Killesberg building requirements", "Killesberg"),
    ]
    
    for i, (query, category) in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}/{len(test_queries)} - {category}")
        print(f"{'='*60}")
        print(f"❓ Query: {query}")
        print("-"*60)
        
        try:
            results = rag.search(query, top_k=3)
            
            if not results:
                print("⚠️  No results found")
                continue
            
            for j, result in enumerate(results, 1):
                print(f"\n[Result {j}] Relevance Score: {result.score:.4f}")
                print(f"📄 Source: {result.source}")
                
                # Show metadata
                if hasattr(result, 'metadata') and result.metadata:
                    doc_type = result.metadata.get('document_type', 'Unknown')
                    district = result.metadata.get('district', 'Unknown')
                    print(f"📋 Type: {doc_type}")
                    print(f"📍 District: {district}")
                
                # Content preview
                content = result.content[:250].replace('\n', ' ')
                print(f"📝 Content: {content}...")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("✅ Testing Complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_rag()