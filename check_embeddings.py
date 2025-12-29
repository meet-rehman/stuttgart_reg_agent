import json
import numpy as np
from pathlib import Path
from collections import Counter

def check_embeddings_setup():
    """Check current embeddings configuration"""
    
    project_root = Path(__file__).parent
    embeddings_dir = project_root / "embeddings"
    
    print("="*60)
    print("📊 EMBEDDINGS DIRECTORY CHECK")
    print("="*60)
    
    print(f"\n📁 Project Root: {project_root}")
    print(f"📁 Embeddings Directory: {embeddings_dir}")
    print(f"✅ Exists: {embeddings_dir.exists()}")
    
    if embeddings_dir.exists():
        print("\n📄 Files in embeddings directory:")
        
        for file in embeddings_dir.iterdir():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   - {file.name}: {size_mb:.2f} MB")
        
        # Check documents.json
        doc_file = embeddings_dir / "documents.json"
        if doc_file.exists():
            with open(doc_file, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            print(f"\n📚 Total documents loaded: {len(documents)}")
            
            # Check first document structure
            if documents:
                print(f"\n📝 Document structure:")
                sample = documents[0]
                for key in sample.keys():
                    value = sample[key]
                    if isinstance(value, str):
                        preview = value[:80] + "..." if len(value) > 80 else value
                        print(f"   {key}: {preview}")
                    else:
                        print(f"   {key}: {type(value).__name__}")
                
                # Count by actual source field
                print(f"\n📋 Document sources:")
                sources = Counter()
                for doc in documents:
                    if isinstance(doc, dict):
                        # Try multiple possible source fields
                        source = (doc.get('source') or 
                                 doc.get('metadata', {}).get('source') or
                                 doc.get('metadata', {}).get('file') or
                                 'unknown')
                        sources[source] += 1
                
                print(f"\n   Found {len(sources)} unique sources:")
                for source, count in sources.most_common(20):  # Show top 20
                    print(f"   - {source}: {count} chunks")
                
                if len(sources) > 20:
                    print(f"   ... and {len(sources) - 20} more sources")
        
        # Check embeddings.npy
        emb_file = embeddings_dir / "embeddings.npy"
        if emb_file.exists():
            embeddings = np.load(emb_file)
            print(f"\n🔢 Embeddings:")
            print(f"   Shape: {embeddings.shape}")
            print(f"   Documents: {embeddings.shape[0]:,}")
            print(f"   Dimensions: {embeddings.shape[1]}")
            print(f"   Memory: {embeddings.nbytes / (1024*1024):.2f} MB")
        
        # Check model_info.json
        model_file = embeddings_dir / "model_info.json"
        if model_file.exists():
            with open(model_file, 'r') as f:
                model_info = json.load(f)
            print(f"\n🤖 Model:")
            print(f"   Name: {model_info.get('model_name', 'unknown')}")
            print(f"   Dimension: {model_info.get('embedding_dim', 'unknown')}")
            print(f"   Created: {model_info.get('created_at', 'unknown')}")
        
        # Consistency check
        print(f"\n🔍 Consistency:")
        if doc_file.exists() and emb_file.exists():
            num_docs = len(documents)
            num_embeddings = embeddings.shape[0]
            if num_docs == num_embeddings:
                print(f"   ✅ {num_docs:,} documents = {num_embeddings:,} embeddings")
            else:
                print(f"   ⚠️  MISMATCH: {num_docs:,} docs vs {num_embeddings:,} embeddings")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    check_embeddings_setup()