# src/stuttgart_reg_agent/optimized_rag.py

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
import time

import numpy as np
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from sklearn.metrics.pairwise import cosine_similarity


class OptimizedRAGSystem:
    """Optimized RAG system with caching and fast startup"""
    
    def __init__(self, data_dir: Path, memory_dir: Path):
        self.data_dir = Path(data_dir)
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache files
        self.embeddings_cache = self.memory_dir / "embeddings_cache.pkl"
        self.documents_cache = self.memory_dir / "documents_cache.json"
        self.index_cache = self.memory_dir / "index_metadata.json"
        
        # In-memory storage
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.document_metadata: List[Dict] = []
        
        # Model (load only when needed)
        self._embedder = None
        self.is_ready = False
        
    @property
    def embedder(self):
        """Lazy load embedder model"""
        if self._embedder is None:
            print("Loading embedding model...")
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder
    
    async def initialize(self):
        """Initialize RAG system - fast startup with caching"""
        print("Initializing RAG system...")
        
        # Check if we have cached embeddings
        if self._has_valid_cache():
            print("Loading from cache...")
            self._load_from_cache()
        else:
            print("Building new index...")
            await self._build_index()
        
        self.is_ready = True
        print(f"RAG system ready with {len(self.documents)} documents")
    
    def _has_valid_cache(self) -> bool:
        """Check if cache is valid and up to date"""
        if not all([
            self.embeddings_cache.exists(),
            self.documents_cache.exists(),
            self.index_cache.exists()
        ]):
            return False
        
        try:
            with open(self.index_cache, 'r') as f:
                metadata = json.load(f)
            
            # Check if PDF files have changed
            current_files = list(self.data_dir.glob("**/*.pdf"))
            cached_files = metadata.get('files', {})
            
            if len(current_files) != len(cached_files):
                return False
            
            for pdf_path in current_files:
                file_hash = self._get_file_hash(pdf_path)
                cached_hash = cached_files.get(str(pdf_path), "")
                if file_hash != cached_hash:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _load_from_cache(self):
        """Load embeddings and documents from cache"""
        # Load documents
        with open(self.documents_cache, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
            self.documents = cache_data['documents']
            self.document_metadata = cache_data['metadata']
        
        # Load embeddings
        with open(self.embeddings_cache, 'rb') as f:
            self.embeddings = pickle.load(f)
    
    async def _build_index(self):
        """Build new index from PDF files"""
        pdf_files = list(self.data_dir.glob("**/*.pdf"))
        
        # Remove duplicates based on file name and size
        unique_files = {}
        for pdf_path in pdf_files:
            key = (pdf_path.name, pdf_path.stat().st_size)
            if key not in unique_files:
                unique_files[key] = pdf_path
        
        pdf_files = list(unique_files.values())
        
        if not pdf_files:
            print("⚠️ No PDF files found in data directory")
            self.documents = ["No building regulations documents available."]
            self.document_metadata = [{"source": "default", "page": 0}]
            self.embeddings = self.embedder.encode(self.documents)
            return
        
        print(f"Processing {len(pdf_files)} PDF files...")
        
        documents = []
        metadata = []
        file_hashes = {}
        
        for pdf_path in pdf_files:
            print(f"Processing {pdf_path.name}...")
            
            # Extract text
            text_chunks = self._extract_text_from_pdf(pdf_path)
            
            for i, chunk in enumerate(text_chunks):
                if chunk.strip():  # Only add non-empty chunks
                    documents.append(chunk)
                    metadata.append({
                        "source": pdf_path.name,
                        "page": i,
                        "path": str(pdf_path)
                    })
            
            # Store file hash for cache validation
            file_hashes[str(pdf_path)] = self._get_file_hash(pdf_path)
        
        if not documents:
            print("⚠️ No text extracted from PDFs")
            documents = ["No building regulations content available."]
            metadata = [{"source": "default", "page": 0}]
        
        self.documents = documents
        self.document_metadata = metadata
        
        # Generate embeddings in batches for memory efficiency
        print("Generating embeddings...")
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_embeddings = self.embedder.encode(batch, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        
        self.embeddings = np.vstack(embeddings) if embeddings else np.array([])
        
        # Save to cache
        self._save_to_cache(file_hashes)
        
        print(f"Index built with {len(documents)} text chunks")
    
    def _extract_text_from_pdf(self, pdf_path: Path, max_chars_per_chunk: int = 2000) -> List[str]:
        """Extract text from PDF and split into chunks"""
        chunks = []
        
        try:
            doc = fitz.open(pdf_path)
            current_chunk = ""
            
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                
                if not page_text.strip():
                    continue
                
                # Split page text into sentences to avoid cutting mid-sentence
                sentences = page_text.replace('\n', ' ').split('. ')
                
                for sentence in sentences:
                    sentence = sentence.strip() + '. '
                    
                    if len(current_chunk) + len(sentence) > max_chars_per_chunk:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = sentence
                        else:
                            # Single sentence is too long, just add it
                            chunks.append(sentence)
                    else:
                        current_chunk += sentence
            
            # Add remaining chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            doc.close()
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            chunks.append(f"Error reading document {pdf_path.name}")
        
        return chunks
    
    def _save_to_cache(self, file_hashes: Dict[str, str]):
        """Save embeddings and documents to cache"""
        # Save documents and metadata
        cache_data = {
            'documents': self.documents,
            'metadata': self.document_metadata
        }
        with open(self.documents_cache, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        # Save embeddings
        with open(self.embeddings_cache, 'wb') as f:
            pickle.dump(self.embeddings, f)
        
        # Save index metadata
        index_metadata = {
            'created_at': time.time(),
            'files': file_hashes,
            'document_count': len(self.documents)
        }
        with open(self.index_cache, 'w') as f:
            json.dump(index_metadata, f, indent=2)
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file for change detection"""
        hash_obj = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception:
            return ""
    
    def search(self, query: str, top_k: int = 5) -> List[str]:
        """Search for relevant documents"""
        if not self.is_ready or len(self.documents) == 0:
            return ["RAG system not ready or no documents available."]
        
        try:
            # Encode query
            query_embedding = self.embedder.encode([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top-k results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Return documents with similarity > threshold
            threshold = 0.1
            results = []
            for idx in top_indices:
                if similarities[idx] > threshold:
                    results.append(self.documents[idx])
            
            return results if results else ["No relevant information found."]
            
        except Exception as e:
            print(f"Search error: {e}")
            return ["Search error occurred."]
    
    async def reindex_documents(self):
        """Reindex all documents (for background updates)"""
        print("Starting background reindexing...")
        
        # Clear cache
        for cache_file in [self.embeddings_cache, self.documents_cache, self.index_cache]:
            if cache_file.exists():
                cache_file.unlink()
        
        # Rebuild index
        await self._build_index()
        print("Background reindexing completed")


# Usage example
if __name__ == "__main__":
    import asyncio
    
    async def test_rag():
        rag = OptimizedRAGSystem(
            data_dir=Path("data"),
            memory_dir=Path("memory")
        )
        
        await rag.initialize()
        
        # Test search
        results = rag.search("building height regulations Stuttgart")
        for i, result in enumerate(results):
            print(f"Result {i+1}: {result[:200]}...")
    
    asyncio.run(test_rag())