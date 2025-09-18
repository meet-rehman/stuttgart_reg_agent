# precomputed_rag.py
# Lightweight RAG system that loads pre-computed embeddings

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class BuildingResult:
    """Simple result class for search results"""
    def __init__(self, content: str, score: float, metadata: Dict = None, source: str = None):
        self.content = content
        self.score = score
        self.metadata = metadata or {}
        self.source = source

class PrecomputedRAGSystem:
    """Lightweight RAG system using pre-computed embeddings"""
    
    def __init__(self, embeddings_dir: Optional[Path] = None):
        # Auto-detect embeddings directory if not provided
        if embeddings_dir is None:
            current_dir = Path(__file__).parent
            embeddings_dir = current_dir / "embeddings"
        
        self.embeddings_dir = Path(embeddings_dir)
        self.model = None
        self.documents = []
        self.metadata = []
        self.embeddings = None
        self._ready = False
        
        print(f"RAG system initialized with embeddings dir: {self.embeddings_dir}")
        
    def is_ready(self) -> bool:
        """Check if the system is ready"""
        return self._ready
        
    async def initialize(self):
        """Initialize the RAG system with pre-computed data"""
        try:
            print("Loading pre-computed embeddings...")
            
            # Check if embeddings directory exists
            if not self.embeddings_dir.exists():
                print(f"Embeddings directory not found: {self.embeddings_dir}")
                print("Please run build_embeddings_locally.py first")
                return
            
            # Check if all required files exist
            documents_file = self.embeddings_dir / "documents.json"
            embeddings_file = self.embeddings_dir / "embeddings.npy"
            model_info_file = self.embeddings_dir / "model_info.json"
            
            missing_files = []
            for file_path, name in [(documents_file, "documents.json"), 
                                   (embeddings_file, "embeddings.npy"), 
                                   (model_info_file, "model_info.json")]:
                if not file_path.exists():
                    missing_files.append(name)
            
            if missing_files:
                print(f"Missing files: {missing_files}")
                print("Please run build_embeddings_locally.py first")
                return
            
            print("Loading model info...")
            with open(model_info_file, 'r') as f:
                model_info = json.load(f)
            
            print(f"Model: {model_info['model_name']}")
            print(f"Total chunks: {model_info['total_chunks']}")
            
            print("Loading documents and metadata...")
            with open(documents_file, 'r', encoding='utf-8') as f:
                documents_data = json.load(f)
            
            self.documents = documents_data['documents']
            self.metadata = documents_data.get('metadata', [])
            
            print(f"Loaded {len(self.documents)} document chunks")
            
            print("Loading embeddings array...")
            self.embeddings = np.load(embeddings_file)
            print(f"Loaded embeddings: {self.embeddings.shape}")
            
            print("Loading query encoder model...")
            self.model = SentenceTransformer(model_info['model_name'])
            
            self._ready = True
            print("RAG system ready!")
            
        except Exception as e:
            print(f"Error initializing RAG system: {e}")
            logger.error(f"RAG initialization error: {e}")
            import traceback
            traceback.print_exc()
    
    async def search(self, query: str, top_k: int = 5) -> List[BuildingResult]:
        """Search for relevant documents"""
        if not self._ready:
            print("RAG system not ready")
            return []
        
        try:
            print(f"Searching for: {query[:50]}...")
            
            # Encode the query
            query_embedding = self.model.encode([query])
            
            # Calculate cosine similarities
            similarities = np.dot(self.embeddings, query_embedding.T).flatten()
            
            # Get top-k most similar documents
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Return the most relevant documents
            results = []
            for idx in top_indices:
                if idx < len(self.documents) and similarities[idx] > 0.1:  # Minimum similarity threshold
                    source = None
                    if idx < len(self.metadata):
                        source = self.metadata[idx].get('source', 'unknown')
                    
                    result = BuildingResult(
                        content=self.documents[idx],
                        score=float(similarities[idx]),
                        metadata=self.metadata[idx] if idx < len(self.metadata) else {},
                        source=source
                    )
                    results.append(result)
            
            print(f"Found {len(results)} relevant results")
            return results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            print(f"Search error: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            'total_documents': len(self.documents),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'is_ready': self._ready,
            'sources': list(set([meta.get('source', 'unknown') for meta in self.metadata])) if self.metadata else []
        }