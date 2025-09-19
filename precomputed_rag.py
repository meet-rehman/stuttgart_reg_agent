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
        """Initialize the RAG system with pre-computed data"""
        # Auto-detect embeddings directory if not provided
        if embeddings_dir is None:
            current_dir = Path(__file__).parent
            embeddings_dir = current_dir / "embeddings"
        
        self.embeddings_dir = Path(embeddings_dir)
        self.model = None
        self.documents = []
        self.embeddings = None
        self.is_ready = False
        
        logger.info(f"Initializing RAG system with embeddings from: {self.embeddings_dir}")
        
        try:
            # Initialize the model first
            logger.info("Loading SentenceTransformer model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SentenceTransformer model loaded successfully")
            
            # Load precomputed data
            self._load_precomputed_data()
            
            self.is_ready = True
            logger.info("RAG system initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def _load_precomputed_data(self):
        """Load precomputed embeddings and documents"""
        try:
            # Check if embeddings directory exists
            if not self.embeddings_dir.exists():
                logger.warning(f"Embeddings directory not found: {self.embeddings_dir}")
                logger.info("Creating empty embeddings structure...")
                self.documents = []
                self.embeddings = np.array([]).reshape(0, 384)  # Empty array with correct dimensions
                return
            
            # Load documents
            documents_path = self.embeddings_dir / "documents.json"
            if documents_path.exists():
                with open(documents_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                logger.info(f"Loaded {len(self.documents)} documents")
            else:
                logger.warning("documents.json not found, using empty document set")
                self.documents = []
            
            # Load embeddings
            embeddings_path = self.embeddings_dir / "embeddings.npy"
            if embeddings_path.exists():
                self.embeddings = np.load(embeddings_path)
                logger.info(f"Loaded embeddings with shape: {self.embeddings.shape}")
            else:
                logger.warning("embeddings.npy not found, using empty embeddings")
                self.embeddings = np.array([]).reshape(0, 384)  # Empty array with correct dimensions
                
        except Exception as e:
            logger.error(f"Error loading precomputed data: {str(e)}")
            # Fallback to empty data
            self.documents = []
            self.embeddings = np.array([]).reshape(0, 384)

    def search(self, query: str, top_k: int = 5) -> List[BuildingResult]:
        """Search for relevant documents using semantic similarity"""
        if not self.is_ready:
            logger.error("RAG system not ready")
            return []
            
        if len(self.documents) == 0:
            logger.warning("No documents available for search")
            return [BuildingResult(
                content="I don't have any specific building registration documents loaded right now. However, I can help you with general information about building regulations in Stuttgart.",
                score=0.5,
                metadata={"type": "fallback"}
            )]
        
        try:
            # Get query embedding
            query_embedding = self.model.encode([query])
            
            # Calculate similarities
            if self.embeddings.shape[0] > 0:
                similarities = np.dot(query_embedding, self.embeddings.T).flatten()
                
                # Get top k results
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                results = []
                for idx in top_indices:
                    if idx < len(self.documents):
                        doc = self.documents[idx]
                        score = float(similarities[idx])
                        
                        results.append(BuildingResult(
                            content=doc.get('content', ''),
                            score=score,
                            metadata=doc.get('metadata', {}),
                            source=doc.get('source', 'Unknown')
                        ))
                
                return results
            else:
                # No embeddings available, return fallback
                return [BuildingResult(
                    content="I don't have specific building registration documents loaded, but I can help with general Stuttgart building regulation information.",
                    score=0.3,
                    metadata={"type": "no_embeddings"}
                )]
                
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return [BuildingResult(
                content="I'm having technical difficulties accessing the building documents right now. Please try again in a moment.",
                score=0.1,
                metadata={"type": "error", "error": str(e)}
            )]

    def get_context_for_query(self, query: str, max_tokens: int = 2000) -> str:
        """Get relevant context for a query, formatted for the LLM"""
        results = self.search(query, top_k=3)
        
        if not results:
            return "No specific building registration documents are currently available."
        
        context_parts = []
        total_length = 0
        
        for result in results:
            content = result.content[:500]  # Limit each result
            if total_length + len(content) > max_tokens:
                break
            context_parts.append(f"[Score: {result.score:.3f}] {content}")
            total_length += len(content)
        
        return "\n\n---\n\n".join(context_parts)