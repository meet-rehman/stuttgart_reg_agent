# enhanced_precomputed_rag.py
# Enhanced RAG system with detailed citations and PropTech focus

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class BuildingResult:
    """Enhanced result class with detailed citations"""
    def __init__(self, content: str, score: float, metadata: Dict = None, source: str = None, citation: str = None, document_id: str = None):
        self.content = content
        self.score = score
        self.metadata = metadata or {}
        self.source = source
        self.citation = citation
        self.document_id = document_id
    
    def get_detailed_citation(self) -> str:
        """Generate detailed citation with all available metadata"""
        citation_parts = []
        
        # Document type and name
        doc_type = self.metadata.get("document_type", "Document")
        doc_name = self.metadata.get("document_name", "Unknown")
        citation_parts.append(f"{doc_type}: {doc_name}")
        
        # Page number
        page_num = self.metadata.get("page_number")
        if page_num:
            citation_parts.append(f"Page {page_num}")
        
        # Legal sections
        sections = self.metadata.get("sections", [])
        if sections:
            citation_parts.append(f"Section(s): {', '.join(sections[:3])}")
        
        # Form numbers
        forms = self.metadata.get("form_numbers", [])
        if forms:
            citation_parts.append(f"Form(s): {', '.join(forms[:2])}")
        
        # Official IDs
        ids = self.metadata.get("official_ids", [])
        if ids:
            citation_parts.append(f"ID(s): {', '.join(ids[:2])}")
        
        return " | ".join(citation_parts)
    
    def get_district_specific_info(self) -> Dict:
        """Extract district-specific information"""
        district_info = self.metadata.get("district_specific", {})
        return {
            "mentioned_districts": district_info.get("mentioned_districts", []),
            "specific_rules": district_info.get("specific_rules", [])
        }

class EnhancedPrecomputedRAGSystem:
    """Enhanced RAG system with PropTech-focused features"""
    
    def __init__(self, embeddings_dir: Optional[Path] = None):
        """Initialize the enhanced RAG system"""
        if embeddings_dir is None:
            current_dir = Path(__file__).parent
            embeddings_dir = current_dir / "embeddings"
        
        self.embeddings_dir = Path(embeddings_dir)
        self.model = None
        self.documents = []
        self.embeddings = None
        self.is_ready = False
        
        logger.info(f"Initializing enhanced RAG system from: {self.embeddings_dir}")
        
        try:
            # Initialize the model
            logger.info("Loading SentenceTransformer model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SentenceTransformer model loaded successfully")
            
            # Load enhanced precomputed data
            self._load_precomputed_data()
            
            self.is_ready = True
            logger.info("Enhanced RAG system initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced RAG system: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def _load_precomputed_data(self):
        """Load precomputed embeddings and enhanced documents"""
        try:
            if not self.embeddings_dir.exists():
                logger.warning(f"Embeddings directory not found: {self.embeddings_dir}")
                self.documents = []
                self.embeddings = np.array([]).reshape(0, 384)
                return
            
            # Load enhanced documents
            documents_path = self.embeddings_dir / "documents.json"
            if documents_path.exists():
                with open(documents_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                logger.info(f"Loaded {len(self.documents)} enhanced documents")
                
                # Log metadata richness
                if self.documents:
                    sample_doc = self.documents[0]
                    metadata_fields = len(sample_doc.get("metadata", {}))
                    logger.info(f"Enhanced metadata includes {metadata_fields} fields per document")
            else:
                logger.warning("documents.json not found")
                self.documents = []
            
            # Load embeddings
            embeddings_path = self.embeddings_dir / "embeddings.npy"
            if embeddings_path.exists():
                self.embeddings = np.load(embeddings_path)
                logger.info(f"Loaded embeddings with shape: {self.embeddings.shape}")
            else:
                logger.warning("embeddings.npy not found")
                self.embeddings = np.array([]).reshape(0, 384)
                
        except Exception as e:
            logger.error(f"Error loading enhanced precomputed data: {str(e)}")
            self.documents = []
            self.embeddings = np.array([]).reshape(0, 384)

    def search(self, query: str, top_k: int = 5, filter_district: Optional[str] = None, filter_document_type: Optional[str] = None) -> List[BuildingResult]:
        """Enhanced search with filtering capabilities"""
        if not self.is_ready:
            logger.error("Enhanced RAG system not ready")
            return []
            
        if len(self.documents) == 0:
            logger.warning("No documents available for search")
            return [BuildingResult(
                content="No Stuttgart building regulation documents are currently loaded. Please check the system configuration.",
                score=0.0,
                metadata={"type": "system_error"}
            )]
        
        try:
            # Get query embedding
            query_embedding = self.model.encode([query])
            
            if self.embeddings.shape[0] > 0:
                # Calculate similarities
                similarities = np.dot(query_embedding, self.embeddings.T).flatten()
                
                # Get top k results
                top_indices = np.argsort(similarities)[-top_k*2:][::-1]  # Get more results for filtering
                
                results = []
                for idx in top_indices:
                    if idx < len(self.documents) and len(results) < top_k:
                        doc = self.documents[idx]
                        
                        # Apply filters
                        if filter_district:
                            district_info = doc.get("metadata", {}).get("district_specific", {})
                            mentioned_districts = district_info.get("mentioned_districts", [])
                            if filter_district not in mentioned_districts:
                                continue
                        
                        if filter_document_type:
                            doc_type = doc.get("metadata", {}).get("document_type", "")
                            if filter_document_type.lower() not in doc_type.lower():
                                continue
                        
                        score = float(similarities[idx])
                        
                        # Create enhanced result
                        result = BuildingResult(
                            content=doc.get('content', ''),
                            score=score,
                            metadata=doc.get('metadata', {}),
                            source=doc.get('source', 'Unknown'),
                            citation=doc.get('citation', ''),
                            document_id=doc.get('document_id', '')
                        )
                        
                        results.append(result)
                
                return results
            else:
                return [BuildingResult(
                    content="No document embeddings available for search.",
                    score=0.0,
                    metadata={"type": "no_embeddings"}
                )]
                
        except Exception as e:
            logger.error(f"Error during enhanced search: {str(e)}")
            return [BuildingResult(
                content="Technical error occurred during document search. Please try again.",
                score=0.0,
                metadata={"type": "search_error", "error": str(e)}
            )]

    def get_context_for_query(self, query: str, max_tokens: int = 2000, include_citations: bool = True) -> str:
        """Get enhanced context with detailed citations"""
        results = self.search(query, top_k=4)
        
        if not results:
            return "No relevant Stuttgart building regulation documents found for this query."
        
        context_parts = []
        total_length = 0
        
        for i, result in enumerate(results):
            if include_citations:
                citation = result.get_detailed_citation()
                content_with_citation = f"[Source {i+1}] {citation}\n\nContent: {result.content}"
                
                # Add district-specific info if available
                district_info = result.get_district_specific_info()
                if district_info["mentioned_districts"]:
                    content_with_citation += f"\n\nDistrict(s): {', '.join(district_info['mentioned_districts'])}"
                
                if district_info["specific_rules"]:
                    for rule in district_info["specific_rules"][:2]:  # Limit to 2 rules
                        content_with_citation += f"\n• {rule['district']}: {rule['rule']}"
                
            else:
                content_with_citation = result.content
            
            if total_length + len(content_with_citation) > max_tokens:
                break
                
            context_parts.append(content_with_citation)
            total_length += len(content_with_citation)
        
        return "\n\n" + "="*50 + "\n\n".join(context_parts)

    def search_by_district(self, query: str, district: str, top_k: int = 3) -> List[BuildingResult]:
        """Search specifically for district-related regulations"""
        return self.search(query, top_k=top_k, filter_district=district)

    def search_by_document_type(self, query: str, doc_type: str, top_k: int = 3) -> List[BuildingResult]:
        """Search within specific document types"""
        return self.search(query, top_k=top_k, filter_document_type=doc_type)

    def get_forms_for_process(self, process_type: str) -> List[Dict]:
        """Get relevant forms for a specific building process"""
        forms = []
        for doc in self.documents:
            metadata = doc.get("metadata", {})
            form_numbers = metadata.get("form_numbers", [])
            if form_numbers and process_type.lower() in doc.get("content", "").lower():
                forms.append({
                    "form_numbers": form_numbers,
                    "document_name": metadata.get("document_name", ""),
                    "content_preview": doc.get("content", "")[:200] + "..."
                })
        return forms[:5]  # Return top 5 relevant forms