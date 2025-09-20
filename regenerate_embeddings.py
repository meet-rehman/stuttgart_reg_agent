#!/usr/bin/env python3
"""
Enhanced document processor that captures detailed metadata for PropTech bot
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDocumentProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = []
    
    def extract_enhanced_metadata(self, pdf_path: Path, text: str, page_num: int) -> Dict:
        """Extract enhanced metadata from document structure"""
        metadata = {
            "document_name": pdf_path.stem,
            "file_extension": pdf_path.suffix,
            "category": pdf_path.parent.name,
            "page_number": page_num + 1,
            "file_path": str(pdf_path),
            "document_type": self.identify_document_type(pdf_path.name, text),
            "sections": self.extract_sections(text),
            "legal_references": self.extract_legal_references(text),
            "form_numbers": self.extract_form_numbers(text),
            "official_ids": self.extract_official_ids(text),
            "district_specific": self.extract_district_info(text),
        }
        return metadata
    
    def identify_document_type(self, filename: str, text: str) -> str:
        """Identify the type of document based on filename and content"""
        filename_lower = filename.lower()
        text_lower = text.lower()
        
        if any(term in filename_lower for term in ['bauordnung', 'lbo', 'building_code']):
            return "Building Code (LBO)"
        elif any(term in filename_lower for term in ['baugb', 'building_law']):
            return "Federal Building Law (BauGB)"
        elif any(term in filename_lower for term in ['bebauungsplan', 'zoning']):
            return "Zoning Plan"
        elif any(term in filename_lower for term in ['energieeinspar', 'energy']):
            return "Energy Efficiency Regulation"
        elif any(term in filename_lower for term in ['antrag', 'form', 'application']):
            return "Application Form"
        elif any(term in text_lower for term in ['paragraph', '§', 'article']):
            return "Legal Regulation"
        else:
            return "Municipal Document"
    
    def extract_sections(self, text: str) -> List[str]:
        """Extract section headers and numbers"""
        sections = []
        # Look for section patterns like "§ 5", "Article 3", "Section 2.1"
        section_patterns = [
            r'§\s*(\d+[a-z]?)',
            r'Article\s+(\d+)',
            r'Section\s+(\d+(?:\.\d+)?)',
            r'Abschnitt\s+(\d+)',
            r'Artikel\s+(\d+)'
        ]
        
        for pattern in section_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            sections.extend([f"§{match}" if not pattern.startswith('§') else f"§{match}" for match in matches])
        
        return list(set(sections))  # Remove duplicates
    
    def extract_legal_references(self, text: str) -> List[str]:
        """Extract legal references and citations"""
        references = []
        
        # Common legal reference patterns
        patterns = [
            r'LBO\s+BW\s+[§]*\s*(\d+)',
            r'BauGB\s+[§]*\s*(\d+)',
            r'DIN\s+(\d+(?:-\d+)?)',
            r'EnEV\s+[§]*\s*(\d+)',
            r'nach\s+[§]*\s*(\d+)\s+LBO',
            r'gemäß\s+[§]*\s*(\d+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            references.extend(matches)
        
        return list(set(references))
    
    def extract_form_numbers(self, text: str) -> List[str]:
        """Extract form numbers and official document IDs"""
        form_numbers = []
        
        patterns = [
            r'Formular\s+([A-Z0-9-]+)',
            r'Form\s+([A-Z0-9-]+)',
            r'Antrag\s+([A-Z0-9-]+)',
            r'Nr\.\s*([A-Z0-9-]+)',
            r'Nummer\s+([A-Z0-9-]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            form_numbers.extend(matches)
        
        return list(set(form_numbers))
    
    def extract_official_ids(self, text: str) -> List[str]:
        """Extract official document IDs and reference numbers"""
        ids = []
        
        patterns = [
            r'Aktenzeichen\s+([A-Z0-9/-]+)',
            r'Az\.\s*([A-Z0-9/-]+)',
            r'Geschäftszeichen\s+([A-Z0-9/-]+)',
            r'GZ\s+([A-Z0-9/-]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            ids.extend(matches)
        
        return list(set(ids))
    
    def extract_district_info(self, text: str) -> Dict:
        """Extract district-specific information"""
        districts = {
            "mentioned_districts": [],
            "specific_rules": []
        }
        
        # Stuttgart districts
        stuttgart_districts = [
            'Zuffenhausen', 'Feuerbach', 'Weilimdorf', 'Kornthal-Münchingen',
            'Stammheim', 'Mühlhausen', 'Freiberg', 'Mönchfeld',
            'Bad Cannstatt', 'Sommerrain', 'Steinhaldenfeld'
        ]
        
        text_lower = text.lower()
        for district in stuttgart_districts:
            if district.lower() in text_lower:
                districts["mentioned_districts"].append(district)
                
                # Look for specific rules for this district
                district_context = self.extract_district_context(text, district)
                if district_context:
                    districts["specific_rules"].append({
                        "district": district,
                        "rule": district_context
                    })
        
        return districts
    
    def extract_district_context(self, text: str, district: str) -> str:
        """Extract context around district mentions"""
        sentences = text.split('.')
        for sentence in sentences:
            if district.lower() in sentence.lower():
                return sentence.strip()
        return ""
    
    def process_pdf_with_pages(self, pdf_path: Path, category: str) -> None:
        """Process PDF page by page for better granularity"""
        logger.info(f"Processing: {pdf_path.name}")
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                if not page_text.strip():
                    continue
                
                # Split page into chunks
                chunks = self.chunk_text(page_text)
                
                for chunk_i, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 50:  # Skip very short chunks
                        continue
                    
                    # Create enhanced metadata
                    metadata = self.extract_enhanced_metadata(pdf_path, chunk, page_num)
                    metadata["chunk_id"] = chunk_i
                    metadata["total_chunks_on_page"] = len(chunks)
                    
                    # Create document with enhanced structure
                    doc_entry = {
                        "content": chunk,
                        "metadata": metadata,
                        "source": f"{pdf_path.name}, Page {page_num + 1}, Section {chunk_i + 1}",
                        "citation": self.generate_citation(pdf_path, page_num + 1, metadata),
                        "document_id": f"{pdf_path.stem}_p{page_num + 1}_c{chunk_i}"
                    }
                    
                    self.documents.append(doc_entry)
                    
                    # Generate embedding
                    embedding = self.model.encode(chunk)
                    self.embeddings.append(embedding)
            
            doc.close()
            logger.info(f"Processed {pdf_path.name}: {len(doc)} pages")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
    
    def generate_citation(self, pdf_path: Path, page_num: int, metadata: Dict) -> str:
        """Generate proper citation format"""
        doc_type = metadata.get("document_type", "Document")
        sections = metadata.get("sections", [])
        
        citation = f"{doc_type}: {pdf_path.name}, Page {page_num}"
        
        if sections:
            citation += f", {', '.join(sections[:3])}"  # Include up to 3 sections
        
        return citation
    
    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks with better sentence boundaries"""
        if not text:
            return []
        
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed chunk size, save current chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                words = current_chunk.split()
                if len(words) > overlap:
                    current_chunk = " ".join(words[-overlap:]) + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_directory(self, data_dir: Path) -> None:
        """Process all PDF files with enhanced metadata extraction"""
        raw_dir = data_dir / "raw"
        
        if not raw_dir.exists():
            logger.error(f"Raw directory not found: {raw_dir}")
            return
        
        # Process each subdirectory
        for subdir in raw_dir.iterdir():
            if subdir.is_dir():
                category = subdir.name
                logger.info(f"Processing category: {category}")
                
                # Process all PDFs in this category
                pdf_files = list(subdir.glob("*.pdf"))
                logger.info(f"Found {len(pdf_files)} PDF files in {category}")
                
                for pdf_file in pdf_files:
                    self.process_pdf_with_pages(pdf_file, category)
    
    def save_embeddings(self, output_dir: Path) -> None:
        """Save enhanced documents and embeddings"""
        output_dir.mkdir(exist_ok=True)
        
        # Save documents with enhanced metadata
        documents_path = output_dir / "documents.json"
        with open(documents_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        
        # Save embeddings
        embeddings_array = np.array(self.embeddings)
        embeddings_path = output_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings_array)
        
        # Save enhanced model info
        model_info = {
            "model_name": "all-MiniLM-L6-v2",
            "embedding_dim": embeddings_array.shape[1],
            "document_count": len(self.documents),
            "embedding_count": embeddings_array.shape[0],
            "processing_date": "2024-09-19",
            "enhancement_level": "detailed_citations_and_metadata"
        }
        
        model_info_path = output_dir / "model_info.json"
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"✅ Enhanced embeddings saved!")
        logger.info(f"📊 Documents: {len(self.documents)}")
        logger.info(f"📊 Embeddings: {embeddings_array.shape[0]}")
        logger.info(f"📊 Average metadata fields per document: {self.calculate_metadata_richness()}")
    
    def calculate_metadata_richness(self) -> float:
        """Calculate average metadata richness"""
        if not self.documents:
            return 0
        
        total_fields = 0
        for doc in self.documents:
            metadata = doc.get("metadata", {})
            total_fields += len([v for v in metadata.values() if v])  # Count non-empty fields
        
        return total_fields / len(self.documents)

def main():
    """Main function to regenerate enhanced embeddings"""
    
    current_dir = Path(__file__).parent
    data_dir = current_dir / "data"
    embeddings_dir = current_dir / "embeddings"
    
    logger.info(f"🏗️ Starting enhanced PropTech document processing...")
    logger.info(f"📁 Data directory: {data_dir}")
    logger.info(f"📁 Output directory: {embeddings_dir}")
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    # Initialize enhanced processor
    processor = EnhancedDocumentProcessor()
    
    # Process all PDFs with enhanced metadata
    logger.info("🔍 Processing PDFs with enhanced metadata extraction...")
    processor.process_directory(data_dir)
    
    if not processor.documents:
        logger.error("❌ No documents were processed!")
        return
    
    # Save enhanced results
    logger.info("💾 Saving enhanced embeddings...")
    processor.save_embeddings(embeddings_dir)
    
    logger.info("🎉 Enhanced PropTech embeddings complete!")

if __name__ == "__main__":
    main()