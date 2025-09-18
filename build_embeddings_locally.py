# build_embeddings_locally.py
# Run this script locally to pre-compute embeddings

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import pypdf

# Load environment and find correct paths
PROJECT_ROOT = Path(__file__).resolve().parent

# Define possible data directory locations more precisely
possible_data_paths = [
    PROJECT_ROOT / "data",                    # Same directory as script
    PROJECT_ROOT / ".." / "data",            # One level up
    PROJECT_ROOT / ".." / ".." / "data",     # Two levels up  
    PROJECT_ROOT / "../../data",             # Alternative syntax
    Path("C:/Users/USER/ar/projects/Agents/3_crew/data")  # Absolute path
]

DATA_DIR = None
for path in possible_data_paths:
    abs_path = path.resolve()
    print(f"Checking: {abs_path}")
    if abs_path.exists():
        pdf_files = list(abs_path.rglob("*.pdf"))
        print(f"  Found {len(pdf_files)} PDF files")
        if pdf_files:
            DATA_DIR = abs_path
            break
        else:
            print("  Directory exists but no PDFs found")

if not DATA_DIR:
    print("\nManual search for data directory...")
    # Search more broadly
    current = Path.cwd()
    for parent in [current, current.parent, current.parent.parent]:
        data_path = parent / "data" 
        if data_path.exists():
            pdf_files = list(data_path.rglob("*.pdf"))
            print(f"Found data at {data_path} with {len(pdf_files)} PDFs")
            if pdf_files:
                DATA_DIR = data_path
                break

print(f"\nProject root: {PROJECT_ROOT}")
print(f"Data directory: {DATA_DIR}")

# Load environment file
env_files = [PROJECT_ROOT / ".env1", PROJECT_ROOT.parent / ".env1"]
for env_file in env_files:
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
        print(f"Loaded environment from: {env_file}")
        break

EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
EMBEDDINGS_DIR.mkdir(exist_ok=True)

class LocalEmbeddingsBuilder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None
        
    def extract_text_from_pdf(self, pdf_path: Path) -> List[str]:
        """Extract text from PDF and split into chunks"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            
            # Split into chunks (simple approach)
            chunks = []
            sentences = text.split('. ')
            
            chunk = ""
            for sentence in sentences:
                if len(chunk + sentence) < 500:  # Keep chunks under 500 chars
                    chunk += sentence + ". "
                else:
                    if chunk.strip():
                        chunks.append(chunk.strip())
                    chunk = sentence + ". "
            
            if chunk.strip():
                chunks.append(chunk.strip())
                
            print(f"Extracted {len(chunks)} chunks from {pdf_path.name}")
            return chunks
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return []
    
    def process_all_pdfs(self):
        """Process all PDFs in the data directory and subdirectories"""
        # Search recursively for all PDF files
        pdf_files = list(DATA_DIR.rglob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files across all subdirectories")
        
        if not pdf_files:
            print("No PDF files found. Checking subdirectories:")
            for subdir in DATA_DIR.iterdir():
                if subdir.is_dir():
                    pdf_count = len(list(subdir.rglob("*.pdf")))
                    print(f"  {subdir.name}: {pdf_count} PDFs")
            return
        
        # Show which files will be processed
        print("PDF files to process:")
        for pdf in pdf_files[:10]:  # Show first 10
            print(f"  - {pdf.relative_to(DATA_DIR)}")
        if len(pdf_files) > 10:
            print(f"  ... and {len(pdf_files) - 10} more files")
        
        all_chunks = []
        chunk_metadata = []
        
        for pdf_file in pdf_files:
            print(f"Processing {pdf_file.name}...")
            chunks = self.extract_text_from_pdf(pdf_file)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'source': str(pdf_file.relative_to(DATA_DIR)),
                    'chunk_id': i,
                    'length': len(chunk),
                    'category': pdf_file.parent.name
                })
        
        self.documents = all_chunks
        self.metadata = chunk_metadata
        print(f"Total chunks extracted: {len(all_chunks)}")
        
        # Show breakdown by category
        categories = {}
        for meta in chunk_metadata:
            cat = meta['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        print("Chunks by category:")
        for cat, count in categories.items():
            print(f"  {cat}: {count} chunks")
        
    def generate_embeddings(self):
        """Generate embeddings for all document chunks"""
        print("Generating embeddings...")
        print("This may take a few minutes...")
        
        # Generate embeddings in batches to manage memory
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(self.documents), batch_size):
            batch = self.documents[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=True)
            all_embeddings.append(batch_embeddings)
            print(f"Processed batch {i//batch_size + 1}/{(len(self.documents) + batch_size - 1)//batch_size}")
        
        self.embeddings = np.vstack(all_embeddings)
        print(f"Generated embeddings shape: {self.embeddings.shape}")
    
    def save_to_files(self):
        """Save documents and embeddings to files"""
        # Save documents and metadata as JSON
        documents_data = {
            'documents': self.documents,
            'metadata': self.metadata,
            'total_documents': len(self.documents)
        }
        
        with open(EMBEDDINGS_DIR / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(documents_data, f, ensure_ascii=False, indent=2)
        
        # Save embeddings as numpy array
        np.save(EMBEDDINGS_DIR / "embeddings.npy", self.embeddings)
        
        # Save model info
        model_info = {
            'model_name': 'all-MiniLM-L6-v2',
            'embedding_dimension': self.embeddings.shape[1],
            'total_chunks': len(self.documents)
        }
        
        with open(EMBEDDINGS_DIR / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Saved to {EMBEDDINGS_DIR}:")
        print(f"  - documents.json: {len(self.documents)} chunks")
        print(f"  - embeddings.npy: {self.embeddings.shape}")
        print(f"  - model_info.json: metadata")

def main():
    print("Building embeddings locally...")
    
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        return
    
    builder = LocalEmbeddingsBuilder()
    
    # Process all PDFs
    builder.process_all_pdfs()
    
    if not builder.documents:
        print("No documents found to process")
        return
    
    # Generate embeddings
    builder.generate_embeddings()
    
    # Save everything
    builder.save_to_files()
    
    print("Embeddings built successfully!")
    print("\nNext steps:")
    print("1. Commit the embeddings/ folder to git")
    print("2. Deploy the updated app to Railway")

if __name__ == "__main__":
    main()