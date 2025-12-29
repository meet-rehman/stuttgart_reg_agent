#!/usr/bin/env python3
"""
Pre-download sentence-transformers model during Railway build
This reduces startup time and memory usage
"""
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model():
    """Download the sentence-transformers model"""
    try:
        logger.info("Downloading sentence-transformers model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model downloaded successfully!")
        
        # Test the model
        test_embedding = model.encode(["test"])
        logger.info(f"Model test successful! Embedding shape: {test_embedding.shape}")
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise

if __name__ == "__main__":
    download_model()
