#!/usr/bin/env python3
"""
Migration Script: Convert Local Embeddings to Railway-Optimized Format
Converts your existing documents.json from local format to lightweight format
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_to_railway_format(
    old_embeddings_dir: Path,
    output_file: Path
):
    """
    Convert existing documents.json to lightweight format
    
    Args:
        old_embeddings_dir: Your current embeddings/ directory
        output_file: Output path for lightweight documents
    """
    
    logger.info("="*70)
    logger.info("MIGRATION: Local Embeddings → Railway-Optimized")
    logger.info("="*70)
    
    # Load existing documents
    docs_path = old_embeddings_dir / "documents.json"
    if not docs_path.exists():
        raise FileNotFoundError(f"Documents not found: {docs_path}")
    
    logger.info(f"📂 Loading from: {docs_path}")
    with open(docs_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both formats
    if isinstance(data, dict):
        documents = data.get('documents', [])
    else:
        documents = data
    
    logger.info(f"📚 Found {len(documents)} documents")
    
    # Check current size
    original_size = docs_path.stat().st_size / (1024 * 1024)  # MB
    logger.info(f"📏 Original size: {original_size:.2f} MB")
    
    # Check if embeddings.npy exists
    emb_path = old_embeddings_dir / "embeddings.npy"
    if emb_path.exists():
        emb_size = emb_path.stat().st_size / (1024 * 1024)
        logger.info(f"📏 Embeddings.npy size: {emb_size:.2f} MB")
        logger.info(f"💰 Total Git storage: {original_size + emb_size:.2f} MB")
    
    # Create lightweight version (documents only, no embeddings)
    # The documents already have all necessary fields
    lightweight = {
        'documents': documents,
        'migration_info': {
            'original_count': len(documents),
            'migrated_from': 'local_embeddings',
            'format': 'railway_optimized',
            'note': 'Embeddings will be generated via OpenAI API on-demand'
        }
    }
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save lightweight version
    logger.info(f"💾 Saving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(lightweight, f, indent=2, ensure_ascii=False)
    
    # Check new size
    new_size = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"📏 New size: {new_size:.2f} MB")
    logger.info(f"💾 Size savings: {original_size - new_size:.2f} MB")
    
    # Calculate savings
    if emb_path.exists():
        total_saved = (original_size + emb_size) - new_size
        logger.info(f"✅ Total Git storage saved: {total_saved:.2f} MB ({total_saved/(original_size + emb_size)*100:.1f}%)")
    
    logger.info("="*70)
    logger.info("✅ MIGRATION COMPLETE")
    logger.info("="*70)
    
    logger.info("\nNext steps:")
    logger.info("1. Review the new file at: " + str(output_file))
    logger.info("2. Test with railway_optimized_rag.py")
    logger.info("3. Update your code to use RailwayOptimizedRAG")
    logger.info("4. Remove old embeddings.npy from Git (too large)")
    logger.info("5. Deploy to Railway! 🚂")


def create_gitignore_entries():
    """
    Generate .gitignore entries for Railway deployment
    """
    entries = """
# ============================================================================
# RAILWAY DEPLOYMENT - Ignore large files
# ============================================================================

# Old local embeddings (too large for Git/Railway)
embeddings/embeddings.npy
embeddings/*.npy

# Large landuse plan images
data/raw/Landuse Plans/
data/raw/landuse_plans/
*.png
*.jpg
*.jpeg
*.tif
*.tiff

# Model cache
models/
.cache/
*.model

# Keep the lightweight documents
!data/documents_lightweight.json
"""
    return entries


if __name__ == "__main__":
    import sys
    
    # Default paths
    project_root = Path(__file__).parent
    old_embeddings = project_root / "embeddings"
    output_dir = project_root / "data"
    output_file = output_dir / "documents_lightweight.json"
    
    # Allow custom paths
    if len(sys.argv) > 1:
        old_embeddings = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_file = Path(sys.argv[2])
    
    print("\n" + "="*70)
    print("🔄 RAILWAY MIGRATION TOOL")
    print("="*70)
    print(f"\nInput:  {old_embeddings}/documents.json")
    print(f"Output: {output_file}")
    print("\nThis will create a lightweight version for Railway deployment.")
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    try:
        # Run migration
        migrate_to_railway_format(old_embeddings, output_file)
        
        # Generate .gitignore entries
        print("\n📝 Recommended .gitignore entries:")
        print(create_gitignore_entries())
        
        gitignore_path = project_root / ".gitignore"
        if gitignore_path.exists():
            print(f"\n💡 Add these entries to your existing .gitignore at: {gitignore_path}")
        else:
            print(f"\n💡 Create .gitignore at: {gitignore_path}")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)