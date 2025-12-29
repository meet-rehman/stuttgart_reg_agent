#!/usr/bin/env python3
"""
Simple Batch OCR for Landuse Plans
Extracts text from all PDFs in data/raw/Landuse Plans
"""

import os
from pathlib import Path
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import time

# Configure Tesseract path (Windows)
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    print(f"✅ Tesseract found at: {tesseract_path}\n")
else:
    print(f"❌ Tesseract not found at: {tesseract_path}")
    print("   Please install from: https://github.com/UB-Mannheim/tesseract/wiki")
    print("   Or update the 'tesseract_path' variable in this script.\n")
    import sys
    sys.exit(1)

print("="*70)
print("🔍 BATCH OCR - LANDUSE PLANS")
print("="*70)
print()

# Setup directories
landuse_dir = Path("data/raw/Landuse Plans")
output_dir = Path("data/processed/ocr_outputs/Landuse Plans")
output_dir.mkdir(parents=True, exist_ok=True)

# Find all PDFs
pdf_files = list(landuse_dir.rglob("*.pdf"))
pdf_files = [f for f in pdf_files if 'images' not in str(f)]

print(f"📂 Input: {landuse_dir}")
print(f"📂 Output: {output_dir}")
print(f"📄 Found {len(pdf_files)} PDFs to process")
print()

if not pdf_files:
    print("❌ No PDFs found!")
    import sys
    sys.exit(1)

# Process each PDF
success_count = 0
error_count = 0
start_time = time.time()

for i, pdf_path in enumerate(pdf_files, 1):
    print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
    print("-"*70)
    
    # Create output filename
    output_filename = pdf_path.stem + "_ocr.txt"
    output_path = output_dir / output_filename
    
    # Skip if already processed
    if output_path.exists():
        print(f"   ⏭️  Already processed (skipping)")
        success_count += 1
        continue
    
    try:
        # Convert PDF to images
        print(f"   📄 Converting PDF to images...")
        images = convert_from_path(
            str(pdf_path),
            dpi=200,  # Lower DPI for speed (increase to 300 for better quality)
            fmt='png',
            thread_count=2
        )
        
        print(f"   📸 Extracted {len(images)} pages")
        
        # OCR each page
        all_text = []
        
        for page_num, image in enumerate(images, 1):
            print(f"   🔍 OCR page {page_num}/{len(images)}...", end=" ", flush=True)
            
            # Run OCR (German language)
            text = pytesseract.image_to_string(
                image,
                lang='deu',  # German
                config='--psm 1'  # Automatic page segmentation with OSD
            )
            
            print(f"✓ ({len(text)} chars)")
            
            # Add page header
            all_text.append(f"\n{'='*70}\n")
            all_text.append(f"PAGE {page_num}\n")
            all_text.append(f"{'='*70}\n\n")
            all_text.append(text)
        
        # Save combined text
        full_text = "".join(all_text)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        print(f"   ✅ Saved: {output_filename}")
        print(f"   📊 Total: {len(full_text)} characters")
        
        success_count += 1
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        error_count += 1
        
        # Save error log
        error_log = output_dir / f"{pdf_path.stem}_ERROR.txt"
        with open(error_log, 'w') as f:
            f.write(f"Error processing {pdf_path.name}:\n{str(e)}")

# Summary
elapsed = time.time() - start_time
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)

print("\n" + "="*70)
print("📊 OCR SUMMARY")
print("="*70)
print(f"Total PDFs:       {len(pdf_files)}")
print(f"Successful:       {success_count} ✅")
print(f"Errors:           {error_count} ❌")
print(f"Time elapsed:     {minutes}m {seconds}s")
print()

if success_count > 0:
    print("✅ OCR complete! Text files saved to:")
    print(f"   {output_dir}")
    print()
    print("Next step: Build embeddings")
    print("   python build_embeddings_from_ocr.py")
else:
    print("❌ No files were processed successfully")
    print("\nCheck:")
    print("   - Tesseract is installed")
    print("   - PDFs are readable")
    print("   - Enough disk space")

print()