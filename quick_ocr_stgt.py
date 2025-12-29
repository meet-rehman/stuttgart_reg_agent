#!/usr/bin/env python3
"""
Quick OCR - Process only Stgt 272/283 files first
For quick testing before processing all PDFs
"""

import os
from pathlib import Path
from pdf2image import convert_from_path
import pytesseract
import time

# Configure Tesseract path (Windows)
# Update this path if Tesseract is installed elsewhere
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    print(f"✅ Tesseract found at: {tesseract_path}\n")
else:
    print(f"❌ Tesseract not found at: {tesseract_path}")
    print("   Please install from: https://github.com/UB-Mannheim/tesseract/wiki")
    print("   Or update the path in this script.\n")
    import sys
    sys.exit(1)

print("="*70)
print("⚡ QUICK OCR - STGT FILES ONLY")
print("="*70)
print()

# Setup
landuse_dir = Path("data/raw/Landuse Plans")
output_dir = Path("data/processed/ocr_outputs/Landuse Plans")
output_dir.mkdir(parents=True, exist_ok=True)

# Find only Stgt files
all_pdfs = list(landuse_dir.rglob("*.pdf"))
stgt_files = [f for f in all_pdfs if any(kw in f.name.lower() 
              for kw in ['stgt-272', 'stgt-283', 'stgt272', 'stgt283', 
                         'nordbahnhof', 'killesberg'])]

print(f"📄 Found {len(stgt_files)} Stgt/key files:")
for f in stgt_files:
    print(f"   - {f.name}")
print()

if not stgt_files:
    print("❌ No Stgt files found!")
    import sys
    sys.exit(1)

response = input(f"Process these {len(stgt_files)} files? (yes/no): ").strip().lower()
if response not in ['yes', 'y']:
    print("⏸️  Cancelled")
    import sys
    sys.exit(0)

print()
start_time = time.time()

for i, pdf_path in enumerate(stgt_files, 1):
    print(f"\n[{i}/{len(stgt_files)}] {pdf_path.name}")
    print("-"*70)
    
    output_path = output_dir / (pdf_path.stem + "_ocr.txt")
    
    if output_path.exists():
        print("   ⏭️  Already processed")
        continue
    
    try:
        images = convert_from_path(str(pdf_path), dpi=200)
        print(f"   📸 {len(images)} pages")
        
        all_text = []
        for page_num, image in enumerate(images, 1):
            print(f"   🔍 Page {page_num}/{len(images)}...", end=" ", flush=True)
            text = pytesseract.image_to_string(image, lang='deu', config='--psm 1')
            print(f"✓ ({len(text)} chars)")
            
            all_text.append(f"\n{'='*70}\nPAGE {page_num}\n{'='*70}\n\n{text}")
        
        full_text = "".join(all_text)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        print(f"   ✅ Saved ({len(full_text)} total chars)")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

elapsed = time.time() - start_time
print(f"\n✅ Quick OCR complete! ({int(elapsed//60)}m {int(elapsed%60)}s)")
print()
print("Next: python build_embeddings_from_ocr.py")