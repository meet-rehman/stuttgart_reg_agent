import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
import json
from pathlib import Path
from datetime import datetime

# Increase PIL image size limit (for large PDFs)
Image.MAX_IMAGE_PIXELS = 500000000  # 500 megapixels

# Configure paths
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r'C:\Program Files\poppler\poppler-25.07.0\Library\bin'

PROJECT_ROOT = Path(__file__).parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "Stuttgart_Nord"

def ocr_stgt272(dpi=300):  # Reduced from 400 to 300
    """OCR the stgt-272 PDF specifically"""
    
    # Find the file
    pdf_files = list(RAW_DIR.rglob("*272*.pdf"))
    
    if not pdf_files:
        print("❌ stgt-272-bebauungsplan-2024-02.pdf not found!")
        return None
    
    pdf_path = pdf_files[0]
    output_dir = PROCESSED_DIR / "stgt_272_bebauungsplan"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("📄 OCR for Stgt 272 Bebauungsplan")
    print("="*80)
    print(f"Input: {pdf_path.name}")
    print(f"Output: {output_dir}")
    print(f"DPI: {dpi}")
    print(f"Max image pixels: {Image.MAX_IMAGE_PIXELS:,}")
    print("="*80)
    
    # Convert PDF to images
    print("\n🔄 Converting PDF to images (this may take a few minutes)...")
    
    try:
        images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            fmt='png',
            poppler_path=POPPLER_PATH,
            thread_count=2  # Use multiple threads for faster processing
        )
        print(f"✅ Converted to {len(images)} page(s)")
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        print("\n💡 Trying with lower DPI (200)...")
        
        try:
            images = convert_from_path(
                str(pdf_path),
                dpi=200,  # Fallback to lower DPI
                fmt='png',
                poppler_path=POPPLER_PATH,
                thread_count=2
            )
            print(f"✅ Converted to {len(images)} page(s) at 200 DPI")
        except Exception as e2:
            print(f"❌ Still failed: {e2}")
            return None
    
    # OCR each page
    all_text = []
    ocr_results = {
        'source_file': 'stgt-272-bebauungsplan-2024-02.pdf',
        'source_path': str(pdf_path),
        'total_pages': len(images),
        'pages': [],
        'processed_at': datetime.now().isoformat()
    }
    
    for i, image in enumerate(images, 1):
        print(f"\n📖 Processing page {i}/{len(images)}...", end=" ")
        
        # Save image (optional, for debugging)
        image_path = output_dir / f"page_{i}.png"
        image.save(image_path)
        
        # OCR with German language
        custom_config = r'--oem 3 --psm 3 -l deu'
        
        try:
            text = pytesseract.image_to_string(image, config=custom_config)
            char_count = len(text.strip())
            
            if char_count == 0:
                print(f"⚠️ Empty (trying alternative OCR settings)...")
                # Try with different PSM mode
                custom_config = r'--oem 3 --psm 6 -l deu'
                text = pytesseract.image_to_string(image, config=custom_config)
                char_count = len(text.strip())
            
            print(f"✅ {char_count} chars")
            
            all_text.append(text)
            ocr_results['pages'].append({
                'page': i,
                'text': text,
                'char_count': char_count
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            all_text.append("")
            ocr_results['pages'].append({
                'page': i,
                'text': "",
                'char_count': 0,
                'error': str(e)
            })
    
    # Save combined text
    combined_file = output_dir / "stgt_272_combined.txt"
    with open(combined_file, 'w', encoding='utf-8') as f:
        for i, text in enumerate(all_text, 1):
            f.write(f"\n{'='*80}\nPAGE {i}\n{'='*80}\n\n")
            f.write(text)
    
    # Save JSON
    json_file = output_dir / "stgt_272_ocr.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(ocr_results, f, ensure_ascii=False, indent=2)
    
    # Summary
    total_chars = sum(p['char_count'] for p in ocr_results['pages'])
    empty_pages = sum(1 for p in ocr_results['pages'] if p['char_count'] == 0)
    
    print("\n" + "="*80)
    print("✅ OCR COMPLETE")
    print("="*80)
    print(f"📊 Total pages: {len(images)}")
    print(f"📊 Total characters: {total_chars:,}")
    print(f"📊 Empty pages: {empty_pages}")
    print(f"📊 Avg chars/page: {total_chars // len(images) if len(images) > 0 else 0:,}")
    print(f"📁 Output: {output_dir}")
    print("="*80)
    
    # Show sample from first page with text
    print("\n📝 Sample from first page with content:")
    for page in ocr_results['pages']:
        if page['char_count'] > 100:
            sample = page['text'][:300].replace('\n', ' ')
            print(f"   Page {page['page']}: {sample}...")
            break
    
    # Check quality
    if total_chars < 1000:
        print("\n⚠️ WARNING: Very little text extracted!")
        print("   This might be:")
        print("   - A scanned image PDF (low quality scan)")
        print("   - An architectural drawing (mostly graphics)")
        print("   - OCR quality issue")
        print("\n💡 Check the saved images in the output folder")
    elif empty_pages > len(images) * 0.3:
        print(f"\n⚠️ WARNING: {empty_pages} empty pages (>{30}% of total)")
        print("   Some pages might be graphics/diagrams without text")
    
    return ocr_results

if __name__ == "__main__":
    print("\n🏗️ STGT 272 BEBAUUNGSPLAN OCR")
    print("This will OCR the stgt-272-bebauungsplan-2024-02.pdf file")
    print("Using DPI 300 with automatic fallback to 200 if needed\n")
    
    result = ocr_stgt272(dpi=300)
    
    if result:
        print("\n✨ Next steps:")
        print("1. Check the OCR output quality:")
        print("   Get-Content .\\data\\processed\\Stuttgart_Nord\\stgt_272_bebauungsplan\\stgt_272_combined.txt -Head 100")
        print("\n2. If quality is good, add to embeddings:")
        print("   python add_stgt272_to_embeddings.py")