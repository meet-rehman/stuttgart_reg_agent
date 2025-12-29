import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
import json
import re

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Auto-detect Poppler
def find_poppler():
    """Find Poppler installation automatically"""
    possible_paths = [
        r'C:\Program Files\poppler\poppler-25.07.0\Library\bin',
        r'C:\poppler\poppler-24.08.0\Library\bin',
        r'C:\Program Files\poppler\Library\bin',
        r'C:\poppler\Library\bin',
    ]
    
    for path in possible_paths:
        if os.path.exists(os.path.join(path, 'pdftoppm.exe')):
            print(f"✅ Found Poppler at: {path}")
            return path
    
    # Search in common directories
    for base_dir in [r'C:\Program Files\poppler', r'C:\poppler']:
        if os.path.exists(base_dir):
            for root, dirs, files in os.walk(base_dir):
                if 'pdftoppm.exe' in files:
                    print(f"✅ Found Poppler at: {root}")
                    return root
    
    return None

POPPLER_PATH = find_poppler()

if not POPPLER_PATH:
    print("❌ Poppler not found!")
    print("Please install Poppler or check the installation")
    exit(1)

def ocr_bebauungsplan(pdf_path, output_dir="ocr_output", dpi=300):
    """Extract text from a Bebauungsplan PDF using Tesseract OCR"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"📄 Processing: {pdf_path}")
    print(f"📊 DPI: {dpi}")
    print(f"🔧 Poppler: {POPPLER_PATH}")
    print(f"🔧 Tesseract: {pytesseract.pytesseract.tesseract_cmd}")
    print(f"{'='*60}\n")
    print("🔄 Converting PDF to images (this may take a minute)...")
    
    try:
        images = convert_from_path(
            pdf_path, 
            dpi=dpi,
            fmt='png',
            poppler_path=POPPLER_PATH
        )
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return None
    except Exception as e:
        print(f"❌ Error converting PDF: {e}")
        return None
    
    print(f"✅ Converted to {len(images)} page(s)")
    
    all_text = []
    
    for i, image in enumerate(images, 1):
        print(f"\n📖 Processing page {i}/{len(images)}...")
        
        # Save image
        image_path = os.path.join(output_dir, f"page_{i}.png")
        image.save(image_path)
        print(f"   💾 Saved image: page_{i}.png ({image.size[0]}x{image.size[1]})")
        
        # OCR with German
        custom_config = r'--oem 3 --psm 3 -l deu'
        
        try:
            print(f"   🔍 Running OCR...")
            text = pytesseract.image_to_string(image, config=custom_config)
            char_count = len(text.strip())
            word_count = len(text.split())
            print(f"   ✅ Extracted {char_count} characters, {word_count} words")
        except Exception as e:
            print(f"   ❌ OCR Error: {e}")
            text = ""
        
        # Save text
        text_file = os.path.join(output_dir, f"page_{i}.txt")
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        all_text.append({
            'page': i, 
            'text': text, 
            'char_count': len(text.strip()),
            'word_count': len(text.split())
        })
    
    # Save combined
    print(f"\n📝 Saving combined output...")
    combined_file = os.path.join(output_dir, "combined_text.txt")
    with open(combined_file, 'w', encoding='utf-8') as f:
        for page_data in all_text:
            f.write(f"\n{'='*60}\nPAGE {page_data['page']}\n{'='*60}\n\n")
            f.write(page_data['text'])
    
    # Save JSON
    json_file = os.path.join(output_dir, "ocr_output.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_text, f, ensure_ascii=False, indent=2)
    
    # Summary
    total_chars = sum(p['char_count'] for p in all_text)
    total_words = sum(p['word_count'] for p in all_text)
    
    print(f"\n{'='*60}")
    print(f"✅ OCR COMPLETE!")
    print(f"{'='*60}")
    print(f"📊 Total pages: {len(all_text)}")
    print(f"📊 Total characters: {total_chars:,}")
    print(f"📊 Total words: {total_words:,}")
    print(f"📁 Output directory: {os.path.abspath(output_dir)}")
    print(f"   Files created:")
    print(f"   - combined_text.txt (all text in one file)")
    print(f"   - ocr_output.json (structured JSON data)")
    print(f"   - page_X.txt (individual page text)")
    print(f"   - page_X.png (page images)")
    print(f"{'='*60}")
    
    return all_text

def extract_regulations(text):
    """Extract key regulations from OCR text"""
    regulations = {}
    print("\n🔍 Searching for regulations in extracted text...")
    
    # GRZ (Grundflächenzahl)
    grz = re.findall(r'GRZ[:\s=]+([0-9],[0-9]+|[0-9]\.[0-9]+)', text, re.IGNORECASE)
    if grz:
        regulations['GRZ'] = list(set(grz))
        print(f"   ✅ Found GRZ: {regulations['GRZ']}")
    
    # GFZ (Geschossflächenzahl)
    gfz = re.findall(r'GFZ[:\s=]+([0-9],[0-9]+|[0-9]\.[0-9]+)', text, re.IGNORECASE)
    if gfz:
        regulations['GFZ'] = list(set(gfz))
        print(f"   ✅ Found GFZ: {regulations['GFZ']}")
    
    # Height (Höhe, Traufhöhe, Firsthöhe)
    height = re.findall(r'(?:Höhe|Traufhöhe|Firsthöhe)[:\s]+(?:max\.?\s*)?([0-9]+[,.]?[0-9]*)\s*m', text, re.IGNORECASE)
    if height:
        regulations['height_m'] = list(set(height))
        print(f"   ✅ Found heights: {regulations['height_m']} m")
    
    # Stories (Vollgeschosse)
    stories = re.findall(r'([IVX]+)\s+Vollgeschoss', text, re.IGNORECASE)
    if stories:
        regulations['stories'] = list(set(stories))
        print(f"   ✅ Found stories: {regulations['stories']}")
    
    # Zone types
    zones = re.findall(r'\b(WA|MI|GE|GI|MD|WR|MK)\s*[0-9]*\b', text)
    if zones:
        regulations['zones'] = list(set(zones))
        print(f"   ✅ Found zones: {regulations['zones']}")
    
    if not regulations:
        print("   ⚠️  No regulations automatically detected")
        print("   💡 Check the text files manually - OCR may have formatting issues")
    
    return regulations

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏗️  BEBAUUNGSPLAN OCR TOOL")
    print("="*60)
    
    pdf_path = "stgt-272-bebauungsplan-2024-02.pdf"
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"\n❌ PDF not found: {pdf_path}")
        print(f"📁 Current directory: {os.getcwd()}")
        print(f"📋 Files in directory:")
        for f in os.listdir('.'):
            if f.endswith('.pdf'):
                print(f"   - {f}")
        exit(1)
    
    # Get file size
    file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
    print(f"\n📄 Found PDF: {pdf_path} ({file_size:.1f} MB)")
    
    # Run OCR
    print("\n⏳ Starting OCR process...")
    results = ocr_bebauungsplan(pdf_path, output_dir="stgt_272_ocr", dpi=300)
    
    if results:
        # Extract regulations
        combined_text = " ".join([p['text'] for p in results])
        regulations = extract_regulations(combined_text)
        
        # Save regulations
        if regulations:
            reg_file = "stgt_272_ocr/regulations.json"
            with open(reg_file, 'w', encoding='utf-8') as f:
                json.dump(regulations, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Regulations saved to: {reg_file}")
        
        print("\n✨ Process complete! Check the stgt_272_ocr/ folder for results.")
    else:
        print("\n❌ OCR process failed. Check error messages above.")