from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

print("="*80)
print("🔍 OCR OUTPUT DIAGNOSTIC")
print("="*80)

# Check raw files
print("\n📁 RAW FILES (what should have been OCR'd):")
raw_stgt272 = list(RAW_DIR.rglob("*272*.pdf"))
raw_nordbahnhof = list(RAW_DIR.rglob("*[Nn]ordbahnhof*.pdf"))

print(f"\n   Files with '272': {len(raw_stgt272)}")
for f in raw_stgt272:
    print(f"      - {f.name}")

print(f"\n   Files with 'Nordbahnhof': {len(raw_nordbahnhof)}")
for f in raw_nordbahnhof:
    print(f"      - {f.name}")

# Check processed files
print("\n" + "="*80)
print("📄 PROCESSED FILES (OCR output):")

if PROCESSED_DIR.exists():
    # Find all OCR JSON files
    ocr_files = list(PROCESSED_DIR.rglob("*_ocr.json"))
    
    print(f"\n   Total OCR output files: {len(ocr_files)}")
    
    # Look for Nordbahnhof/272 related
    stgt272_ocr = [f for f in ocr_files if "272" in f.name.lower()]
    nordbahnhof_ocr = [f for f in ocr_files if "nordbahnhof" in f.name.lower()]
    
    print(f"\n   Files with '272' in name: {len(stgt272_ocr)}")
    for f in stgt272_ocr:
        print(f"      - {f.parent.name}/{f.name}")
    
    print(f"\n   Files with 'Nordbahnhof' in name: {len(nordbahnhof_ocr)}")
    for f in nordbahnhof_ocr:
        print(f"      - {f.parent.name}/{f.name}")
    
    # Check Stuttgart_Nord directory specifically
    stuttgart_nord = PROCESSED_DIR / "Stuttgart_Nord"
    if stuttgart_nord.exists():
        print(f"\n📂 Stuttgart_Nord directory contents:")
        subdirs = [d for d in stuttgart_nord.iterdir() if d.is_dir()]
        print(f"   {len(subdirs)} subdirectories:")
        for d in subdirs:
            ocr_files_in_dir = list(d.glob("*_ocr.json"))
            if ocr_files_in_dir:
                ocr_file = ocr_files_in_dir[0]
                # Read to get source
                with open(ocr_file, 'r') as f:
                    data = json.load(f)
                    source = data.get('source_file', 'Unknown')
                    pages = data.get('total_pages', 0)
                print(f"      {d.name}: {source} ({pages} pages)")
else:
    print("   ❌ Processed directory doesn't exist!")

# Check embeddings for comparison
print("\n" + "="*80)
print("💾 EMBEDDINGS CHECK:")

embeddings_file = PROJECT_ROOT / "embeddings" / "documents.json"
if embeddings_file.exists():
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    
    # Find unique sources
    sources = set()
    for doc in docs:
        source = doc.get('source', '')
        if 'nordbahnhof' in source.lower() or '272' in source:
            # Extract just the file name part
            source_file = source.split(',')[0] if ',' in source else source
            sources.add(source_file)
    
    print(f"\n   Nordbahnhof-related sources in embeddings:")
    for source in sorted(sources):
        count = sum(1 for doc in docs if source in doc.get('source', ''))
        print(f"      {source}: {count} chunks")

print("\n" + "="*80)