#!/usr/bin/env python3
"""
Check what files you have in your data directories
"""

from pathlib import Path

print("="*70)
print("📁 CHECKING YOUR DATA DIRECTORIES")
print("="*70)
print()

# Check directories
dirs_to_check = [
    "data/raw/Regulations",
    "data/raw/Landuse Plans",
    "data/processed/ocr_outputs",
    "embeddings"
]

for dir_path in dirs_to_check:
    path = Path(dir_path)
    
    print(f"\n📂 {dir_path}")
    print("-"*70)
    
    if not path.exists():
        print("   ❌ Directory does not exist")
        continue
    
    # List files
    files = list(path.rglob("*.*"))
    
    if not files:
        print("   ⚠️  Directory is empty")
        continue
    
    # Group by extension
    by_ext = {}
    for f in files:
        ext = f.suffix.lower()
        if ext not in by_ext:
            by_ext[ext] = []
        by_ext[ext].append(f)
    
    print(f"   ✅ Found {len(files)} files:")
    for ext, file_list in sorted(by_ext.items()):
        print(f"      {ext}: {len(file_list)} files")
    
    # Show specific files for important directories
    if "Landuse Plans" in dir_path:
        print("\n   📄 Landuse Plan files:")
        pdf_files = [f for f in files if f.suffix.lower() == '.pdf']
        for pdf in sorted(pdf_files)[:10]:
            print(f"      - {pdf.name}")
        if len(pdf_files) > 10:
            print(f"      ... and {len(pdf_files) - 10} more")
    
    if "ocr_outputs" in dir_path:
        print("\n   📄 OCR output files:")
        txt_files = [f for f in files if f.suffix.lower() == '.txt']
        for txt in sorted(txt_files)[:10]:
            print(f"      - {txt.name}")
        if len(txt_files) > 10:
            print(f"      ... and {len(txt_files) - 10} more")

print("\n" + "="*70)
print("🔍 LOOKING FOR STGT DOCUMENTS")
print("="*70)
print()

# Search for Stgt files
all_files = []
for dir_path in dirs_to_check:
    path = Path(dir_path)
    if path.exists():
        all_files.extend(path.rglob("*.*"))

stgt_files = [f for f in all_files if 'stgt' in f.name.lower() or '272' in f.name or '283' in f.name]
nordbahnhof_files = [f for f in all_files if 'nordbahnhof' in f.name.lower() or 'friedhof' in f.name.lower()]
killesberg_files = [f for f in all_files if 'killesberg' in f.name.lower() or 'maybach' in f.name.lower()]

print(f"Stgt 272/283 files:      {len(stgt_files)}")
if stgt_files:
    for f in stgt_files[:5]:
        print(f"   - {f.relative_to(f.parents[2])}")

print(f"\nNordbahnhof files:       {len(nordbahnhof_files)}")
if nordbahnhof_files:
    for f in nordbahnhof_files[:5]:
        print(f"   - {f.relative_to(f.parents[2])}")

print(f"\nKillesberg files:        {len(killesberg_files)}")
if killesberg_files:
    for f in killesberg_files[:5]:
        print(f"   - {f.relative_to(f.parents[2])}")

print("\n" + "="*70)
print("💡 NEXT STEPS")
print("="*70)
print()

if not stgt_files and not nordbahnhof_files:
    print("❌ ISSUE: You don't have Stgt 272 or Nordbahnhof documents!")
    print()
    print("📥 You need to add these documents:")
    print("   1. Stgt 272 (Nordbahnhof/Friedhofstrasse) Bebauungsplan PDF")
    print("   2. Stgt 283 (Killesberg/Maybachstraße) Bebauungsplan PDF")
    print()
    print("📁 Where to add them:")
    print("   data/raw/Landuse Plans/")
    print()
    print("Then run:")
    print("   1. python batch_ocr.py              # Extract text")
    print("   2. python build_embeddings_from_ocr.py  # Build embeddings")
    print("   3. python test_rag_local.py         # Test again")
    
elif stgt_files and not any('ocr_outputs' in str(f) for f in stgt_files):
    print("⚠️  You have the PDFs but haven't run OCR yet!")
    print()
    print("Run these commands:")
    print("   1. python batch_ocr.py              # Extract text from PDFs")
    print("   2. python build_embeddings_from_ocr.py  # Build embeddings")
    print("   3. python test_rag_local.py         # Test again")
    
elif any('ocr_outputs' in str(f) for f in stgt_files):
    print("✅ You have OCR outputs for Stgt documents!")
    print()
    print("If embeddings don't have them, rebuild embeddings:")
    print("   python build_embeddings_from_ocr.py")
    print("   python test_rag_local.py")
    
else:
    print("✅ Looks good! If tests still fail, check embedding build process.")

print()