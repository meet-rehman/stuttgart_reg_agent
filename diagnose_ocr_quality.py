#!/usr/bin/env python3
"""Check if OCR text and embeddings contain Stgt data"""

import json
from pathlib import Path

print("="*70)
print("🔍 DIAGNOSING WHY QUERIES FAIL")
print("="*70)
print()

# Check OCR files
print("1️⃣ Checking OCR output files...")
print("-"*70)
ocr_dir = Path("data/processed/ocr_outputs/Landuse Plans")
ocr_files = list(ocr_dir.glob("*_ocr.txt"))

for ocr_file in ocr_files:
    with open(ocr_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Check for key terms
    has_stgt = 'stgt' in text.lower() or '272' in text or '283' in text
    has_grz = 'grz' in text.lower() or 'grundfläche' in text.lower()
    has_height = 'höhe' in text.lower() or 'hba' in text.lower()
    
    status = "✅" if (has_stgt or has_grz or has_height) else "⚠️"
    print(f"{status} {ocr_file.name}: {len(text)} chars")
    
    if has_stgt:
        print(f"   - Contains: Stgt/272/283")
    if has_grz:
        print(f"   - Contains: GRZ/Grundfläche")
    if has_height:
        print(f"   - Contains: Höhe/HbA")
    
    if not (has_stgt or has_grz or has_height):
        print(f"   ⚠️ Preview: {text[:200]}")

print()

# Check embeddings
print("2️⃣ Checking embeddings file...")
print("-"*70)
docs_file = Path("embeddings/documents.json")

with open(docs_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

if isinstance(data, dict):
    docs = data['documents']
else:
    docs = data

print(f"Total documents: {len(docs)}")
print()

# Search last 200 docs (the new ones)
print("3️⃣ Searching last 200 documents for Stgt content...")
print("-"*70)

matches = []
for i, doc in enumerate(docs[-200:], len(docs)-200):
    # Handle both dict and string formats
    if isinstance(doc, dict):
        text = doc.get('content', '') or doc.get('text', '') or str(doc)
    else:
        text = str(doc)
    
    text_lower = text.lower()
    if any(term in text_lower for term in ['stgt', '272', '283', 'nordbahnhof', 'grz', 'grundfläche']):
        matches.append((i, text[:150]))

if matches:
    print(f"✅ Found {len(matches)} matches in last 200 docs")
    print("\nSample matches:")
    for idx, content in matches[:3]:
        print(f"  [{idx}] {content}...")
else:
    print("❌ NO MATCHES in last 200 documents!")
    print("\nThis means OCR text is not being embedded properly.")
    print("\nLast 3 documents in embeddings:")
    for i, doc in enumerate(docs[-3:], len(docs)-3):
        if isinstance(doc, dict):
            text = doc.get('content', '') or doc.get('text', '') or str(doc)
        else:
            text = str(doc)
        print(f"  [{i}] {text[:150]}...")

print()
print("="*70)
print("💡 DIAGNOSIS")
print("="*70)
print()

if not ocr_files:
    print("❌ No OCR files found - run quick_ocr_stgt.py")
elif not matches:
    print("❌ OCR exists but not in embeddings!")
    print("\nPossible issues:")
    print("  1. OCR text is mostly symbols/noise (bad OCR quality)")
    print("  2. build_embeddings_from_ocr.py didn't read OCR files correctly")
    print("  3. Chunks were too small and filtered out")
    print("\n🔧 Solution:")
    print("  - Check OCR quality: cat data/processed/ocr_outputs/Landuse Plans/*_ocr.txt | more")
    print("  - Re-run with verbose: python build_embeddings_from_ocr.py")
else:
    print("✅ Data is in embeddings, but query matching is poor")
    print("\nPossible issues:")
    print("  1. Query wording doesn't match OCR text")
    print("  2. Need better chunk splitting")
    print("  3. Need metadata/keywords for better search")

print()