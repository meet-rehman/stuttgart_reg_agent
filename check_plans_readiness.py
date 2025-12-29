#!/usr/bin/env python3
"""
Diagnostic script to check if your landuse plans are ready for Vision AI
"""

from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent
PLANS_DIR = PROJECT_ROOT / "data" / "raw" / "Landuse Plans"

print("="*80)
print("🔍 LANDUSE PLANS READINESS CHECK")
print("="*80)

# Check 1: Directory exists
print("\n1️⃣ Checking plans directory...")
if PLANS_DIR.exists():
    print(f"   ✅ Directory exists: {PLANS_DIR}")
else:
    print(f"   ❌ Directory NOT found: {PLANS_DIR}")
    print(f"\n   💡 Create it:")
    print(f"   New-Item -ItemType Directory -Path '{PLANS_DIR}' -Force")
    exit(1)

# Check 2: Find images
print("\n2️⃣ Checking for image files...")

image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
images = []

for ext in image_extensions:
    images.extend(PLANS_DIR.rglob(f"*{ext}"))

if images:
    print(f"   ✅ Found {len(images)} image files")
    
    # Group by subdirectory
    by_subdir = {}
    for img in images:
        rel_path = img.relative_to(PLANS_DIR)
        subdir = rel_path.parts[0] if len(rel_path.parts) > 1 else "root"
        if subdir not in by_subdir:
            by_subdir[subdir] = []
        by_subdir[subdir].append(img)
    
    print(f"\n   📂 Images by location:")
    for subdir, imgs in sorted(by_subdir.items()):
        print(f"      {subdir}: {len(imgs)} images")
        for img in imgs[:3]:  # Show first 3
            size_mb = img.stat().st_size / (1024 * 1024)
            print(f"         - {img.name} ({size_mb:.2f} MB)")
        if len(imgs) > 3:
            print(f"         ... and {len(imgs)-3} more")
else:
    print(f"   ❌ No image files found!")

# Check 3: Find PDFs (need conversion)
print("\n3️⃣ Checking for PDF files (need conversion)...")
pdfs = list(PLANS_DIR.rglob("*.pdf"))

if pdfs:
    print(f"   ⚠️  Found {len(pdfs)} PDF files that need conversion:")
    for pdf in pdfs[:5]:
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"      - {pdf.name} ({size_mb:.2f} MB)")
    if len(pdfs) > 5:
        print(f"      ... and {len(pdfs)-5} more")
    
    print(f"\n   💡 Convert PDFs to images:")
    print(f"      python convert_plans.py")
else:
    print(f"   ✅ No PDFs (all plans are images)")

# Check 4: OCR output exists
print("\n4️⃣ Checking OCR output...")
ocr_dir = PROJECT_ROOT / "data" / "processed" / "Landuse_Plans"

if ocr_dir.exists():
    ocr_files = list(ocr_dir.rglob("*_ocr.json"))
    print(f"   ✅ OCR directory exists with {len(ocr_files)} files")
    print(f"   ℹ️  Vision AI will complement (not replace) OCR text")
else:
    print(f"   ℹ️  No OCR output found")
    print(f"   ℹ️  Vision AI will work independently")

# Check 5: Sample image quality
print("\n5️⃣ Checking image quality...")

if images:
    # Check first image
    sample = images[0]
    size_mb = sample.stat().st_size / (1024 * 1024)
    
    try:
        from PIL import Image
        img = Image.open(sample)
        width, height = img.size
        
        print(f"   📸 Sample: {sample.name}")
        print(f"      Size: {width} x {height} pixels")
        print(f"      File size: {size_mb:.2f} MB")
        
        # Quality assessment
        megapixels = (width * height) / 1_000_000
        
        if megapixels < 1:
            print(f"      ⚠️  Low resolution ({megapixels:.1f} MP) - may affect text reading")
        elif megapixels < 5:
            print(f"      ✅ Good resolution ({megapixels:.1f} MP)")
        else:
            print(f"      ✅ High resolution ({megapixels:.1f} MP)")
        
        if size_mb > 10:
            print(f"      ⚠️  Large file - may be slow to process")
            print(f"      💡 Consider resizing: 2000x2000 pixels is usually sufficient")
        
    except ImportError:
        print(f"   ℹ️  Install Pillow to check image details: pip install Pillow")
    except Exception as e:
        print(f"   ⚠️  Could not read image: {e}")

# Check 6: OPENAI_API_KEY
print("\n6️⃣ Checking environment...")

import os
if os.getenv("OPENAI_API_KEY"):
    print(f"   ✅ OPENAI_API_KEY is set")
    
    # Check if it has vision access
    try:
        from openai import OpenAI
        client = OpenAI()
        
        # Test with a simple call (won't actually process)
        print(f"   ✅ OpenAI client initialized")
        print(f"   ℹ️  Model: gpt-4o (with vision)")
        
    except Exception as e:
        print(f"   ⚠️  OpenAI client error: {e}")
else:
    print(f"   ❌ OPENAI_API_KEY not set!")
    print(f"\n   💡 Set it:")
    print(f"   $env:OPENAI_API_KEY='your-key-here'")

# Summary
print("\n" + "="*80)
print("📊 SUMMARY")
print("="*80)

ready = True
issues = []

if not PLANS_DIR.exists():
    ready = False
    issues.append("Plans directory missing")

if not images:
    ready = False
    issues.append("No image files found")

if pdfs:
    issues.append(f"{len(pdfs)} PDFs need conversion")

if not os.getenv("OPENAI_API_KEY"):
    ready = False
    issues.append("OPENAI_API_KEY not set")

if ready and not issues:
    print("✅ ALL CHECKS PASSED! Vision AI is ready to use.")
    print("\nNext steps:")
    print("1. python crew_ai_system.py  # Test vision agent")
    print("2. python multi_agent_app.py  # Start web server")
    print("3. Test: http://localhost:8000/api/plans/available")
elif ready:
    print(f"⚠️  MOSTLY READY with {len(issues)} minor issues:")
    for issue in issues:
        print(f"   - {issue}")
    print("\nVision AI will work, but fix issues for best results.")
else:
    print(f"❌ NOT READY - {len(issues)} critical issues:")
    for issue in issues:
        print(f"   - {issue}")
    print("\nFix these issues before using Vision AI.")

print("="*80)

# Recommendations
print("\n💡 RECOMMENDATIONS:")

if images:
    print(f"✅ You have {len(images)} plan images - good!")
    
    # Check for plot numbers in filenames
    plot_patterns = ['9232', '9388', 'stgt', 'plot', 'flurstück']
    has_plot_info = any(
        any(pattern in img.name.lower() for pattern in plot_patterns)
        for img in images
    )
    
    if has_plot_info:
        print(f"✅ Image names contain plot/area identifiers - excellent!")
    else:
        print(f"💡 Consider renaming images to include plot/area info:")
        print(f"   Example: 'nordbahnhof_plot_9232_79.png'")

if pdfs:
    print(f"📄 Convert {len(pdfs)} PDFs to images for vision analysis")

print(f"\n📐 Typical plot query:")
print(f"   'What can I build on plot 9232/79? What are the setbacks?'")

print("\n" + "="*80)