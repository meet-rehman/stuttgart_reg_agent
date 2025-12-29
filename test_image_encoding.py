#!/usr/bin/env python3
"""
Diagnostic script to test if your plan images can be properly encoded
Run this to identify image issues before sending to OpenAI Vision API
"""

import base64
import io
from pathlib import Path
from PIL import Image
import sys

def test_image_encoding(image_path: str):
    """Test if an image can be properly encoded for Vision API"""
    
    print("="*80)
    print(f"TESTING IMAGE: {image_path}")
    print("="*80)
    
    path = Path(image_path)
    
    if not path.exists():
        print(f"❌ ERROR: File not found: {image_path}")
        return False
    
    try:
        # Step 1: Open with PIL
        print("\n1️⃣ Opening image with PIL...")
        img = Image.open(path)
        print(f"   ✅ Success")
        print(f"   Format: {img.format}")
        print(f"   Size: {img.size} ({img.width}x{img.height})")
        print(f"   Mode: {img.mode}")
        
        # Step 2: Check file size
        print("\n2️⃣ Checking file size...")
        file_size_kb = path.stat().st_size / 1024
        file_size_mb = file_size_kb / 1024
        print(f"   File size: {file_size_kb:.1f} KB ({file_size_mb:.2f} MB)")
        
        if file_size_mb > 20:
            print(f"   ⚠️  WARNING: File is very large (>{file_size_mb:.1f} MB)")
            print(f"      OpenAI Vision API may have issues with files >20 MB")
        
        # Step 3: Test resize if needed
        max_dimension = 2048
        if max(img.size) > max_dimension:
            print(f"\n3️⃣ Testing resize (image is larger than {max_dimension}px)...")
            if img.width > img.height:
                new_width = max_dimension
                new_height = int((img.height / img.width) * max_dimension)
            else:
                new_height = max_dimension
                new_width = int((img.width / img.height) * max_dimension)
            
            print(f"   Original: {img.size}")
            print(f"   Resized: ({new_width}, {new_height})")
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"   ✅ Resize successful")
            img = img_resized
        else:
            print(f"\n3️⃣ Image size OK ({img.size}), no resize needed")
        
        # Step 4: Test mode conversion
        print(f"\n4️⃣ Testing mode conversion...")
        if img.mode in ('RGBA', 'LA', 'P'):
            print(f"   Converting {img.mode} → RGB...")
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
            print(f"   ✅ Converted to RGB")
        else:
            print(f"   Mode is {img.mode}, no conversion needed")
        
        # Step 5: Test JPEG encoding
        print(f"\n5️⃣ Testing JPEG encoding...")
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85, optimize=True)
        buffer.seek(0)
        jpeg_size_kb = len(buffer.getvalue()) / 1024
        print(f"   ✅ JPEG encoding successful")
        print(f"   JPEG size: {jpeg_size_kb:.1f} KB")
        
        # Step 6: Test base64 encoding
        print(f"\n6️⃣ Testing base64 encoding...")
        base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        base64_size_kb = len(base64_string) * 3 / 4 / 1024
        print(f"   ✅ Base64 encoding successful")
        print(f"   Base64 size: {base64_size_kb:.1f} KB")
        print(f"   Base64 preview: {base64_string[:50]}...")
        
        if base64_size_kb > 20000:
            print(f"   ⚠️  WARNING: Encoded image very large ({base64_size_kb:.1f} KB)")
        
        # Step 7: Test data URL format
        print(f"\n7️⃣ Testing data URL format...")
        data_url = f"data:image/jpeg;base64,{base64_string}"
        print(f"   ✅ Data URL created")
        print(f"   URL length: {len(data_url):,} characters")
        print(f"   URL preview: {data_url[:80]}...")
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nImage is ready to be sent to Vision API")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_images(image_dir: str):
    """Test all images in a directory"""
    
    path = Path(image_dir)
    
    if not path.exists():
        print(f"❌ ERROR: Directory not found: {image_dir}")
        return
    
    if not path.is_dir():
        print(f"❌ ERROR: Not a directory: {image_dir}")
        return
    
    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
    images = [f for f in path.iterdir() if f.suffix.lower() in image_extensions]
    
    if not images:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"\nFound {len(images)} images in {image_dir}")
    print("="*80)
    
    results = []
    for img_path in images:
        success = test_image_encoding(str(img_path))
        results.append((img_path.name, success))
        print("\n")
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed
    
    print(f"\nTotal images: {len(results)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed > 0:
        print("\nFailed images:")
        for name, success in results:
            if not success:
                print(f"  - {name}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Test single image:")
        print("    python test_image_encoding.py path/to/image.png")
        print()
        print("  Test all images in directory:")
        print("    python test_image_encoding.py path/to/image/directory/")
        print()
        print("Example:")
        print("    python test_image_encoding.py data/landuse_plans/Stuttgart_Nord/")
        sys.exit(1)
    
    test_path = sys.argv[1]
    path = Path(test_path)
    
    if path.is_file():
        # Test single image
        success = test_image_encoding(test_path)
        sys.exit(0 if success else 1)
    elif path.is_dir():
        # Test all images in directory
        test_multiple_images(test_path)
        sys.exit(0)
    else:
        print(f"❌ ERROR: Path not found: {test_path}")
        sys.exit(1)


# ============================================================================
# HOW TO USE THIS SCRIPT
# ============================================================================
#
# 1. Save this file as: test_image_encoding.py
#
# 2. Make it executable:
#    chmod +x test_image_encoding.py
#
# 3. Test a single image:
#    python test_image_encoding.py "data/landuse_plans/Stuttgart_Nord/Flächennutzungsplan Nordbahnhofstarsse_page_14.png"
#
# 4. Or test all images in a directory:
#    python test_image_encoding.py data/landuse_plans/Stuttgart_Nord/
#
# 5. Look for any ❌ ERRORS or ⚠️ WARNINGS
#
# ============================================================================


# ============================================================================
# WHAT TO LOOK FOR IN THE OUTPUT
# ============================================================================
#
# ✅ GOOD (All tests passed):
#    Format: PNG
#    Size: (1024, 768)
#    Mode: RGB
#    File size: 856.3 KB
#    JPEG size: 234.5 KB
#    Base64 size: 312.6 KB
#    ✅ ALL TESTS PASSED!
#
# ❌ BAD (Image has issues):
#    ❌ ERROR: cannot identify image file
#    → Image is corrupted
#
#    ⚠️ WARNING: File is very large (>25.3 MB)
#    → Image too large, will fail with Vision API
#
#    ❌ ERROR during testing: OSError: cannot write mode RGBA as JPEG
#    → Transparency issue (should be handled by script)
#
# ============================================================================