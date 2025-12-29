#!/usr/bin/env python3
"""
Convert Landuse Plan PDFs to Images for Vision AI
Specifically for your Stuttgart landuse plans structure
"""

from pdf2image import convert_from_path
from pathlib import Path
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuration
POPPLER_PATH = r'C:\Program Files\poppler\poppler-25.07.0\Library\bin'
PROJECT_ROOT = Path(__file__).parent
PLANS_DIR = PROJECT_ROOT / "data" / "raw" / "Landuse Plans"

# DPI settings (balance of quality vs file size)
DPI = 200  # Good for reading text and seeing details
MAX_IMAGE_SIZE_MB = 15  # Skip images larger than this

# Increase PIL limit for large images
Image.MAX_IMAGE_PIXELS = 500000000

def get_output_dir(pdf_path: Path) -> Path:
    """
    Create output directory structure
    Example: Stuttgart Nord/Nordbahnhofstarsse/.../plan.pdf
         ->  Stuttgart Nord/Nordbahnhofstarsse/.../images/plan_page_1.png
    """
    # Get relative path from Landuse Plans
    rel_path = pdf_path.relative_to(PLANS_DIR)
    
    # Create images subdirectory in same location
    output_dir = PLANS_DIR / rel_path.parent / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def convert_pdf_to_images(pdf_path: Path, dpi: int = DPI) -> list:
    """Convert a PDF to images"""
    
    output_dir = get_output_dir(pdf_path)
    
    logger.info(f"\n📄 Converting: {pdf_path.name}")
    logger.info(f"   Output: {output_dir.relative_to(PLANS_DIR)}")
    
    converted_images = []
    
    try:
        # Convert PDF to images
        logger.info(f"   Converting at {dpi} DPI...")
        images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            fmt='png',
            poppler_path=POPPLER_PATH,
            thread_count=2
        )
        
        logger.info(f"   ✅ Converted {len(images)} page(s)")
        
        # Save each page
        for i, image in enumerate(images, 1):
            # Create filename
            output_name = f"{pdf_path.stem}_page_{i}.png"
            output_path = output_dir / output_name
            
            # Save image
            image.save(output_path, 'PNG', optimize=True)
            
            # Check size
            size_mb = output_path.stat().st_size / (1024 * 1024)
            
            if size_mb > MAX_IMAGE_SIZE_MB:
                logger.warning(f"      ⚠️  Page {i}: {size_mb:.1f} MB (large!)")
            else:
                logger.info(f"      ✅ Page {i}: {output_name} ({size_mb:.1f} MB)")
            
            converted_images.append(output_path)
        
        return converted_images
        
    except Exception as e:
        logger.error(f"   ❌ Error: {e}")
        
        # Try with lower DPI
        if dpi > 150:
            logger.info(f"   Retrying with lower DPI (150)...")
            return convert_pdf_to_images(pdf_path, dpi=150)
        
        return []


def should_convert(pdf_path: Path) -> bool:
    """
    Determine if PDF should be converted based on filename
    
    Convert these types:
    - Landuse plans (Flächennutzungsplan)
    - Site plans (Bebauungsplan, site draft)
    - Development plans
    - Plan drawings (planzeichnung)
    
    Skip these:
    - Text documents (begruendung, justification)
    - Reports (umweltbericht, environmental report)
    - Regulations (building regulations, statutes)
    - Administrative (bekanntmachung, suggestions)
    """
    
    filename_lower = pdf_path.name.lower()
    
    # Definitely convert (plan drawings)
    convert_keywords = [
        'landuse',
        'flächennutzungsplan',
        'bebauungsplan',
        'site draft',
        'site plan',
        'planzeichnung',
        'development plan',
        'stgt-272',  # Your specific Nordbahnhof plan
        'stgt-286',
        'ca-283',
        'fnp-61-anlage1',  # Main plan drawing
        'fnp-61-planzeichnung',
    ]
    
    # Definitely skip (text documents)
    skip_keywords = [
        'begruendung',
        'justification',
        'umweltbericht',
        'environmental report',
        'building regulations',
        'statutes',
        'bekanntmachung',
        'suggestions',
        'beschlussvorlage',
        'erklaerung',
        'beteiligung',
        'unterrichtung',
        'green space',
        'enactment',
        'model calculation',
        'beiblatt',
        'legende',
        'text.pdf'
    ]
    
    # Check for skip keywords first
    if any(keyword in filename_lower for keyword in skip_keywords):
        return False
    
    # Check for convert keywords
    if any(keyword in filename_lower for keyword in convert_keywords):
        return True
    
    # Default: don't convert unless explicitly marked
    return False


def main():
    """Main conversion process"""
    
    print("="*80)
    print("🎨 LANDUSE PLAN PDF TO IMAGE CONVERTER")
    print("="*80)
    print(f"\nPlans directory: {PLANS_DIR}")
    print(f"DPI: {DPI}")
    print(f"Output: images/ subdirectories\n")
    
    # Find all PDFs
    all_pdfs = list(PLANS_DIR.rglob("*.pdf"))
    
    print(f"📚 Found {len(all_pdfs)} total PDFs\n")
    
    # Filter PDFs to convert
    pdfs_to_convert = [pdf for pdf in all_pdfs if should_convert(pdf)]
    pdfs_to_skip = [pdf for pdf in all_pdfs if not should_convert(pdf)]
    
    print(f"✅ Will convert: {len(pdfs_to_convert)} plan drawings")
    print(f"⏭️  Will skip: {len(pdfs_to_skip)} text documents\n")
    
    # Show what will be converted
    print("="*80)
    print("PLAN DRAWINGS TO CONVERT:")
    print("="*80)
    for pdf in pdfs_to_convert:
        rel_path = pdf.relative_to(PLANS_DIR)
        print(f"  ✅ {rel_path}")
    
    print("\n" + "="*80)
    print("TEXT DOCUMENTS TO SKIP:")
    print("="*80)
    for pdf in pdfs_to_skip[:10]:  # Show first 10
        rel_path = pdf.relative_to(PLANS_DIR)
        print(f"  ⏭️  {rel_path}")
    if len(pdfs_to_skip) > 10:
        print(f"  ... and {len(pdfs_to_skip)-10} more")
    
    # Confirm
    print("\n" + "="*80)
    response = input(f"\nProceed with converting {len(pdfs_to_convert)} PDFs? (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print("❌ Conversion cancelled")
        return
    
    # Convert
    print("\n" + "="*80)
    print("CONVERTING...")
    print("="*80)
    
    total_images = 0
    successful = 0
    failed = 0
    
    for i, pdf in enumerate(pdfs_to_convert, 1):
        print(f"\n[{i}/{len(pdfs_to_convert)}]")
        
        images = convert_pdf_to_images(pdf)
        
        if images:
            total_images += len(images)
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*80)
    print("✅ CONVERSION COMPLETE")
    print("="*80)
    print(f"📊 PDFs processed: {len(pdfs_to_convert)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"📸 Total images created: {total_images}")
    print(f"📁 Images saved in: {PLANS_DIR / '*/images/'}")
    
    # Show directory structure
    print("\n" + "="*80)
    print("📂 OUTPUT STRUCTURE:")
    print("="*80)
    
    image_dirs = list(PLANS_DIR.rglob("images"))
    for img_dir in sorted(image_dirs):
        images_in_dir = list(img_dir.glob("*.png"))
        if images_in_dir:
            rel_path = img_dir.relative_to(PLANS_DIR)
            print(f"\n{rel_path}:")
            print(f"  {len(images_in_dir)} images")
            for img in images_in_dir[:3]:
                size_mb = img.stat().st_size / (1024 * 1024)
                print(f"    - {img.name} ({size_mb:.1f} MB)")
            if len(images_in_dir) > 3:
                print(f"    ... and {len(images_in_dir)-3} more")
    
    print("\n" + "="*80)
    print("🎯 NEXT STEPS:")
    print("="*80)
    print("1. Check image quality in the images/ folders")
    print("2. Run: python check_plans_readiness.py")
    print("3. Update vision agent to look in images/ subdirectories")
    print("4. Test: python crew_ai_system.py")
    print("="*80)


if __name__ == "__main__":
    main()