#!/usr/bin/env python3
"""
Find all legend and key files
"""

from pathlib import Path

def find_legend_files():
    """Find all potential legend files"""
    
    plans_dir = Path("data/raw/Landuse Plans")
    
    if not plans_dir.exists():
        print(f"❌ Directory not found: {plans_dir}")
        return
    
    print("="*60)
    print("🔍 SEARCHING FOR LEGEND FILES")
    print("="*60)
    
    # Keywords that indicate legend pages
    legend_keywords = [
        'legend', 'legende', 'zeichenerklärung', 'zeichen',
        'anlage', 'key', 'symbole'
    ]
    
    all_images = list(plans_dir.glob("**/*.png"))
    
    print(f"\n📊 Total images found: {len(all_images)}")
    
    # Find potential legends
    legend_files = []
    for img in all_images:
        name_lower = img.name.lower()
        if any(keyword in name_lower for keyword in legend_keywords):
            legend_files.append(img)
    
    print(f"\n📋 POTENTIAL LEGEND FILES ({len(legend_files)}):")
    print("="*60)
    
    for i, legend in enumerate(legend_files, 1):
        print(f"\n{i}. {legend.name}")
        print(f"   Path: {legend}")
        print(f"   Parent: {legend.parent.name}")
    
    # Also show all files in case we missed some
    print(f"\n\n📁 ALL FILES (first 20):")
    print("="*60)
    
    for i, img in enumerate(all_images[:20], 1):
        print(f"{i}. {img.parent.name}/{img.name}")
    
    return legend_files

if __name__ == "__main__":
    legends = find_legend_files()
    
    if legends:
        print(f"\n\n✅ Found {len(legends)} legend files!")
        print("\nTo extract a legend, use:")
        print(f'python legend_extractor.py "{legends[0]}"')
    else:
        print("\n⚠️ No obvious legend files found")
        print("Check the full list above - legends might have different names")