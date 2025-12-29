#!/usr/bin/env python3
"""
Extract all legend files in batch
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from legend_extractor import LegendExtractor
import json
import time

load_dotenv()

def extract_all_legends():
    """Extract all legend files found in the system"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return
    
    extractor = LegendExtractor(api_key)
    
    # List of all legend files
    legend_files = [
        "data/raw/Landuse Plans/Stuttgart Nord/Bürgerhospital area/images/fnp-61-anlage1-planzeichnung_page_1.png",
        "data/raw/Landuse Plans/Stuttgart Nord/Bürgerhospital area/images/fnp-61-anlage1-planzeichnung_page_2.png",
        "data/raw/Landuse Plans/Stuttgart Nord/Bürgerhospital area/images/fnp-61-anlage1-planzeichnung_page_3.png"
    ]
    
    print("="*60)
    print("🔍 BATCH LEGEND EXTRACTION")
    print("="*60)
    print(f"Found {len(legend_files)} legend files to process\n")
    
    results = {}
    successful = 0
    failed = 0
    
    for i, legend_path_str in enumerate(legend_files, 1):
        legend_path = Path(legend_path_str)
        
        print(f"\n{'='*60}")
        print(f"[{i}/{len(legend_files)}] Processing: {legend_path.name}")
        print('='*60)
        
        if not legend_path.exists():
            print(f"❌ File not found: {legend_path}")
            failed += 1
            continue
        
        try:
            # Extract legend
            legend_data = extractor.extract_legend(legend_path)
            
            # Check if extraction was successful
            if "error" in legend_data:
                print(f"⚠️ Extraction failed: {legend_data.get('error')}")
                print(f"   Response: {legend_data.get('raw_response', '')[:200]}...")
                failed += 1
                results[legend_path.name] = {"status": "failed", "data": legend_data}
            else:
                print(f"✅ Successfully extracted!")
                print(f"   Zoning types: {len(legend_data.get('zoning_types', {}))}")
                print(f"   Colors: {len(legend_data.get('color_meanings', {}))}")
                print(f"   Symbols: {len(legend_data.get('symbols', {}))}")
                print(f"   Abbreviations: {len(legend_data.get('abbreviations', {}))}")
                successful += 1
                results[legend_path.name] = {"status": "success", "data": legend_data}
            
            # Brief pause between API calls
            if i < len(legend_files):
                time.sleep(2)
                
        except Exception as e:
            print(f"❌ Error: {e}")
            failed += 1
            results[legend_path.name] = {"status": "error", "error": str(e)}
    
    # Summary
    print("\n" + "="*60)
    print("📊 EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total files: {len(legend_files)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    
    # Save comprehensive results
    output_file = Path("data/all_legends_extracted.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Print which legends were successfully extracted
    print("\n" + "="*60)
    print("✅ SUCCESSFULLY EXTRACTED LEGENDS")
    print("="*60)
    
    for filename, result in results.items():
        if result["status"] == "success":
            data = result["data"]
            print(f"\n📄 {filename}")
            print(f"   Document: {data.get('document_name', 'N/A')}")
            print(f"   Type: {data.get('document_type', 'N/A')}")
            
            # Show key zoning types
            zoning = data.get('zoning_types', {})
            if zoning:
                print(f"   Zoning types: {', '.join(list(zoning.keys())[:5])}")
    
    # Show cached legends
    print("\n" + "="*60)
    print("💾 CACHED LEGENDS (available for vision agent)")
    print("="*60)
    
    cache_file = Path("data/legends_cache.json")
    if cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        
        for filename, legend_data in cache.items():
            if not isinstance(legend_data, dict) or "error" in legend_data:
                continue
            print(f"✅ {filename}")
            print(f"   → {legend_data.get('document_type', 'Unknown type')}")

if __name__ == "__main__":
    extract_all_legends()