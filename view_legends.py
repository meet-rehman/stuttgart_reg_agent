# Create view_legends.py
import json
from pathlib import Path

cache_file = Path("data/legends_cache.json")

if cache_file.exists():
    with open(cache_file, 'r', encoding='utf-8') as f:
        legends = json.load(f)
    
    for filename, data in legends.items():
        if "error" in data:
            continue
            
        print("="*60)
        print(f"📄 {filename}")
        print("="*60)
        print(f"Document: {data.get('document_name', 'N/A')}")
        print(f"Type: {data.get('document_type', 'N/A')}\n")
        
        print("Zoning Types:")
        for code, meaning in data.get('zoning_types', {}).items():
            print(f"  {code}: {meaning}")
        
        print("\nColor Meanings:")
        for color, meaning in data.get('color_meanings', {}).items():
            print(f"  {color}: {meaning}")
        
        print("\nAbbreviations:")
        for abbr, meaning in data.get('abbreviations', {}).items():
            print(f"  {abbr}: {meaning}")
        
        print()