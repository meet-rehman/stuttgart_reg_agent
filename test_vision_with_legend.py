#!/usr/bin/env python3
"""
Test vision agent with legend knowledge
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from optimized_vision_agent import OptimizedVisionAgent, VisionConfig

load_dotenv()

def test_with_legend():
    """Test that vision agent uses legend knowledge"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    config = VisionConfig(
        model="gpt-4o",
        max_plans_to_analyze=1,
        timeout_per_plan=60
    )
    
    agent = OptimizedVisionAgent(api_key, config=config)
    
    # Test with a Flächennutzungsplan
    test_plan = Path("data/raw/Landuse Plans/Stuttgart Nord/Nordbahnhofstarsse_Freidhofsstrasse (strasse)/images/Flächennutzungsplan Nordbahnhofstarsse_page_1.png")
    
    if not test_plan.exists():
        print(f"❌ Test plan not found: {test_plan}")
        return
    
    print("="*60)
    print("🧪 TESTING VISION AGENT WITH LEGEND KNOWLEDGE")
    print("="*60)
    print(f"\n📄 Analyzing: {test_plan.name}")
    
    # Test 1: Ask about zoning (should use legend)
    print("\n" + "="*60)
    print("TEST 1: What zoning types are visible?")
    print("="*60)
    
    result1 = agent.analyze_plan(
        test_plan,
        "What zoning types can you see? Use the legend to identify WA, MI, GE, SO zones and their colors.",
        detail="high"
    )
    
    print(f"\n📊 RESULT:\n{result1}")
    
    # Test 2: Ask about colors (should reference legend)
    print("\n" + "="*60)
    print("TEST 2: What do the colors mean?")
    print("="*60)
    
    result2 = agent.analyze_plan(
        test_plan,
        "What colored areas can you see? Match the colors to their meanings using the legend (red=residential, yellow=commercial, green=green space, blue=water).",
        detail="high"
    )
    
    print(f"\n📊 RESULT:\n{result2}")
    
    # Check if legend was used
    print("\n" + "="*60)
    print("📋 LEGEND USAGE CHECK")
    print("="*60)
    
    if "Wohngebiet" in result1 or "residential" in result1.lower():
        print("✅ Legend terminology detected in response")
    else:
        print("⚠️ Legend terminology NOT detected")
    
    if "WA" in result1 or "MI" in result1 or "GE" in result1:
        print("✅ Zoning codes detected in response")
    else:
        print("⚠️ Zoning codes NOT detected")

if __name__ == "__main__":
    test_with_legend()