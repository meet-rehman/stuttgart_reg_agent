#!/usr/bin/env python3
"""
Direct test of vision for plot 18A
"""

import os
from dotenv import load_dotenv
from optimized_vision_agent import OptimizedVisionAgent, VisionConfig

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

print("="*60)
print("🔍 DIRECT VISION TEST FOR PLOT 18A")
print("="*60)

# Create vision agent
print("\n📦 Creating vision agent...")
agent = OptimizedVisionAgent(
    api_key,
    config=VisionConfig(
        model="gpt-4o",
        max_plans_to_analyze=3,
        timeout_per_plan=60
    )
)

print("✅ Vision agent created")
print(f"📋 Legends loaded: {len(agent.legend_extractor.legends_cache)}")

# Search for plot 18A
print("\n🔍 Searching for plot 18A...")
print("(This may take 30-60 seconds)")

result = agent.find_plot_parallel("18A", timeout=120)

# Results
print("\n" + "="*60)
print("📊 VISION SEARCH RESULT")
print("="*60)

print(f"\nFound: {result['found']}")

if result['found']:
    print(f"✅ SUCCESS!")
    print(f"   Plan file: {result['plan_file']}")
    print(f"   Search time: {result['search_time']:.1f}s")
    print(f"   Plans analyzed: {result.get('plans_analyzed', 0)}")
    
    print(f"\n📄 ANALYSIS:")
    print("="*60)
    print(result['analysis'])
    print("="*60)
    
    # Check for actual values
    has_grz = "GRZ" in result['analysis'] or "0." in result['analysis']
    has_wa = "WA" in result['analysis'] or "Wohngebiet" in result['analysis']
    
    print(f"\n✅ Contains GRZ: {has_grz}")
    print(f"✅ Contains zoning (WA): {has_wa}")
else:
    print(f"⚠️ NOT FOUND")
    print(f"   Searched: {result.get('searched_plans', 0)} plans")
    print(f"   Time: {result['search_time']:.1f}s")
    print(f"   Message: {result.get('message', 'Unknown')}")

# Metrics
print("\n" + "="*60)
agent.print_metrics()

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)