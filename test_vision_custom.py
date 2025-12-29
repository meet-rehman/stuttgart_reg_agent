#!/usr/bin/env python3
"""
Test vision agent with custom config for slow connections
"""

import os
from dotenv import load_dotenv
load_dotenv()

from optimized_vision_agent import OptimizedVisionAgent, VisionConfig

# Custom config for slow network connections
config = VisionConfig(
    max_plans_to_analyze=2,      # Analyze fewer plans
    parallel_workers=1,           # No parallel (reduces load)
    timeout_per_plan=30,          # Longer timeout per plan (was 10s)
    total_timeout=90,             # Longer total timeout (was 30s)
    initial_detail="low",         # Keep low detail
    cache_results=True
)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY not set")
    exit(1)

# Create agent with custom config
agent = OptimizedVisionAgent(api_key, config=config)

# Test with a simpler query first
print("\n🧪 TEST 1: List available plans")
print("="*60)
plans = agent._get_available_plans()
print(f"Found {len(plans)} plan images")
if plans:
    print("\nFirst 5 plans:")
    for i, plan in enumerate(plans[:5], 1):
        print(f"  {i}. {plan.name}")

# Test plot search
print("\n🧪 TEST 2: Search for plot")
print("="*60)
test_plot = "9232/79"
print(f"Searching for plot: {test_plot}")
print("⚠️ This may take 1-2 minutes with slow connection...")

try:
    result = agent.find_plot_parallel(test_plot)
    
    if result['found']:
        print(f"\n✅ SUCCESS! Found plot in: {result['plan_file']}")
        print(f"Search time: {result.get('search_time', 0):.1f}s")
        print(f"\nAnalysis:\n{result['analysis'][:500]}...")
    else:
        print(f"\n❌ Plot not found")
        print(f"Searched {result.get('searched_plans', 0)} plans")
        print(f"Search time: {result.get('search_time', 0):.1f}s")
        
except Exception as e:
    print(f"\n❌ Error: {e}")

# Print metrics
print("\n")
agent.print_metrics()