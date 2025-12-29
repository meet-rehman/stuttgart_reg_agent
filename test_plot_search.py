#!/usr/bin/env python3
"""
Test searching for specific plots
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from optimized_vision_agent import OptimizedVisionAgent, VisionConfig

load_dotenv()

def test_plot_search():
    """Test plot search with known Bebauungspläne"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Create agent
    config = VisionConfig(
        model="gpt-4o",  # Use the working model
        max_plans_to_analyze=3,
        parallel_workers=1,
        timeout_per_plan=60
    )
    
    agent = OptimizedVisionAgent(api_key, config=config)
    
    # List available plans
    plans_dir = Path("data/raw/Landuse Plans")
    all_plans = list(plans_dir.glob("**/*.png"))
    
    print("="*60)
    print("📋 AVAILABLE PLANS:")
    print("="*60)
    for i, plan in enumerate(all_plans[:10], 1):
        print(f"{i}. {plan.parent.name}/{plan.name}")
    
    # Test with different images
    test_cases = [
        {
            "plot": "18A",
            "expected_plan": "Stgt 286"  # This plot should be in Bebauungsplan Stgt 286-2
        },
        {
            "plot": "9378/2",
            "expected_plan": "Stgt 272"  # This plot should be in Bebauungsplan Stgt 272
        }
    ]
    
    for test in test_cases:
        print("\n" + "="*60)
        print(f"🔍 Searching for plot: {test['plot']}")
        print(f"   Expected in: {test['expected_plan']}")
        print("="*60)
        
        result = agent.find_plot_parallel(test['plot'], timeout=120)
        
        print(f"\n📊 Result:")
        print(f"   Found: {result.get('found', False)}")
        
        if result['found']:
            print(f"   Plan: {result['plan_file']}")
            print(f"   Time: {result['search_time']:.1f}s")
            print(f"\n📄 Analysis:\n{result['analysis'][:500]}...")
        else:
            print(f"   Message: {result.get('message', 'Not found')}")
            print(f"   Searched: {result.get('searched_plans', 0)} plans")

if __name__ == "__main__":
    test_plot_search()