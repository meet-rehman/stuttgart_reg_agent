#!/usr/bin/env python3
"""
Quick test script for optimized vision agent
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from optimized_vision_agent import OptimizedVisionAgent, VisionConfig

# Load environment variables
load_dotenv()

def test_single_plan():
    """Test vision analysis on a single plan"""
    
    print("="*60)
    print("🧪 TESTING VISION AGENT")
    print("="*60)
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("   Make sure you have a .env file with OPENAI_API_KEY=sk-...")
        return
    
    # Create agent
    print("\n📦 Initializing Vision Agent...")
    config = VisionConfig(
        max_plans_to_analyze=2,
        parallel_workers=1,
        timeout_per_plan=60
    )
    agent = OptimizedVisionAgent(api_key, config=config)
    
    # Find a test plan
    plans_dir = Path("data/raw/Landuse Plans")
    
    # Try to find any PNG image
    test_plans = list(plans_dir.glob("**/*.png"))
    
    if not test_plans:
        print(f"❌ No PNG files found in {plans_dir}")
        print("   Available directories:")
        for item in plans_dir.iterdir():
            print(f"   - {item}")
        return
    
    # Use first available plan
    test_plan = test_plans[0]
    print(f"\n📄 Testing with: {test_plan.name}")
    print(f"   Path: {test_plan}")
    
    # Test 1: Simple visibility check
    print("\n" + "="*60)
    print("TEST 1: What can you see?")
    print("="*60)
    
    result1 = agent.analyze_plan(
        test_plan,
        "What can you see in this plan? List any plot numbers, street names, or zoning information visible.",
        detail="high"
    )
    
    print("\n📊 RESULT:")
    print(result1)
    
    # Test 2: Specific question
    print("\n" + "="*60)
    print("TEST 2: Specific search")
    print("="*60)
    
    result2 = agent.analyze_plan(
        test_plan,
        "Are there any plot numbers visible? If yes, list them. If no, say 'No plot numbers visible'.",
        detail="high"
    )
    
    print("\n📊 RESULT:")
    print(result2)
    
    # Print metrics
    print("\n" + "="*60)
    print("📈 PERFORMANCE METRICS")
    print("="*60)
    agent.print_metrics()
    
    print("\n✅ Testing complete!")

if __name__ == "__main__":
    test_single_plan()