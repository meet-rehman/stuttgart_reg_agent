#!/usr/bin/env python3
"""
Quick test after fixes
"""

import os
from dotenv import load_dotenv
from optimized_crew_ai_system import StuttgartBuildingRegulationCrew, RegulationQuery

load_dotenv()

def quick_test():
    """Simple test to verify fixes work"""
    
    print("="*60)
    print("🧪 QUICK TEST - Post-Fix Verification")
    print("="*60)
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No API key found")
        return
    
    # Create crew (fast mode - no vision, cheaper model)
    print("\n📦 Creating crew...")
    crew = StuttgartBuildingRegulationCrew(
        openai_api_key=api_key,
        use_gpt4=False,  # Use GPT-3.5 for speed
        enable_vision=False  # Disable vision for faster test
    )
    print("✅ Crew created")
    
    # Simple query (should complete in ~20-30 seconds)
    print("\n🔍 Testing query...")
    query = RegulationQuery(
        query="What is GRZ?",
        project_type="Residential",
        location="Stuttgart",
        enable_vision=False
    )
    
    result = crew.execute_analysis(query)
    
    # Check results
    print("\n" + "="*60)
    print("📊 TEST RESULTS")
    print("="*60)
    
    print(f"\n✅ Success: {result['success']}")
    print(f"⏱️  Time: {result['processing_time']:.1f}s")
    
    # Get the analysis
    analysis = result.get('analysis', result.get('error', 'No output'))
    
    # Critical checks
    print("\n🔍 OUTPUT QUALITY CHECKS:")
    
    checks = {
        "Has content (>100 chars)": len(analysis) > 100,
        "No [X] placeholders": "[X]" not in analysis,
        "No [value] placeholders": "[value]" not in analysis,
        "No [N] placeholders": "[N]" not in analysis,
        "Contains 'GRZ' keyword": "GRZ" in analysis or "Grundflächenzahl" in analysis,
    }
    
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
    
    # Show preview
    print("\n📄 OUTPUT PREVIEW (first 500 chars):")
    print("-"*60)
    print(analysis[:500])
    print("-"*60)
    
    # Final verdict
    all_passed = all(checks.values())
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED - Ready for deployment!")
        return True
    else:
        print("\n⚠️  SOME CHECKS FAILED - Fix issues before deploying")
        return False

if __name__ == "__main__":
    success = quick_test()
    exit(0 if success else 1)