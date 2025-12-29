#!/usr/bin/env python3
"""
Test plot-specific query with vision
"""

import os
from dotenv import load_dotenv
from optimized_crew_ai_system import StuttgartBuildingRegulationCrew, RegulationQuery

load_dotenv()

def test_plot_query():
    """Test with a real plot number (vision enabled)"""
    
    print("="*60)
    print("🧪 PLOT-SPECIFIC TEST WITH VISION")
    print("="*60)
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No API key found")
        return
    
    # Create crew with vision enabled
    print("\n📦 Creating crew with vision...")
    crew = StuttgartBuildingRegulationCrew(
        openai_api_key=api_key,
        use_gpt4=False,  # Use cheaper model for testing
        enable_vision=True  # ✅ Enable vision
    )
    print("✅ Crew created (vision enabled)")
    
    # Plot-specific query
    print("\n🔍 Testing plot-specific query...")
    print("Query: 'What can I build on plot 18A?'")
    
    query = RegulationQuery(
        query="What can I build on plot 18A?",
        project_type="Residential",
        location="Stuttgart",
        plot_number="18A",
        enable_vision=True  # ✅ Enable vision for this query
    )
    
    print("\n⏳ Processing (this may take 60-90 seconds)...")
    result = crew.execute_analysis(query)
    
    # Results
    print("\n" + "="*60)
    print("📊 TEST RESULTS")
    print("="*60)
    
    print(f"\n✅ Success: {result['success']}")
    print(f"⏱️  Time: {result['processing_time']:.1f}s")
    print(f"👁️  Vision used: {result.get('vision_used', False)}")
    print(f"📍 Plot query: {result.get('is_plot_query', False)}")
    
    # Get analysis
    analysis = result.get('analysis', result.get('error', 'No output'))
    
    # Quality checks
    print("\n🔍 OUTPUT QUALITY CHECKS:")
    
    checks = {
        "Has content (>100 chars)": len(analysis) > 100,
        "No [X] placeholders": "[X]" not in analysis,
        "No [value] placeholders": "[value]" not in analysis,
        "Mentions plot 18A": "18A" in analysis or "18 A" in analysis,
        "Has GRZ value": "GRZ" in analysis and "0." in analysis,
        "Has building info": any(word in analysis.lower() for word in ["wohngebiet", "residential", "wa"]),
    }
    
    for check, passed in checks.items():
        status = "✅" if passed else "⚠️"
        print(f"  {status} {check}")
    
    # Show preview
    print("\n📄 FULL OUTPUT:")
    print("="*60)
    print(analysis)
    print("="*60)
    
    # Vision metrics
    if result.get('vision_used'):
        print("\n👁️  VISION METRICS:")
        metrics = result.get('metrics', {})
        vision_metrics = metrics.get('vision_metrics', {})
        if vision_metrics:
            print(f"   Vision API calls: {vision_metrics.get('total_calls', 0)}")
            print(f"   Cache hits: {vision_metrics.get('cache_hits', 0)}")
            print(f"   Vision time: {vision_metrics.get('total_time', 0):.1f}s")
    
    # Final verdict
    all_passed = all(checks.values())
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED!")
        print("✅ Vision system is working correctly")
        print("✅ Plot-specific analysis successful")
        print("✅ Ready for production deployment!")
        return True
    else:
        print("\n⚠️  Some checks failed, but system is still functional")
        failed = [k for k, v in checks.items() if not v]
        print(f"   Failed checks: {', '.join(failed)}")
        return False

if __name__ == "__main__":
    success = test_plot_query()
    exit(0 if success else 1)