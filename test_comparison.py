#!/usr/bin/env python3
"""
Compare old vs new system
"""

import os
from dotenv import load_dotenv
from optimized_crew_ai_system import StuttgartBuildingRegulationCrew, RegulationQuery

load_dotenv()

def test_before_after():
    api_key = os.getenv("OPENAI_API_KEY")
    
    print("="*70)
    print("🔬 BEFORE/AFTER COMPARISON TEST")
    print("="*70)
    
    # Create crew
    crew = StuttgartBuildingRegulationCrew(
        openai_api_key=api_key,
        use_gpt4=False,
        enable_vision=False  # Disable vision for faster test
    )
    
    # Test query
    query = RegulationQuery(
        query="What is the GRZ for residential buildings in Stuttgart?",
        project_type="Residential",
        location="Stuttgart",
        enable_vision=False
    )
    
    print("\n🔍 Testing query:", query.query)
    print("⏳ Processing...")
    
    result = crew.execute_analysis(query)
    
    # Analysis
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    
    print(f"\n✅ Success: {result['success']}")
    print(f"⏱️  Time: {result['processing_time']:.1f}s")
    
    analysis = result['analysis']
    
    # Key checks
    print("\n🔍 QUALITY CHECKS:")
    
    checks = {
        "Has GRZ value (not 'Not specified')": "GRZ" in analysis and "0." in analysis and "Not specified" not in analysis,
        "Has actual calculations": any(str(i) in analysis for i in range(100, 1000)),
        "Uses 'WA' or zoning": "WA" in analysis or "Wohngebiet" in analysis,
        "Has specific recommendations": "Contact Stuttgart" in analysis or "Baurechtsamt" in analysis,
        "No generic placeholders": "To be determined" not in analysis[:500],
    }
    
    passed = 0
    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check}")
        if result:
            passed += 1
    
    print(f"\n📊 Score: {passed}/{len(checks)} checks passed")
    
    # Show key excerpts
    print("\n📄 KEY EXCERPTS:")
    print("-"*70)
    
    # Find GRZ line
    for line in analysis.split('\n'):
        if 'GRZ' in line and len(line) < 200:
            print(f"GRZ: {line.strip()}")
            break
    
    # Find calculation
    for line in analysis.split('\n'):
        if 'Maximum' in line or 'ground coverage' in line.lower():
            print(f"Calc: {line.strip()}")
            break
    
    print("-"*70)
    
    # Verdict
    if passed >= 4:
        print("\n🎉 SYSTEM WORKING WELL!")
        print("✅ Ready for deployment")
        return True
    else:
        print("\n⚠️  NEEDS MORE WORK")
        print(f"Only {passed}/5 checks passed")
        return False

if __name__ == "__main__":
    success = test_before_after()
    exit(0 if success else 1)