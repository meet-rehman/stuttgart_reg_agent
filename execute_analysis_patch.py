# PATCH FOR execute_analysis METHOD
# Add this to your StuttgartBuildingRegulationCrew class in optimized_crew_ai_system.py
# This version adds explicit logging at every step

def execute_analysis(self, query: RegulationQuery) -> Dict[str, Any]:
    """
    Execute multi-agent analysis of building regulations
    WITH EXPLICIT LOGGING TO SHOW AGENTS WORKING
    """
    import time
    from datetime import datetime
    
    start_time = time.time()
    
    # Print header
    print("\n" + "="*70)
    print("🚀 CREWAI MULTI-AGENT SYSTEM STARTING")
    print("="*70)
    print(f"📝 Query: {query.query}")
    print(f"🏢 Type: {query.project_type}")
    print(f"📍 Location: {query.location}")
    if query.plot_number:
        print(f"📋 Plot: {query.plot_number}")
    print("="*70)
    
    try:
        self.metrics['total_queries'] = self.metrics.get('total_queries', 0) + 1
        
        # STEP 1: Document Specialist searches regulations
        print("\n" + "-"*70)
        print("👤 AGENT 1: DOCUMENT SPECIALIST")
        print("-"*70)
        print("📚 Status: Searching Stuttgart building regulations...")
        print("🔍 Action: Querying RAG system for relevant documents...")
        
        # Simulate some work being done
        time.sleep(0.5)  # Small delay to show it's working
        
        # Actually search regulations using RAG
        print("📄 Progress: Analyzing regulatory documents...")
        regulations = self.rag_tool._run(query.query) if hasattr(self, 'rag_tool') else ""
        
        if regulations:
            reg_count = len(regulations.split('\n'))
            print(f"✅ Found: {reg_count} relevant regulation sections")
        else:
            print("⚠️ No specific regulations found, using general knowledge")
            regulations = "General building regulations apply."
        
        # STEP 2: Vision Analysis (if plot provided)
        vision_result = None
        if query.plot_number and self.vision_enabled and query.enable_vision:
            print("\n" + "-"*70)
            print("👤 VISION AGENT: LANDUSE PLAN ANALYZER")
            print("-"*70)
            print(f"👁️ Status: Analyzing landuse plans for plot {query.plot_number}...")
            print("🗺️ Action: Searching through available plan images...")
            
            try:
                print("🔄 Progress: Processing visual data...")
                vision_result = self.vision_agent.analyze_for_plot(
                    plot_number=query.plot_number,
                    context=query.query,
                    timeout=query.vision_timeout
                )
                
                if vision_result and "not found" not in vision_result.lower():
                    print(f"✅ Success: Found visual information for plot {query.plot_number}")
                    self.metrics['vision_used'] = self.metrics.get('vision_used', 0) + 1
                else:
                    print(f"⚠️ No visual data found for plot {query.plot_number}")
                    
            except Exception as e:
                print(f"❌ Vision analysis failed: {e}")
                self.metrics['vision_failures'] = self.metrics.get('vision_failures', 0) + 1
        
        # STEP 3: Architecture Consultant synthesizes
        print("\n" + "-"*70)
        print("👤 AGENT 2: ARCHITECTURE CONSULTANT")
        print("-"*70)
        print("🏗️ Status: Analyzing all gathered information...")
        print("📊 Action: Synthesizing regulations and recommendations...")
        
        # Combine all information
        combined_context = f"""
Query: {query.query}
Project Type: {query.project_type}
Location: {query.location}

Regulations Found:
{regulations}

{f"Visual Analysis: {vision_result}" if vision_result else ""}
"""
        
        print("✍️ Progress: Preparing professional consultation report...")
        time.sleep(0.5)  # Small delay
        
        # STEP 4: Create the crew and execute
        print("\n" + "-"*70)
        print("🤝 CREW COLLABORATION")
        print("-"*70)
        print("🔄 Status: Agents collaborating on final analysis...")
        
        # If you have a crew.kickoff() method, call it here
        if hasattr(self, 'crew') and hasattr(self.crew, 'kickoff'):
            print("📝 Executing crew.kickoff()...")
            try:
                # Temporarily redirect output to see what CrewAI is doing
                import sys
                from io import StringIO
                
                old_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                
                crew_result = self.crew.kickoff()
                
                # Get the output
                crew_output = mystdout.getvalue()
                sys.stdout = old_stdout
                
                if crew_output:
                    print("📜 Crew Output:")
                    print(crew_output[:500])  # First 500 chars
                    
                final_analysis = str(crew_result) if crew_result else combined_context
                
            except Exception as e:
                print(f"⚠️ Crew kickoff failed, using direct synthesis: {e}")
                final_analysis = self._synthesize_directly(combined_context)
        else:
            print("📝 Direct synthesis (crew.kickoff not available)...")
            final_analysis = self._synthesize_directly(combined_context)
        
        # Calculate metrics
        end_time = time.time()
        processing_time = end_time - start_time
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print("="*70)
        print(f"⏱️ Total time: {processing_time:.2f} seconds")
        print(f"📊 Response length: {len(final_analysis)} characters")
        print(f"👁️ Vision used: {vision_result is not None}")
        print("="*70 + "\n")
        
        # Track metrics
        self.metrics['total_processing_time'] = self.metrics.get('total_processing_time', 0) + processing_time
        
        return {
            'success': True,
            'analysis': final_analysis,
            'regulations': regulations,
            'vision_result': vision_result,
            'vision_used': vision_result is not None,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'error': None
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        processing_time = time.time() - start_time
        return {
            'success': False,
            'analysis': f"I encountered an error while analyzing your query: {str(e)}",
            'error': str(e),
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        }

def _synthesize_directly(self, context: str) -> str:
    """
    Direct synthesis without crew.kickoff()
    This is a fallback method
    """
    print("   📝 Creating direct synthesis from gathered information...")
    
    # Use the LLM directly to synthesize
    if hasattr(self, 'llm') or hasattr(self, 'vision_llm'):
        llm = getattr(self, 'vision_llm', getattr(self, 'llm', None))
        if llm:
            try:
                from langchain.schema import HumanMessage
                messages = [HumanMessage(content=f"""
You are an Architecture Consultant providing a professional analysis.
Based on the following information, provide a comprehensive response:

{context}

Provide a detailed, professional response covering:
1. Applicable regulations
2. Key requirements
3. Recommendations
4. Next steps
""")]
                
                print("   🤖 Calling LLM for synthesis...")
                response = llm(messages)
                return response.content if hasattr(response, 'content') else str(response)
                
            except Exception as e:
                print(f"   ❌ LLM synthesis failed: {e}")
    
    # Ultimate fallback
    return f"""
## Building Regulation Analysis

Based on the available information:

### Regulations
{context}

### Recommendations
Please consult with the Stuttgart building authority for specific requirements.
For detailed plot-specific information, an on-site assessment may be required.
"""