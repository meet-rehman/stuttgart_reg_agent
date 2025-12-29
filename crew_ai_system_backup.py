#!/usr/bin/env python3
"""
Multi-Agent Stuttgart Building Regulations System using CrewAI
COST-OPTIMIZED VERSION with proper tool integration
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool as LangChainTool
from langchain_openai import ChatOpenAI

# Custom imports
from precomputed_rag import EnhancedPrecomputedRAGSystem as PrecomputedRAGSystem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RegulationQuery:
    """Structure for regulation queries"""
    query: str
    project_type: str = "mixed-use"
    location: str = "Stuttgart"
    district: str = "general"
    urgency: str = "normal"

# ============================================================================
# CREWAI NATIVE TOOLS (FUNCTION-BASED) - Your RAG system connected here
# ============================================================================

_rag_system = None

def get_rag_system():
    """Lazy load RAG system"""
    global _rag_system
    if _rag_system is None:
        _rag_system = PrecomputedRAGSystem()
    return _rag_system


def search_regulations(query: str) -> str:
    """Search Stuttgart building regulations using your RAG system.
    Input: query string. Returns: relevant documents with citations."""
    try:
        rag_system = get_rag_system()
        
        # 🔍 DEBUG: Log what we're searching
        logger.info(f"🔍 Searching regulations for: {query}")
        
        results = rag_system.search(query, top_k=3)  # Increased from 2 to 3 for better coverage
        
        if not results:
            logger.warning(f"⚠️ No results found for: {query}")
            return f"No documents found for: {query}. Try broader search terms or check district spelling."
        
        # 🔍 DEBUG: Log what we found
        logger.info(f"📊 Found {len(results)} results")
        
        formatted = []
        for i, result in enumerate(results, 1):
            metadata = result.metadata
            content = result.content[:250]  # Slightly increased for better context
            score = result.score
            
            # 🔍 DEBUG: Log each result
            logger.info(f"  Result {i}: {metadata.get('document_name')} (score: {score:.3f})")
            
            formatted.append(f"""Document {i} (Relevance: {score:.2f}):
- Source: {metadata.get('document_name', 'Unknown')}
- Type: {metadata.get('document_type', 'Unknown')}
- Page: {metadata.get('page_number', 'N/A')}
- Citation: {result.get_detailed_citation()}
- Content: {content}...""")
        
        return "\n\n".join(formatted)
        
    except Exception as e:
        logger.error(f"❌ Search error: {e}", exc_info=True)
        return f"Error searching: {str(e)}"


def get_context(query: str) -> str:
    """Get comprehensive context for query.
    Input: query. Returns: detailed context with citations."""
    try:
        rag_system = get_rag_system()
        
        logger.info(f"📖 Getting context for: {query}")
        
        context = rag_system.get_context_for_query(
            query,
            max_tokens=1000,  # Balance between detail and cost
            include_citations=True
        )
        
        if not context or len(context.strip()) < 50:
            logger.warning(f"⚠️ Insufficient context for: {query}")
            return f"Limited context available for: {query}. Consider rephrasing or broadening the query."
        
        return context
        
    except Exception as e:
        logger.error(f"❌ Context error: {e}", exc_info=True)
        return f"Error getting context: {str(e)}"


def analyze_hierarchy(regulations: str) -> str:
    """Analyze regulatory hierarchy. 
    Input: regulations text. Returns: hierarchy analysis."""
    
    logger.info("⚖️ Analyzing regulatory hierarchy")
    
    hierarchy = {
        "federal": ["BauGB", "EnEV", "GEG", "DIN", "VDI", "Baugesetzbuch"],
        "state": ["LBO", "Baden-Württemberg", "BW", "Landesbauordnung"],
        "local": ["Stuttgart", "Zuffenhausen", "Municipal", "Stadt", "Mitte", "West"]
    }
    
    found = {}
    for level, keywords in hierarchy.items():
        matches = [kw for kw in keywords if kw.lower() in regulations.lower()]
        if matches:
            found[level] = matches
    
    if not found:
        return "⚠️ Regulatory level unclear. Recommend consulting federal BauGB, state LBO BW, and local Stuttgart regulations."
    
    analysis_parts = []
    for level in ["federal", "state", "local"]:
        if level in found:
            analysis_parts.append(f"{level.upper()}: {', '.join(found[level])}")
    
    result = "Regulatory Hierarchy:\n" + "\n".join(analysis_parts)
    
    if len(found) > 1:
        result += "\n\nPrecedence: Local regulations take precedence where specifically permitted by state law. State law takes precedence over federal framework within state competencies."
    
    return result


def estimate_costs(requirements: str) -> str:
    """Estimate compliance costs.
    Input: requirements. Returns: cost and timeline."""
    
    logger.info("💰 Estimating compliance costs")
    
    # More comprehensive cost factors
    factors = {
        "accessibility": {"mult": 1.12, "weeks": 2, "desc": "DIN 18040 compliance"},
        "barrier": {"mult": 1.12, "weeks": 2, "desc": "Barrier-free access"},
        "fire_safety": {"mult": 1.18, "weeks": 3, "desc": "Fire protection systems"},
        "fire": {"mult": 1.18, "weeks": 3, "desc": "Fire protection systems"},
        "energy_efficiency": {"mult": 1.25, "weeks": 4, "desc": "GEG/EnEV compliance"},
        "energy": {"mult": 1.25, "weeks": 4, "desc": "Energy standards"},
        "parking": {"mult": 1.08, "weeks": 1, "desc": "Parking requirements"},
        "setback": {"mult": 1.03, "weeks": 1, "desc": "Setback requirements"},
        "sound": {"mult": 1.10, "weeks": 2, "desc": "Sound insulation DIN 4109"},
        "noise": {"mult": 1.10, "weeks": 2, "desc": "Noise protection"}
    }
    
    applicable = []
    total_mult = 1.0
    total_weeks = 0
    
    requirements_lower = requirements.lower()
    for factor, vals in factors.items():
        if factor in requirements_lower:
            if vals["desc"] not in [a["desc"] for a in applicable]:  # Avoid duplicates
                applicable.append(vals)
                total_mult *= vals["mult"]
                total_weeks += vals["weeks"]
    
    if applicable:
        factor_list = [f"• {a['desc']}: +{(a['mult']-1)*100:.1f}% cost, +{a['weeks']}w" for a in applicable]
        return f"""Compliance Cost Analysis:

Applicable Factors:
{chr(10).join(factor_list)}

Total Impact: +{(total_mult-1)*100:.1f}% additional cost
Timeline Extension: +{total_weeks} weeks
Base Timeline: 6-8 weeks for standard permits

💡 Consider phased implementation to manage costs."""
    else:
        return """Standard Compliance Requirements:
- Base timeline: 6-8 weeks
- Standard permitting costs apply
- No exceptional requirements identified"""


# ============================================================================
# MAIN CREW SYSTEM
# ============================================================================

class StuttgartBuildingRegulationCrew:
    """Multi-agent regulation analysis system - COST OPTIMIZED"""
    
    def __init__(self, openai_api_key: str, use_gpt4: bool = False):
        """
        Args:
            openai_api_key: OpenAI API key
            use_gpt4: If True, use GPT-4 (expensive). If False, use GPT-4o-mini (recommended)
        """
        model = "gpt-4" if use_gpt4 else "gpt-4o-mini"
        
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.1,
            openai_api_key=openai_api_key,
            max_tokens=3000
        )
        
        logger.info(f"🤖 Initialized with model: {model}")
        
        # Create agents with tools
        self.agents = self._create_agents()
    
    def _create_agents(self) -> Dict[str, Agent]:
        """Create specialized agents with proper tool integration"""
        
        # Wrap functions as LangChain tools
        search_tool = LangChainTool(
            name="search_regulations",
            func=search_regulations,
            description="Search Stuttgart building regulations. Input: query string. Returns: relevant documents with citations."
        )
        
        context_tool = LangChainTool(
            name="get_context",
            func=get_context,
            description="Get comprehensive context for query. Input: query. Returns: detailed context with citations."
        )
        
        hierarchy_tool = LangChainTool(
            name="analyze_hierarchy",
            func=analyze_hierarchy,
            description="Analyze regulatory hierarchy. Input: regulations text. Returns: hierarchy analysis."
        )
        
        cost_tool = LangChainTool(
            name="estimate_costs",
            func=estimate_costs,
            description="Estimate compliance costs. Input: requirements. Returns: cost and timeline."
        )
        
        # Create agents with tools
        document_specialist = Agent(
            role="Document Research Specialist",
            goal="Find relevant Stuttgart building regulations using the search system",
            backstory="""Expert at searching German building regulation documents. 
            You MUST use the search_regulations and get_context tools to find specific 
            regulations from the Stuttgart database. Never guess - always search first.""",
            tools=[search_tool, context_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        legal_analyst = Agent(
            role="Regulatory Legal Analyst",
            goal="Interpret regulatory hierarchy and legal precedence",
            backstory="""Legal expert in German building law hierarchy (BauGB > LBO BW > Stuttgart).
            Use the analyze_hierarchy tool to determine which regulations take precedence.""",
            tools=[hierarchy_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        technical_expert = Agent(
            role="Technical Standards Expert",
            goal="Analyze DIN standards and technical requirements",
            backstory="""Technical expert in DIN standards (DIN 18040, DIN 4109), fire safety, and energy efficiency.
            Use search_regulations to find specific technical standards.""",
            tools=[search_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        compliance_strategist = Agent(
            role="Compliance Strategy Advisor",
            goal="Develop cost-effective compliance strategies",
            backstory="""Building industry professional who understands practical compliance implications.
            Use estimate_costs to provide realistic cost and timeline assessments.""",
            tools=[cost_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        synthesis_manager = Agent(
            role="Professional Synthesis Manager",
            goal="Create comprehensive professional recommendations",
            backstory="""Senior regulatory consultant who synthesizes all analyses into 
            clear, actionable professional reports for architects and developers.""",
            tools=[],  # Synthesis agent doesn't need tools
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        return {
            "document_specialist": document_specialist,
            "legal_analyst": legal_analyst,
            "technical_expert": technical_expert,
            "compliance_strategist": compliance_strategist,
            "synthesis_manager": synthesis_manager
        }
    
    def create_tasks(self, query: RegulationQuery) -> List[Task]:
        """Create analysis tasks with clear tool usage instructions"""
        
        research_task = Task(
            description=f"""Research building regulations for: {query.query}

Project: {query.project_type} in {query.district}, Stuttgart

MANDATORY: Use search_regulations tool to find:
1. Federal regulations (BauGB, DIN standards)
2. State regulations (LBO Baden-Württemberg)
3. Local Stuttgart regulations
4. District-specific rules for {query.district}

Then use get_context tool for comprehensive details.

Provide specific citations with document names and page numbers.""",
            expected_output="Detailed regulations with citations from Stuttgart database",
            agent=self.agents["document_specialist"]
        )
        
        hierarchy_task = Task(
            description=f"""Analyze regulatory hierarchy from the previous research.

MANDATORY: Use analyze_hierarchy tool on the regulations found.

Determine:
- Which level applies (federal/state/local)
- Precedence rules
- Any conflicts between levels""",
            expected_output="Legal hierarchy analysis with precedence determination",
            agent=self.agents["legal_analyst"]
        )
        
        technical_task = Task(
            description=f"""Find technical requirements for: {query.query}

MANDATORY: Use search_regulations tool to find:
- DIN 18040 (accessibility)
- DIN 4109 (sound insulation)
- Fire safety requirements
- GEG/EnEV energy standards

Provide specific technical standards and compliance criteria.""",
            expected_output="Technical standards with implementation requirements",
            agent=self.agents["technical_expert"]
        )
        
        compliance_task = Task(
            description=f"""Create compliance strategy based on previous findings.

MANDATORY: Use estimate_costs tool on the requirements identified.

Provide:
- Cost impact analysis
- Timeline estimates
- Risk assessment
- Alternative approaches if applicable""",
            expected_output="Compliance strategy with detailed cost/timeline analysis",
            agent=self.agents["compliance_strategist"]
        )
        
        synthesis_task = Task(
            description=f"""Synthesize all analyses into a professional consultation report.

Create comprehensive report with:
1. **Executive Summary** (3-4 key points)
2. **Regulatory Analysis** (with citations from document specialist)
3. **Compliance Roadmap** (step-by-step)
4. **Cost & Timeline** (from compliance strategist)
5. **Risk Factors** (potential issues)
6. **Required Documents** (forms needed)
7. **Next Steps** (actionable recommendations)
8. **Professional Recommendations**

Format professionally for architects/developers/officials.
Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M')}""",
            expected_output="Professional consultation report with all sections",
            agent=self.agents["synthesis_manager"]
        )
        
        # Set dependencies
        hierarchy_task.context = [research_task]
        technical_task.context = [research_task]
        compliance_task.context = [research_task, hierarchy_task, technical_task]
        synthesis_task.context = [research_task, hierarchy_task, technical_task, compliance_task]
        
        return [research_task, hierarchy_task, technical_task, compliance_task, synthesis_task]
    
    def execute_analysis(self, query: RegulationQuery) -> str:
        """Execute multi-agent analysis"""
        try:
            logger.info(f"🏗️ Starting analysis: {query.query}")
            logger.info(f"📍 Location: {query.district}, {query.location}")
            
            tasks = self.create_tasks(query)
            
            crew = Crew(
                agents=list(self.agents.values()),
                tasks=tasks,
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            logger.info("✅ Analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Analysis error: {e}", exc_info=True)
            return f"Error during analysis: {str(e)}"


def inspect_rag_documents():
    """See what documents are actually loaded"""
    
    print("\n" + "="*100)
    print("📚 INSPECTING RAG DOCUMENT COLLECTION")
    print("="*100)
    
    from precomputed_rag import EnhancedPrecomputedRAGSystem as PrecomputedRAGSystem
    rag = PrecomputedRAGSystem()
    
    # Get all unique document names
    if hasattr(rag, 'chunks'):
        doc_names = set()
        doc_types = {}
        
        print(f"\n📊 Analyzing chunks...")
        total_chunks = len(rag.chunks)
        print(f"   Total chunks in system: {total_chunks}")
        
        # Sample chunks to get document names
        sample_size = min(1000, total_chunks)  # Sample first 1000
        for chunk in rag.chunks[:sample_size]:
            if hasattr(chunk, 'metadata'):
                doc_name = chunk.metadata.get('document_name', 'Unknown')
                doc_type = chunk.metadata.get('document_type', 'Unknown')
                doc_names.add(doc_name)
                doc_types[doc_name] = doc_type
        
        print(f"\n📄 Unique documents found (showing first 30):")
        for i, name in enumerate(sorted(doc_names)[:30], 1):
            dtype = doc_types.get(name, 'Unknown')
            print(f"   {i}. {name} ({dtype})")
        
        print(f"\n💡 Total unique documents in sample: {len(doc_names)}")
        if len(doc_names) >= 30:
            print(f"   (Showing 30 of {len(doc_names)} documents)")
    
    # Search for key terms
    print("\n" + "="*100)
    print("🔍 SEARCHING FOR KEY BUILDING REGULATION TERMS")
    print("="*100)
    
    key_terms = [
        "Grundflächenzahl",
        "GRZ",
        "Baunutzungsverordnung",
        "BauNVO",
        "Geschossflächenzahl",
        "GFZ",
        "Vollgeschosse",
        "WA Wohngebiet",
        "Stuttgart-Mitte",
        "Bebauungsplan",
        "Landesbauordnung",
        "LBO"
    ]
    
    for term in key_terms:
        try:
            results = rag.search(term, top_k=1)
            if results:
                score = results[0].score
                doc = results[0].metadata.get('document_name', 'Unknown')
                content = results[0].content[:80].replace('\n', ' ')
                
                if score > 0.7:
                    status = "🟢 EXCELLENT"
                elif score > 0.6:
                    status = "🟡 GOOD"
                elif score > 0.5:
                    status = "🟠 FAIR"
                else:
                    status = "🔴 LOW"
                
                print(f"\n{status} '{term}': score {score:.3f}")
                print(f"   Document: {doc}")
                print(f"   Content: {content}...")
            else:
                print(f"\n❌ '{term}': No results found")
        except Exception as e:
            print(f"\n❌ '{term}': Error - {e}")
    
    print("\n" + "="*100 + "\n")


def test_specific_query():
    """Test the exact query from your user"""
    
    print("\n" + "="*100)
    print("🧪 TESTING YOUR SPECIFIC QUERY")
    print("="*100)
    
    from precomputed_rag import EnhancedPrecomputedRAGSystem as PrecomputedRAGSystem
    rag = PrecomputedRAGSystem()
    
    test_queries = [
        "maximum building coverage ratio WA zone Stuttgart-Mitte",
        "Grundflächenzahl WA Wohngebiet Stuttgart-Mitte",
        "how many stories allowed residential building Stuttgart",
        "Vollgeschosse Wohngebiet Stuttgart",
        "800m² lot building regulations Stuttgart"
    ]
    
    for query in test_queries:
        print(f"\n{'='*100}")
        print(f"Query: '{query}'")
        print(f"{'='*100}")
        
        results = rag.search(query, top_k=3)
        
        if not results:
            print("❌ NO RESULTS FOUND")
            continue
        
        for i, result in enumerate(results, 1):
            score = result.score
            doc_name = result.metadata.get('document_name', 'Unknown')
            page = result.metadata.get('page_number', 'N/A')
            content = result.content[:200].replace('\n', ' ')
            
            status = "🟢" if score > 0.7 else "🟡" if score > 0.6 else "🟠" if score > 0.5 else "🔴"
            
            print(f"\n{status} Result {i} (score: {score:.3f}):")
            print(f"   Document: {doc_name}")
            print(f"   Page: {page}")
            print(f"   Content: {content}...")
    
    print("\n" + "="*100 + "\n")

def emergency_rag_diagnostic():
    """Emergency diagnostic to see if RAG is working"""
    
    print("\n" + "="*100)
    print("🚨 EMERGENCY RAG DIAGNOSTIC")
    print("="*100)
    
    # Test 1: Can we initialize RAG?
    print("\n1️⃣ Testing RAG initialization...")
    try:
        from precomputed_rag import EnhancedPrecomputedRAGSystem as PrecomputedRAGSystem
        rag = PrecomputedRAGSystem()
        print("   ✅ RAG system initialized")
        
        # Check total documents
        if hasattr(rag, 'chunks'):
            print(f"   📚 Total chunks in system: {len(rag.chunks)}")
        elif hasattr(rag, 'vectorstore') and hasattr(rag.vectorstore, '_collection'):
            try:
                count = rag.vectorstore._collection.count()
                print(f"   📚 Total documents in vectorstore: {count}")
            except:
                print("   📚 Document count: Unknown")
                
    except Exception as e:
        print(f"   ❌ RAG initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Multiple search queries to test coverage
    print("\n2️⃣ Testing RAG search with multiple queries...")
    
    test_queries = [
        "Stuttgart building regulations",
        "Bauordnung Stuttgart",
        "LBO Baden-Württemberg",
        "Grundflächenzahl GRZ",
        "building coverage ratio WA zone"
    ]
    
    for query in test_queries:
        try:
            print(f"\n   🔍 Query: '{query}'")
            results = rag.search(query, top_k=2)
            
            if len(results) == 0:
                print(f"      ❌ No results found!")
                continue
            
            for i, result in enumerate(results, 1):
                doc_name = result.metadata.get('document_name', 'UNKNOWN')
                page = result.metadata.get('page_number', 'N/A')
                score = result.score
                content_preview = result.content[:80].replace('\n', ' ')
                
                # Color code by relevance
                if score > 0.8:
                    status = "🟢 EXCELLENT"
                elif score > 0.7:
                    status = "🟡 GOOD"
                elif score > 0.6:
                    status = "🟠 FAIR"
                else:
                    status = "🔴 LOW"
                
                print(f"      Result {i}: {status} (score: {score:.3f})")
                print(f"         Doc: {doc_name}")
                print(f"         Page: {page}")
                print(f"         Content: {content_preview}...")
                
        except Exception as e:
            print(f"      ❌ Search failed: {e}")
    
    # Test 3: Test the actual tool function
    print("\n3️⃣ Testing tool function with building-specific query...")
    try:
        query = "maximum building coverage ratio for WA zones in Stuttgart-Mitte"
        print(f"   Query: '{query}'")
        
        tool_result = search_regulations(query)
        
        print(f"\n   📋 Tool returned {len(tool_result)} characters")
        print(f"\n   First 500 characters of tool output:")
        print("   " + "-"*80)
        print(tool_result[:500])
        print("   " + "-"*80)
        
        # Check if we got good results
        if len(tool_result) > 200 and "Document 1" in tool_result:
            print("\n   ✅ Tool is returning structured results")
            
            # Check relevance scores in output
            if "0.8" in tool_result or "0.9" in tool_result:
                print("   🟢 High relevance scores detected!")
            elif "0.7" in tool_result:
                print("   🟡 Good relevance scores detected")
            elif "0.6" in tool_result or "0.5" in tool_result:
                print("   🟠 WARNING: Low relevance scores - documents may not match query well")
            
        else:
            print("   ⚠️ Tool output seems incomplete")
            
    except Exception as e:
        print(f"   ❌ Tool call failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Check agent configuration
    print("\n4️⃣ Testing agent configuration...")
    try:
        import os
        crew = StuttgartBuildingRegulationCrew(os.getenv("OPENAI_API_KEY"))
        doc_agent = crew.agents["document_specialist"]
        
        print(f"   Agent has {len(doc_agent.tools)} tools")
        for i, tool in enumerate(doc_agent.tools):
            print(f"   Tool {i+1}: {tool.name} - {tool.description[:60]}...")
        
        if len(doc_agent.tools) > 0:
            print("   ✅ Agent has tools configured")
        else:
            print("   ❌ Agent has NO tools!")
            return False
            
    except Exception as e:
        print(f"   ❌ Agent check failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*100)
    print("📊 DIAGNOSTIC SUMMARY:")
    print("   ✅ RAG system is functional")
    print("   ⚠️  Check relevance scores above - if mostly < 0.7, your documents may not")
    print("      contain specific building regulations, or embeddings need improvement")
    print("="*100 + "\n")
    
    return True

        
def main():
    """Test the system"""
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable required")
    
    # Use GPT-4o-mini (cost-effective) - set use_gpt4=True only if needed
    crew = StuttgartBuildingRegulationCrew(openai_api_key, use_gpt4=False)
    
    test_query = RegulationQuery(
        query="I have a 800m² lot in Stuttgart-Mitte zoned as WA. What's the maximum building coverage ratio and how many stories can I build?",
        project_type="Residential Building",
        location="Stuttgart",
        district="Stuttgart-Mitte"
    )
    
    result = crew.execute_analysis(test_query)
    print("="*80)
    print("ANALYSIS RESULT:")
    print("="*80)
    print(result)


if __name__ == "__main__":
    # Run inspection first
    inspect_rag_documents()
    test_specific_query()
    
    # Ask before running full analysis
    print("\n" + "="*100)
    response = input("Do you want to run the full CrewAI analysis? (y/n): ")
    if response.lower() == 'y':
        main()
    else:
        print("Exiting. Review the document inspection above.")