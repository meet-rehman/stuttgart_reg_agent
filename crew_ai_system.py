#!/usr/bin/env python3
"""
Multi-Agent Stuttgart Building Regulations System using CrewAI
ENHANCED VERSION with Vision AI for Landuse Plans and Plot Analysis
REFACTORED: Eliminated duplications, improved structure
"""

import os
import json
import base64
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool as LangChainTool
from langchain_openai import ChatOpenAI
from openai import OpenAI

# Custom imports
from precomputed_rag import EnhancedPrecomputedRAGSystem as PrecomputedRAGSystem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RegulationQuery:
    """Structure for regulation queries"""
    query: str
    project_type: str = "mixed-use"
    location: str = "Stuttgart"
    district: str = "general"
    urgency: str = "normal"
    plot_number: Optional[str] = None


@dataclass
class ComplianceFactor:
    """Structure for compliance cost factors"""
    keyword: str
    multiplier: float
    weeks: int
    description: str


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration"""
    
    # Vision AI settings
    VISION_MODEL = "gpt-4o"
    VISION_MAX_TOKENS = 1500
    VISION_TEMPERATURE = 0.1
    PLANS_DIRECTORY = "data/raw/Landuse Plans"
    
    # Search settings
    RAG_TOP_K = 3
    RAG_MAX_TOKENS = 1000
    
    # Image formats
    IMAGE_EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
    
    # Compliance factors - CONSOLIDATED (no duplicates)
    COMPLIANCE_FACTORS = [
        ComplianceFactor("accessibility", 1.12, 2, "DIN 18040 compliance / Barrier-free access"),
        ComplianceFactor("fire", 1.18, 3, "Fire protection systems"),
        ComplianceFactor("energy", 1.25, 4, "GEG/EnEV energy standards compliance"),
        ComplianceFactor("parking", 1.08, 1, "Parking requirements"),
        ComplianceFactor("setback", 1.03, 1, "Setback requirements"),
        ComplianceFactor("sound", 1.10, 2, "Sound/noise insulation DIN 4109"),
    ]
    
    # Regulatory hierarchy
    REGULATORY_KEYWORDS = {
        "federal": ["BauGB", "EnEV", "GEG", "DIN", "VDI", "Baugesetzbuch"],
        "state": ["LBO", "Baden-Württemberg", "BW", "Landesbauordnung"],
        "local": ["Stuttgart", "Zuffenhausen", "Municipal", "Stadt", "Mitte", "West"]
    }
    
    # Standard vision analysis template
    VISION_ANALYSIS_TEMPLATE = """You are an expert at reading German architectural site plans, 
Bebauungspläne, and landuse plans (Flächennutzungspläne).

Analyze this plan and answer: {query}

Please identify:
1. Plot numbers and boundaries (Flurstücknummern)
2. Zoning types (WA, MI, GE, etc.)
3. Building footprints and dimensions
4. Streets and access points (Straßen)
5. Setbacks and distances (Abstandsflächen)
6. Any measurements or annotations in German
7. Scale and orientation

Provide precise, measurable information in German and English.
If you cannot find specific information, say so clearly."""

    # Detailed plot analysis template
    DETAILED_PLOT_TEMPLATE = """For plot {plot_number}, provide a detailed analysis:

1. ZONING: What is the zoning type (WA, MI, GE, etc.)?
2. DIMENSIONS: What are the plot dimensions and area in square meters?
3. BOUNDARIES: Which streets/plots border this plot? (North, South, East, West)
4. ACCESS: How is the plot accessed? Which street(s)?
5. EXISTING STRUCTURES: Are there any existing buildings shown?
6. SPECIAL MARKINGS: Any special markings, restrictions, or annotations?
7. CONTEXT: What is the surrounding land use?

Be specific and cite measurements when visible."""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_tool_execution(tool_name: str, func: Callable, *args, **kwargs) -> str:
    """
    Unified error handling wrapper for all tool functions
    Eliminates repetitive try-except blocks
    """
    try:
        logger.info(f"🔧 Executing tool: {tool_name}")
        result = func(*args, **kwargs)
        logger.info(f"✅ Tool {tool_name} completed successfully")
        return result
    except Exception as e:
        logger.error(f"❌ Tool {tool_name} error: {e}", exc_info=True)
        return f"Error in {tool_name}: {str(e)}"


class SingletonManager:
    """
    Unified lazy loading for singleton instances
    Eliminates duplicate lazy loading patterns
    """
    _instances: Dict[str, Any] = {}
    
    @classmethod
    def get_instance(cls, key: str, factory: Callable) -> Any:
        """Get or create singleton instance"""
        if key not in cls._instances:
            cls._instances[key] = factory()
            logger.info(f"🔧 Created singleton: {key}")
        return cls._instances[key]


# ============================================================================
# VISION AI AGENT
# ============================================================================

class VisionAgent:
    """
    Vision AI agent for analyzing architectural drawings, site plans, and Bebauungspläne
    """
    
    def __init__(self, openai_api_key: str, plans_directory: str = None):
        self.client = OpenAI(api_key=openai_api_key)
        self.plans_dir = Path(plans_directory or Config.PLANS_DIRECTORY)
        self.model = Config.VISION_MODEL
        self._plan_cache = {}
        
        logger.info(f"🎨 Vision Agent initialized. Plans directory: {self.plans_dir}")
    
    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _get_available_plans(self) -> List[Path]:
        """Get all available plan images from images/ subdirectories"""
        if not self.plans_dir.exists():
            logger.warning(f"Plans directory not found: {self.plans_dir}")
            return []
        
        plans = []
        for ext in Config.IMAGE_EXTENSIONS:
            # Search in images/ subdirectories (created by PDF converter)
            plans.extend(self.plans_dir.rglob(f"images/{ext}"))
            # Also search root level
            plans.extend(self.plans_dir.rglob(ext))
        
        # Remove duplicates
        plans = list(set(plans))
        logger.info(f"📐 Found {len(plans)} plan images")
        return plans
    
    def analyze_plan(self, plan_path: Path, query: str, analysis_template: str = None) -> str:
        """
        Analyze a specific plan image using GPT-4 Vision
        
        Args:
            plan_path: Path to the plan image
            query: What to analyze in the plan
            analysis_template: Custom template (uses default if None)
            
        Returns:
            Analysis result as text
        """
        template = analysis_template or Config.VISION_ANALYSIS_TEMPLATE
        
        # Encode image
        image_data = self._encode_image(plan_path)
        
        # Call GPT-4 Vision
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": template.format(query=query)},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=Config.VISION_MAX_TOKENS,
            temperature=Config.VISION_TEMPERATURE
        )
        
        result = response.choices[0].message.content
        logger.info(f"✅ Analyzed plan: {plan_path.name}")
        return result
    
    def find_plot(self, plot_number: str) -> Dict[str, Any]:
        """
        Search all plans for a specific plot number
        
        Args:
            plot_number: Plot number to find (e.g., "9232/79")
            
        Returns:
            Dict with found plan and analysis
        """
        logger.info(f"🔍 Searching for plot {plot_number}")
        
        available_plans = self._get_available_plans()
        
        if not available_plans:
            return {
                'found': False,
                'error': 'No plan images available',
                'searched_directory': str(self.plans_dir)
            }
        
        # Search through plans
        for plan_path in available_plans:
            logger.info(f"   Checking: {plan_path.name}")
            
            query = f"Does this plan show plot number {plot_number} (Flurstück {plot_number})? If yes, describe its location, boundaries, and any visible information about it."
            analysis = self.analyze_plan(plan_path, query)
            
            # Check if plot was found
            if plot_number in analysis or "yes" in analysis.lower()[:100]:
                logger.info(f"✅ Found plot {plot_number} in {plan_path.name}")
                
                return {
                    'found': True,
                    'plan_file': plan_path.name,
                    'plan_path': str(plan_path),
                    'analysis': analysis
                }
        
        logger.warning(f"❌ Plot {plot_number} not found in any plans")
        return {
            'found': False,
            'searched_plans': len(available_plans),
            'message': f'Plot {plot_number} not found in {len(available_plans)} available plans'
        }
    
    def analyze_plot_requirements(self, plot_number: str) -> str:
        """
        Comprehensive analysis of a specific plot
        
        Args:
            plot_number: Plot number to analyze
            
        Returns:
            Detailed analysis text
        """
        # Find the plot first
        plot_info = self.find_plot(plot_number)
        
        if not plot_info['found']:
            return f"Could not locate plot {plot_number} in available plans. {plot_info.get('message', '')}"
        
        # Detailed analysis using template
        plan_path = Path(plot_info['plan_path'])
        detailed_query = Config.DETAILED_PLOT_TEMPLATE.format(plot_number=plot_number)
        detailed_analysis = self.analyze_plan(plan_path, detailed_query)
        
        return f"""PLOT ANALYSIS FOR {plot_number}
Plan Source: {plot_info['plan_file']}

{detailed_analysis}

Note: This analysis is based on visual inspection of the plan. 
Always verify with official cadastral documents."""


# ============================================================================
# RAG TOOLS (Text-based regulation search)
# ============================================================================

def _get_rag_system() -> PrecomputedRAGSystem:
    """Get or create RAG system singleton"""
    return SingletonManager.get_instance('rag_system', PrecomputedRAGSystem)


def search_regulations(query: str) -> str:
    """
    Search Stuttgart building regulations using RAG system
    Input: query string. Returns: relevant documents with citations
    """
    def _search():
        rag_system = _get_rag_system()
        logger.info(f"🔍 Searching regulations for: {query}")
        
        results = rag_system.search(query, top_k=Config.RAG_TOP_K)
        
        if not results:
            logger.warning(f"⚠️ No results found for: {query}")
            return f"No documents found for: {query}. Try broader search terms or check district spelling."
        
        logger.info(f"📊 Found {len(results)} results")
        
        formatted = []
        for i, result in enumerate(results, 1):
            metadata = result.metadata
            content = result.content[:250]
            score = result.score
            
            logger.info(f"  Result {i}: {metadata.get('document_name')} (score: {score:.3f})")
            
            formatted.append(f"""Document {i} (Relevance: {score:.2f}):
- Source: {metadata.get('document_name', 'Unknown')}
- Type: {metadata.get('document_type', 'Unknown')}
- Page: {metadata.get('page_number', 'N/A')}
- Citation: {result.get_detailed_citation()}
- Content: {content}...""")
        
        return "\n\n".join(formatted)
    
    return safe_tool_execution("search_regulations", _search)


def get_context(query: str) -> str:
    """
    Get comprehensive context for query
    Input: query. Returns: detailed context with citations
    """
    def _get_context():
        rag_system = _get_rag_system()
        logger.info(f"📖 Getting context for: {query}")
        
        context = rag_system.get_context_for_query(
            query,
            max_tokens=Config.RAG_MAX_TOKENS,
            include_citations=True
        )
        
        if not context or len(context.strip()) < 50:
            logger.warning(f"⚠️ Insufficient context for: {query}")
            return f"Limited context available for: {query}. Consider rephrasing or broadening the query."
        
        return context
    
    return safe_tool_execution("get_context", _get_context)


def analyze_hierarchy(regulations: str) -> str:
    """
    Analyze regulatory hierarchy
    Input: regulations text. Returns: hierarchy analysis
    """
    def _analyze():
        logger.info("⚖️ Analyzing regulatory hierarchy")
        
        found = {}
        for level, keywords in Config.REGULATORY_KEYWORDS.items():
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
    
    return safe_tool_execution("analyze_hierarchy", _analyze)


def estimate_costs(requirements: str) -> str:
    """
    Estimate compliance costs
    Input: requirements. Returns: cost and timeline
    """
    def _estimate():
        logger.info("💰 Estimating compliance costs")
        
        applicable = []
        total_mult = 1.0
        total_weeks = 0
        
        requirements_lower = requirements.lower()
        
        # Check each factor (no duplicates now)
        for factor in Config.COMPLIANCE_FACTORS:
            if factor.keyword in requirements_lower:
                applicable.append(factor)
                total_mult *= factor.multiplier
                total_weeks += factor.weeks
        
        if applicable:
            factor_list = [
                f"• {f.description}: +{(f.multiplier-1)*100:.1f}% cost, +{f.weeks}w" 
                for f in applicable
            ]
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
    
    return safe_tool_execution("estimate_costs", _estimate)


# ============================================================================
# VISION TOOLS (Image-based plan analysis)
# ============================================================================

def _get_vision_agent() -> VisionAgent:
    """Get or create vision agent singleton"""
    return SingletonManager.get_instance(
        'vision_agent',
        lambda: VisionAgent(os.getenv("OPENAI_API_KEY"))
    )


def search_plot_in_plans(plot_number: str) -> str:
    """
    Search for a specific plot in landuse plans
    Input: plot number (e.g., '9232/79'). Returns: plan location and basic info
    """
    def _search():
        vision_agent = _get_vision_agent()
        logger.info(f"🔍 Vision tool: Searching for plot {plot_number}")
        
        result = vision_agent.find_plot(plot_number)
        
        if result['found']:
            return f"""Plot {plot_number} found in plan: {result['plan_file']}

Analysis:
{result['analysis']}"""
        else:
            return f"""Plot {plot_number} not found in available plans.
Searched {result.get('searched_plans', 0)} plan images.
Plans directory: {Config.PLANS_DIRECTORY}

Suggestion: Verify plot number spelling or check if plans are available."""
    
    return safe_tool_execution("search_plot_in_plans", _search)


def analyze_plot_details(plot_number: str) -> str:
    """
    Get detailed analysis of a specific plot from plans
    Input: plot number. Returns: comprehensive plot information
    """
    def _analyze():
        vision_agent = _get_vision_agent()
        logger.info(f"📐 Vision tool: Analyzing plot {plot_number} in detail")
        
        return vision_agent.analyze_plot_requirements(plot_number)
    
    return safe_tool_execution("analyze_plot_details", _analyze)


def analyze_general_plan(query: str) -> str:
    """
    Analyze landuse plans for general questions (not plot-specific)
    Input: general query about plans. Returns: analysis of relevant plans
    """
    def _analyze():
        vision_agent = _get_vision_agent()
        logger.info(f"🗺️ Vision tool: General plan analysis - {query}")
        
        plans = vision_agent._get_available_plans()
        
        if not plans:
            return "No landuse plans available for analysis."
        
        # Analyze the first plan (TODO: use embeddings for best match)
        plan_path = plans[0]
        analysis = vision_agent.analyze_plan(plan_path, query)
        
        return f"""Analysis of {plan_path.name}:

{analysis}

Note: Analyzed {plan_path.name}. There are {len(plans)} total plans available."""
    
    return safe_tool_execution("analyze_general_plan", _analyze)


# ============================================================================
# MAIN CREW SYSTEM
# ============================================================================

class StuttgartBuildingRegulationCrew:
    """Multi-agent regulation analysis system with Vision AI"""
    
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
            openai_api_key=openai_api_key
        )
        
        logger.info(f"🤖 Initializing crew with model: {model}")
        
        # Initialize vision agent
        try:
            self.vision_agent = VisionAgent(openai_api_key)
            logger.info("✅ Vision agent initialized")
        except Exception as e:
            logger.warning(f"⚠️ Vision agent initialization failed: {e}")
            self.vision_agent = None
        
        # Create agents
        self.agents = self._create_agents()
        
        logger.info("✅ Crew system initialized")
    
    def _create_agents(self) -> Dict[str, Agent]:
        """Create specialized agents with their tools"""
        
        # Define all tools
        tools = {
            'text_search': LangChainTool(
                name="search_regulations",
                description="Search Stuttgart building regulations from text documents. Use for finding rules, requirements, and regulations.",
                func=search_regulations
            ),
            'context': LangChainTool(
                name="get_context",
                description="Get detailed context about regulations with citations.",
                func=get_context
            ),
            'hierarchy': LangChainTool(
                name="analyze_hierarchy",
                description="Analyze regulatory hierarchy (federal/state/local).",
                func=analyze_hierarchy
            ),
            'cost': LangChainTool(
                name="estimate_costs",
                description="Estimate compliance costs and timeline.",
                func=estimate_costs
            ),
            'plot_search': LangChainTool(
                name="search_plot_in_plans",
                description="Search for a specific plot number in landuse plans. Use when user mentions a plot number (e.g., '9232/79').",
                func=search_plot_in_plans
            ),
            'plot_analysis': LangChainTool(
                name="analyze_plot_details",
                description="Get detailed analysis of a specific plot from visual plans (zoning, dimensions, boundaries). Use after finding a plot.",
                func=analyze_plot_details
            ),
            'general_plan': LangChainTool(
                name="analyze_general_plan",
                description="Analyze landuse plans for general questions about area, zoning, or layout (not plot-specific).",
                func=analyze_general_plan
            )
        }
        
        # DOCUMENT & VISUAL SPECIALIST
        document_specialist = Agent(
            role="Document & Visual Plan Specialist",
            goal="Find relevant building regulations from documents AND analyze site plans visually",
            backstory="""Expert at searching German building regulation documents and reading 
architectural plans, Bebauungspläne, and landuse plans. Can identify plots, measure distances, 
and extract information from visual plans.""",
            llm=self.llm,
            verbose=True,
            tools=[tools['text_search'], tools['context'], tools['plot_search'], tools['plot_analysis']],
            allow_delegation=False
        )
        
        # LEGAL ANALYST
        legal_analyst = Agent(
            role="Regulatory Legal Analyst",
            goal="Interpret regulatory hierarchy and legal requirements",
            backstory="Legal expert in German building law hierarchy (BauGB, LBO, local regulations).",
            llm=self.llm,
            verbose=True,
            tools=[tools['hierarchy']],
            allow_delegation=False
        )
        
        # COMPLIANCE EXPERT
        compliance_expert = Agent(
            role="Compliance & Cost Analyst",
            goal="Estimate compliance costs and requirements",
            backstory="Building compliance expert specializing in cost estimation and timeline planning.",
            llm=self.llm,
            verbose=True,
            tools=[tools['cost']],
            allow_delegation=False
        )
        
        return {
            "document_specialist": document_specialist,
            "legal_analyst": legal_analyst,
            "compliance_expert": compliance_expert
        }
    
    def execute_analysis(self, query: RegulationQuery) -> str:
        """
        Execute multi-agent analysis
        
        Args:
            query: RegulationQuery with optional plot_number
            
        Returns:
            Comprehensive analysis result
        """
        logger.info(f"🎯 Executing analysis for: {query.query}")
        
        # Detect if this is a plot-specific query
        is_plot_query = query.plot_number is not None or any(
            keyword in query.query.lower() 
            for keyword in ['plot', 'flurstück', 'grundstück', 'parzelle', '/']
        )
        
        if is_plot_query:
            logger.info("📍 Detected plot-specific query")
        
        # TASK 1: Document & Visual Research
        research_task = Task(
            description=f"""Research the following building regulation query:
"{query.query}"

Project Details:
- Type: {query.project_type}
- Location: {query.location}
- District: {query.district}
{'- Plot Number: ' + query.plot_number if query.plot_number else ''}

Instructions:
1. If a plot number is mentioned, FIRST use search_plot_in_plans tool
2. If plot is found, use analyze_plot_details for comprehensive plot analysis
3. Search text regulations using search_regulations tool
4. For general area questions, use analyze_general_plan
5. Combine visual plan analysis with text regulations

Provide detailed findings with sources.""",
            agent=self.agents["document_specialist"],
            expected_output="Detailed research findings with citations and visual plan analysis if applicable"
        )
        
        # TASK 2: Legal Analysis
        legal_task = Task(
            description=f"""Analyze the regulatory hierarchy and legal requirements for:
"{query.query}"

Based on the research findings, determine:
1. Which regulatory levels apply (federal/state/local)
2. Precedence and conflicts
3. Legal interpretations

Focus on Stuttgart regulations and Baden-Württemberg building code.""",
            agent=self.agents["legal_analyst"],
            expected_output="Clear hierarchy analysis with legal interpretations"
        )
        
        # TASK 3: Compliance & Costs
        compliance_task = Task(
            description=f"""Based on the requirements found, estimate:
1. Compliance costs and timeline
2. Special requirements and their impact
3. Recommendations for compliance strategy

Consider the project type: {query.project_type}""",
            agent=self.agents["compliance_expert"],
            expected_output="Cost estimates and compliance recommendations"
        )
        
        # Create and run crew
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=[research_task, legal_task, compliance_task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            result = crew.kickoff()
            logger.info("✅ Crew analysis complete")
            return str(result)
        except Exception as e:
            logger.error(f"❌ Crew execution error: {e}", exc_info=True)
            return f"Error during analysis: {str(e)}"


# ============================================================================
# TESTING / MAIN
# ============================================================================

if __name__ == "__main__":
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable required")
    
    # Test vision agent
    print("\n" + "="*80)
    print("🧪 TESTING VISION AGENT")
    print("="*80)
    
    vision_agent = VisionAgent(openai_api_key)
    
    # Test plot search
    test_plot = "9232/79"  # Example plot number
    print(f"\n🔍 Searching for plot: {test_plot}")
    result = vision_agent.find_plot(test_plot)
    print(json.dumps(result, indent=2))
    
    # Test full crew with plot query
    print("\n" + "="*80)
    print("🧪 TESTING FULL CREW WITH PLOT QUERY")
    print("="*80)
    
    crew = StuttgartBuildingRegulationCrew(openai_api_key, use_gpt4=False)
    
    test_query = RegulationQuery(
        query="What can I build on plot 9232/79? What are the setback requirements and maximum building coverage?",
        project_type="Residential Building",
        location="Stuttgart",
        district="Nordbahnhof",
        plot_number="9232/79"
    )
    
    result = crew.execute_analysis(test_query)
    print("\n" + "="*80)
    print("ANALYSIS RESULT:")
    print("="*80)
    print(result)