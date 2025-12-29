#!/usr/bin/env python3
"""
Updated schemas for multi-agent Stuttgart Building Regulations system
Includes both legacy and new multi-agent request/response models
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime

# =================================================================
# LEGACY SCHEMAS (Backward Compatibility)
# =================================================================

class ChatRequest(BaseModel):
    """Legacy chat request model"""
    message: str = Field(..., description="User's building regulation question")
    conversation_id: Optional[str] = Field(None, description="Optional conversation tracking ID")

class ChatResponse(BaseModel):
    """Legacy chat response model"""
    message: str = Field(..., description="AI response to the query")
    timestamp: str = Field(..., description="Response timestamp")
    context_used: int = Field(..., description="Number of document contexts used")
    conversation_id: Optional[str] = Field(None, description="Conversation tracking ID")

# =================================================================
# MULTI-AGENT SCHEMAS (New System)
# =================================================================

class ProjectType(str):
    """Enum for project types"""
    MIXED_USE = "mixed-use"
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    OFFICE = "office"
    RENOVATION = "renovation"

class StuttgartDistrict(str):
    """Enum for Stuttgart districts"""
    GENERAL = "general"
    ZUFFENHAUSEN = "Zuffenhausen"
    MITTE = "Stuttgart-Mitte"
    WEST = "Stuttgart-West"
    OST = "Stuttgart-Ost"
    NORD = "Stuttgart-Nord"
    SUED = "Stuttgart-Süd"

class AnalysisUrgency(str):
    """Enum for analysis urgency levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class MultiAgentRequest(BaseModel):
    """Request model for multi-agent analysis"""
    query: str = Field(
        ..., 
        description="Detailed building regulation question",
        min_length=10,
        max_length=2000,
        example="What are the complete requirements for building a mixed-use development with residential and commercial space in Stuttgart?"
    )
    project_type: str = Field(
        default="mixed-use",
        description="Type of building project"
    )
    location: str = Field(
        default="Stuttgart",
        description="Project location (currently supports Stuttgart)"
    )
    district: str = Field(
        default="general",
        description="Specific Stuttgart district if applicable"
    )
    urgency: str = Field(
        default="normal",
        description="Analysis urgency level"
    )
    use_multi_agent: bool = Field(
        default=True,
        description="Whether to use multi-agent analysis (vs single agent)"
    )
    include_cost_analysis: bool = Field(
        default=True,
        description="Include cost and timeline analysis"
    )
    include_alternatives: bool = Field(
        default=True,
        description="Include alternative compliance approaches"
    )

class AgentExecution(BaseModel):
    """Model for individual agent execution details"""
    agent_name: str = Field(..., description="Name of the agent")
    role: str = Field(..., description="Agent's role/specialization")
    execution_time: Optional[float] = Field(None, description="Time taken by this agent")
    tokens_used: Optional[int] = Field(None, description="Tokens consumed by this agent")
    status: Literal["success", "partial", "failed"] = Field(..., description="Execution status")
    key_findings: List[str] = Field(default=[], description="Key findings from this agent")

class ComplianceAssessment(BaseModel):
    """Model for compliance assessment results"""
    compliance_level: Literal["full", "partial", "requires_review", "non_compliant"] = Field(
        ..., description="Overall compliance assessment"
    )
    critical_requirements: List[str] = Field(default=[], description="Must-have requirements")
    optional_requirements: List[str] = Field(default=[], description="Recommended but not mandatory")
    cost_estimate: Optional[Dict[str, Any]] = Field(None, description="Cost estimation breakdown")
    timeline_estimate: Optional[Dict[str, Any]] = Field(None, description="Timeline estimation")
    risk_factors: List[str] = Field(default=[], description="Identified risk factors")

class DocumentCitation(BaseModel):
    """Model for document citations"""
    document_name: str = Field(..., description="Name of the cited document")
    section: Optional[str] = Field(None, description="Specific section or paragraph")
    page_number: Optional[int] = Field(None, description="Page number")
    legal_reference: Optional[str] = Field(None, description="Legal reference (e.g., § 34 LBO BW)")
    category: str = Field(..., description="Document category (federal/state/local)")
    relevance_score: Optional[float] = Field(None, description="Relevance score (0-1)")

class MultiAgentResponse(BaseModel):
    """Comprehensive response model for multi-agent analysis"""
    
    # Core Analysis
    analysis: str = Field(..., description="Complete multi-agent analysis report")
    executive_summary: Optional[str] = Field(None, description="Executive summary of key points")
    
    # Metadata
    timestamp: str = Field(..., description="Analysis completion timestamp")
    query_details: Dict[str, Any] = Field(..., description="Original query parameters")
    
    # Performance Metrics
    processing_time: Optional[float] = Field(None, description="Total processing time in seconds")
    agents_used: List[str] = Field(default=[], description="List of agents that participated")
    agent_executions: Optional[List[AgentExecution]] = Field(None, description="Detailed agent execution info")
    
    # Analysis Components
    compliance_assessment: Optional[ComplianceAssessment] = Field(None, description="Compliance assessment results")
    document_citations: List[DocumentCitation] = Field(default=[], description="All documents cited in analysis")
    
    # Structured Recommendations
    immediate_actions: List[str] = Field(default=[], description="Immediate actions required")
    next_steps: List[str] = Field(default=[], description="Recommended next steps")
    alternative_approaches: List[str] = Field(default=[], description="Alternative compliance approaches")
    
    # Quality Indicators
    confidence_score: Optional[float] = Field(None, description="Overall confidence in analysis (0-1)")
    completeness_score: Optional[float] = Field(None, description="Completeness of regulatory coverage (0-1)")
    
    # Additional Metadata
    analysis_version: str = Field(default="2.0.0", description="Multi-agent system version")
    fallback_used: bool = Field(default=False, description="Whether fallback to single-agent was used")

# =================================================================
# HEALTH CHECK AND SYSTEM STATUS SCHEMAS
# =================================================================

class SystemComponent(BaseModel):
    """Model for system component status"""
    name: str = Field(..., description="Component name")
    status: Literal["ready", "degraded", "unavailable", "error"] = Field(..., description="Component status")
    last_check: Optional[str] = Field(None, description="Last health check timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional component details")

class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Overall system status")
    timestamp: str = Field(..., description="Health check timestamp")
    
    # Core System Status
    multi_agent_ready: bool = Field(..., description="Multi-agent system readiness")
    document_database_ready: bool = Field(..., description="Document database readiness")
    
    # Component Status
    components: List[SystemComponent] = Field(default=[], description="Individual component statuses")
    
    # Agent Status
    agents: List[str] = Field(default=[], description="Available agent list")
    agent_count: int = Field(default=0, description="Number of active agents")
    
    # System Metrics
    total_documents: Optional[int] = Field(None, description="Total documents in system")
    system_uptime: Optional[float] = Field(None, description="System uptime in seconds")
    memory_usage: Optional[Dict[str, float]] = Field(None, description="Memory usage statistics")

# =================================================================
# ERROR HANDLING SCHEMAS
# =================================================================

class ErrorDetail(BaseModel):
    """Detailed error information"""
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Human-readable error message")
    error_type: str = Field(..., description="Error type/category")
    timestamp: str = Field(..., description="Error occurrence timestamp")
    agent_context: Optional[str] = Field(None, description="Which agent caused the error")
    traceback: Optional[str] = Field(None, description="Error traceback (development only)")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: ErrorDetail = Field(..., description="Error details")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    fallback_available: bool = Field(default=False, description="Whether fallback options are available")
    retry_recommended: bool = Field(default=False, description="Whether retry is recommended")

# =================================================================
# VALIDATION AND HELPER FUNCTIONS
# =================================================================

def validate_stuttgart_query(query: str) -> bool:
    """Validate if query is relevant to Stuttgart building regulations"""
    stuttgart_keywords = [
        "stuttgart", "baden-württemberg", "lbo bw", "zuffenhausen",
        "building", "construction", "permit", "regulation", "code"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in stuttgart_keywords)

def create_error_response(error_message: str, error_type: str = "general_error") -> ErrorResponse:
    """Helper function to create standardized error responses"""
    return ErrorResponse(
        error=ErrorDetail(
            error_code=f"ERR_{error_type.upper()}",
            error_message=error_message,
            error_type=error_type,
            timestamp=datetime.now().isoformat()
        )
    )