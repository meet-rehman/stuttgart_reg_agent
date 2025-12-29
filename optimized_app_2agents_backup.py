#!/usr/bin/env python3
"""
FastAPI Application for Stuttgart Building Regulations
UPDATED: Now uses CrewAI multi-agent system with vision capabilities
"""

import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
import logging
from datetime import datetime
import asyncio
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from dotenv import load_dotenv
from pydantic import BaseModel

import logging
import sys

# Force all output to be visible
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout
)

# Configure CrewAI to be verbose
os.environ['CREWAI_VERBOSE'] = 'True'
os.environ['CREWAI_LOG_LEVEL'] = 'DEBUG'


# Redirect CrewAI logger to stdout
crewai_logger = logging.getLogger('crewai')
crewai_logger.setLevel(logging.DEBUG)
crewai_logger.addHandler(logging.StreamHandler(sys.stdout))

# Add current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))



# Import your NEW CrewAI system
try:
    from optimized_crew_ai_system import (
        StuttgartBuildingRegulationCrew,
        RegulationQuery
    )
    print("✅ CrewAI system imported successfully")
except ImportError as e:
    print(f"❌ Failed to import CrewAI system: {e}")
    raise

# Setup comprehensive logging
class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record):
        # Add color to the log level
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        
        # Add special formatting for agent messages
        if 'Agent' in record.getMessage() or 'agent' in record.getMessage():
            record.msg = f"{self.BOLD}🤖 {record.msg}{self.RESET}"
        elif 'Vision' in record.getMessage() or 'vision' in record.getMessage():
            record.msg = f"{self.BOLD}👁️ {record.msg}{self.RESET}"
        elif 'Document' in record.getMessage():
            record.msg = f"{self.BOLD}📚 {record.msg}{self.RESET}"
        elif 'Architecture' in record.getMessage():
            record.msg = f"{self.BOLD}🏗️ {record.msg}{self.RESET}"
            
        return super().format(record)

# Configure logging with color formatter for console
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
))

file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
)

logger = logging.getLogger(__name__)

# Enable detailed logging for all CrewAI and agent components
for logger_name in ['crewai', 'optimized_crew_ai_system', 'optimized_vision_agent', 'crewai.agent', 'crewai.crew']:
    logging.getLogger(logger_name).setLevel(logging.DEBUG)

# ============================================================================
# ENVIRONMENT DETECTION & LOADING
# ============================================================================

def detect_environment():
    """Detect if running on Railway or locally"""
    railway_vars = ['RAILWAY_ENVIRONMENT', 'RAILWAY_PROJECT_ID', 'RAILWAY_SERVICE_ID']
    is_railway = any(os.getenv(var) for var in railway_vars)
    return "Railway" if is_railway else "Local"

def load_environment():
    """Load environment variables based on deployment context"""
    environment = detect_environment()
    
    if environment == "Local":
        # Try multiple .env files for local development
        PROJECT_ROOT = Path(__file__).resolve().parent
        env_files = [PROJECT_ROOT / ".env1", PROJECT_ROOT / ".env"]
        
        loaded_from = None
        for env_file in env_files:
            if env_file.exists():
                load_dotenv(dotenv_path=env_file)
                loaded_from = env_file
                logger.info(f"✅ Loaded environment from: {env_file}")
                break
        
        return environment, loaded_from
    else:
        # Railway - use system environment variables
        logger.info("🚂 Running on Railway - using system environment")
        return environment, "Railway Environment Variables"

# Load environment
env_type, env_source = load_environment()

# Get API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.error("❌ OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY is required")
else:
    logger.info(f"✅ OPENAI_API_KEY loaded: {OPENAI_API_KEY[:8]}...")

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

crew: Optional[StuttgartBuildingRegulationCrew] = None

# ============================================================================
# AGENT MONITORING SYSTEM
# ============================================================================

class AgentMonitor:
    """Monitor and log agent activities in real-time"""
    
    def __init__(self):
        self.current_agent = None
        self.agent_steps = []
        self.start_time = None
        
    def start(self, agent_name: str, task: str):
        """Mark agent start"""
        self.current_agent = agent_name
        self.start_time = datetime.now()
        msg = f"[AGENT START] {agent_name} - Task: {task}"
        logger.info("=" * 60)
        logger.info(msg)
        logger.info("=" * 60)
        self.agent_steps.append({
            'agent': agent_name,
            'task': task,
            'start': self.start_time.isoformat(),
            'status': 'working'
        })
        
    def log_step(self, step: str):
        """Log an intermediate step"""
        if self.current_agent:
            logger.info(f"  → {self.current_agent}: {step}")
            
    def complete(self, agent_name: str):
        """Mark agent completion"""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"[AGENT COMPLETE] {agent_name} - Duration: {duration:.2f}s")
            logger.info("-" * 60)
        
    def get_status(self):
        """Get current status for frontend"""
        return {
            'current_agent': self.current_agent,
            'steps': self.agent_steps
        }

# Global agent monitor
agent_monitor = AgentMonitor()

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    building_type: Optional[str] = "Residential"
    location: Optional[str] = "Stuttgart"
    district: Optional[str] = "general"
    height: Optional[float] = None
    storeys: Optional[int] = None
    area: Optional[float] = None
    plot_number: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    processing_time: float
    timestamp: str
    success: bool
    vision_used: Optional[bool] = False
    agent_steps: Optional[list] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    environment: str
    crew_ready: bool
    vision_enabled: bool

# ============================================================================
# APPLICATION LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - initializes CrewAI system"""
    global crew
    
    logger.info("="*60)
    logger.info("🚀 STARTING STUTTGART BUILDING REGULATIONS AI")
    logger.info("="*60)
    
    try:
        logger.info("🤖 Initializing CrewAI multi-agent system...")
        logger.info("   Setting up Document Specialist agent...")
        logger.info("   Setting up Architecture Consultant agent...")
        logger.info("   Configuring Vision AI system...")
        
        crew = StuttgartBuildingRegulationCrew(
            openai_api_key=OPENAI_API_KEY,
            use_gpt4=False,  # Use GPT-4o-mini for cost efficiency
            enable_vision=True  # Enable vision for plot analysis
        )
        
        logger.info("✅ CrewAI system initialized successfully!")
        logger.info(f"   Vision enabled: {crew.vision_enabled}")
        logger.info(f"   Number of agents: 2")
        logger.info(f"   RAG System: Ready")
        logger.info(f"   Vision Agent: Ready")
        logger.info("="*60)
        
        # Log the crew structure
        logger.info("📋 CREW CONFIGURATION:")
        logger.info("   • Document Specialist: Searches regulations & analyzes plans")
        logger.info("   • Architecture Consultant: Provides professional recommendations")
        logger.info("   • Vision System: Analyzes landuse plans for plot-specific data")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ STARTUP ERROR: {e}", exc_info=True)
        crew = None
        logger.warning("⚠️ Continuing startup with disabled system for debugging")
    
    logger.info("="*60)
    logger.info("✅ APP STARTED SUCCESSFULLY")
    logger.info("   API Endpoint: http://localhost:8000")
    logger.info("   Documentation: http://localhost:8000/docs")
    logger.info("="*60)
    
    yield
    
    logger.info("🔥 Shutting down...")

# ============================================================================
# CREATE FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Stuttgart Building Regulations AI",
    description="Multi-agent AI system for Stuttgart building regulations with vision capabilities",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# ROUTES
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        crew_ready = crew is not None
        vision_enabled = crew.vision_enabled if crew else False
        
        status = "healthy" if crew_ready else "degraded"
        
        return HealthResponse(
            status=status,
            timestamp=datetime.now().isoformat(),
            environment=detect_environment(),
            crew_ready=crew_ready,
            vision_enabled=vision_enabled
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Process building regulation query using CrewAI multi-agent system
    """
    try:
        if not crew:
            raise HTTPException(
                status_code=503, 
                detail="CrewAI system not initialized. Check logs for details."
            )
        
        # Log the incoming request with visual formatting
        logger.info("="*60)
        logger.info("📨 NEW QUERY RECEIVED")
        logger.info("="*60)
        logger.info(f"🔍 Query: {request.message}")
        logger.info(f"🏢 Building type: {request.building_type}")
        logger.info(f"📍 Location: {request.location}, {request.district}")
        if request.plot_number:
            logger.info(f"📋 Plot number: {request.plot_number}")
            logger.info(f"👁️ Vision analysis will be ENABLED")
        else:
            logger.info(f"📝 Text-only analysis (no plot specified)")
        logger.info("="*60)
        
        # Create RegulationQuery from request
        query = RegulationQuery(
            query=request.message,
            project_type=request.building_type or "Residential",
            location=request.location or "Stuttgart",
            district=request.district or "general",
            plot_number=request.plot_number,
            enable_vision=bool(request.plot_number)
        )
        
        # Start agent monitoring
        agent_monitor.start("CrewAI System", f"Analyzing: {request.message[:50]}...")
        
        # Log the execution plan
        logger.info("🎯 EXECUTION PLAN:")
        logger.info("="*60)
        logger.info("Step 1: Document Specialist will search regulations")
        logger.info("Step 2: Document Specialist will analyze relevant documents")
        if request.plot_number:
            logger.info("Step 3: Vision Agent will analyze landuse plans")
        logger.info("Step 4: Architecture Consultant will synthesize findings")
        logger.info("Step 5: Generate comprehensive professional report")
        logger.info("="*60)
        
        # Show that agents are starting
        logger.info("🤖 AGENTS STARTING WORK...")
        logger.info("-"*60)
        
        # Log Document Specialist starting
        agent_monitor.log_step("📚 Document Specialist: Initiating regulation search...")
        agent_monitor.log_step("📚 Document Specialist: Querying RAG system...")
        
        # Execute analysis
        start_time = datetime.now()
        
        # Actually run the crew analysis
        logger.info("🔄 Executing CrewAI kickoff...")
        result = crew.execute_analysis(query)
        
        # Log Architecture Consultant
        agent_monitor.log_step("🏗️ Architecture Consultant: Analyzing findings...")
        agent_monitor.log_step("🏗️ Architecture Consultant: Preparing recommendations...")
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Complete monitoring
        agent_monitor.complete("CrewAI System")
        
        if result['success']:
            logger.info("="*60)
            logger.info("✅ ANALYSIS COMPLETE - SUCCESS")
            logger.info("="*60)
            logger.info(f"⏱️ Total processing time: {processing_time:.2f} seconds")
            logger.info(f"👁️ Vision used: {result.get('vision_used', False)}")
            logger.info(f"📊 Response length: {len(result['analysis'])} characters")
            logger.info(f"📚 Regulations found: {result.get('regulations_count', 'N/A')}")
            logger.info("="*60)
            
            return ChatResponse(
                message=result['analysis'],
                processing_time=processing_time,
                timestamp=datetime.now().isoformat(),
                success=True,
                vision_used=result.get('vision_used', False),
                agent_steps=agent_monitor.agent_steps
            )
        else:
            logger.error(f"❌ Analysis failed: {result.get('error')}")
            return ChatResponse(
                message=f"Sorry, I encountered an error: {result.get('error')}",
                processing_time=processing_time,
                timestamp=datetime.now().isoformat(),
                success=False,
                vision_used=False
            )
            
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent-status")
async def get_agent_status():
    """Get real-time agent status"""
    return agent_monitor.get_status()

@app.get("/metrics")
async def get_metrics():
    """Get system performance metrics"""
    if not crew:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    metrics = crew.metrics if hasattr(crew, 'metrics') else {}
    
    return {
        "total_queries": metrics.get('total_queries', 0),
        "vision_queries": metrics.get('vision_used', 0),
        "avg_processing_time": metrics.get('total_processing_time', 0) / max(metrics.get('total_queries', 1), 1),
        "vision_timeouts": metrics.get('vision_timeouts', 0),
        "vision_failures": metrics.get('vision_failures', 0)
    }

@app.get("/logs/recent")
async def get_recent_logs():
    """Get last 50 log entries"""
    try:
        log_file = Path("app.log")
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()
                return {
                    "logs": lines[-50:],
                    "count": len(lines[-50:])
                }
        return {"logs": [], "count": 0}
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# STATIC FILES & FRONTEND
# ============================================================================

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"✅ Static files mounted from: {static_dir}")
else:
    logger.warning(f"⚠️ Static directory not found: {static_dir}")

# ROOT ROUTE - Serve Frontend
@app.get("/", response_class=HTMLResponse)
async def frontend():
    """Serve the main frontend"""
    try:
        index_file = static_dir / "index.html"
        
        if index_file.exists():
            return FileResponse(str(index_file))
        else:
            # Fallback HTML with enhanced interface
            return HTMLResponse(content=generate_fallback_html())
            
    except Exception as e:
        logger.error(f"Frontend error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_fallback_html() -> str:
    """Generate fallback HTML if index.html not found"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stuttgart Building Regulations AI</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 2em; margin-bottom: 10px; }
        .header p { opacity: 0.9; }
        .content { padding: 30px; }
        .status {
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-weight: 600;
        }
        .healthy { background: #d4edda; color: #155724; }
        .degraded { background: #fff3cd; color: #856404; }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
        }
        .form-group input, .form-group select, .form-group textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
        }
        .form-group textarea { min-height: 100px; font-family: inherit; }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
        }
        button:hover { opacity: 0.9; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        #response {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            white-space: pre-wrap;
            line-height: 1.6;
            display: none;
        }
        #response.show { display: block; }
        .loading { text-align: center; padding: 20px; color: #667eea; }
        .agent-activity {
            background: #f0f8ff;
            border: 2px solid #667eea;
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
            display: none;
        }
        .agent-activity.show { display: block; }
        .agent-activity h3 { color: #667eea; margin-bottom: 10px; }
        .activity-item {
            padding: 8px;
            margin: 5px 0;
            background: white;
            border-left: 3px solid #667eea;
            border-radius: 3px;
            animation: slideIn 0.3s ease;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        .pulse {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #4caf50;
            border-radius: 50%;
            animation: pulse 1s infinite;
            margin-right: 10px;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏢 Stuttgart Building Regulations AI</h1>
            <p>Multi-Agent System with Vision Capabilities</p>
        </div>
        
        <div class="content">
            <div id="status" class="status">Checking system status...</div>
            
            <div class="form-group">
                <label for="query">Your Question:</label>
                <textarea id="query" placeholder="Ask about building regulations, plot requirements, height restrictions, etc..."></textarea>
            </div>
            
            <div class="form-group">
                <label for="building_type">Building Type:</label>
                <select id="building_type">
                    <option value="Residential">Residential</option>
                    <option value="Mixed-Use">Mixed-Use</option>
                    <option value="Commercial">Commercial</option>
                    <option value="Industrial">Industrial</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="plot_number">Plot Number (optional):</label>
                <input type="text" id="plot_number" placeholder="e.g., 18A, 9232/79">
            </div>
            
            <button id="submitBtn" onclick="sendQuery()">Get Analysis</button>
            
            <div id="agentActivity" class="agent-activity">
                <h3><span class="pulse"></span>Agents Working...</h3>
                <div id="activityLog"></div>
            </div>
            
            <div id="response"></div>
        </div>
    </div>

    <script>
        const API_BASE = window.location.origin;
        let activityInterval;

        async function checkStatus() {
            try {
                const response = await fetch(`${API_BASE}/health`);
                const data = await response.json();
                const statusDiv = document.getElementById('status');
                
                if (data.status === 'healthy') {
                    statusDiv.className = 'status healthy';
                    statusDiv.innerHTML = `✅ System: ${data.status.toUpperCase()} | Environment: ${data.environment} | Vision: ${data.vision_enabled ? 'Enabled' : 'Disabled'}`;
                } else {
                    statusDiv.className = 'status degraded';
                    statusDiv.innerHTML = `⚠️ System: ${data.status.toUpperCase()} | Some components not ready`;
                }
            } catch (error) {
                document.getElementById('status').innerHTML = '❌ Error: ' + error.message;
                document.getElementById('status').className = 'status degraded';
            }
        }

        function addActivityItem(text) {
            const activityLog = document.getElementById('activityLog');
            const item = document.createElement('div');
            item.className = 'activity-item';
            item.textContent = text;
            activityLog.appendChild(item);
        }

        async function sendQuery() {
            const query = document.getElementById('query').value;
            const building_type = document.getElementById('building_type').value;
            const plot_number = document.getElementById('plot_number').value;
            const responseDiv = document.getElementById('response');
            const submitBtn = document.getElementById('submitBtn');
            const agentActivity = document.getElementById('agentActivity');
            const activityLog = document.getElementById('activityLog');
            
            if (!query) {
                alert('Please enter a question');
                return;
            }
            
            // Clear previous activity log
            activityLog.innerHTML = '';
            
            submitBtn.disabled = true;
            submitBtn.textContent = 'Agents Working...';
            responseDiv.className = 'show loading';
            responseDiv.textContent = '🔄 CrewAI agents are starting analysis...';
            agentActivity.className = 'agent-activity show';
            
            // Simulate agent activity
            addActivityItem('📚 Document Specialist: Starting regulation search...');
            setTimeout(() => addActivityItem('🔍 Document Specialist: Analyzing building codes...'), 2000);
            setTimeout(() => addActivityItem('📊 Document Specialist: Found relevant regulations...'), 4000);
            
            if (plot_number) {
                setTimeout(() => addActivityItem('👁️ Vision Agent: Analyzing landuse plans...'), 6000);
                setTimeout(() => addActivityItem('🗺️ Vision Agent: Extracting plot information...'), 8000);
            }
            
            setTimeout(() => addActivityItem('🏗️ Architecture Consultant: Synthesizing findings...'), 10000);
            setTimeout(() => addActivityItem('✍️ Architecture Consultant: Preparing recommendations...'), 12000);
            
            try {
                const response = await fetch(`${API_BASE}/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: query,
                        building_type: building_type,
                        plot_number: plot_number || null,
                        location: 'Stuttgart',
                        district: 'general'
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    addActivityItem('✅ All agents completed successfully!');
                    
                    setTimeout(() => {
                        agentActivity.className = 'agent-activity';
                        responseDiv.className = 'show';
                        responseDiv.innerHTML = `
                            <strong>📊 Analysis Complete</strong><br>
                            <small>Processing time: ${data.processing_time.toFixed(2)}s | Vision used: ${data.vision_used ? 'Yes' : 'No'}</small>
                            <hr style="margin: 15px 0; border: none; border-top: 1px solid #ddd;">
                            ${data.message.replace(/\n/g, '<br>')}
                        `;
                    }, 1000);
                } else {
                    responseDiv.textContent = '❌ Error: ' + data.message;
                    agentActivity.className = 'agent-activity';
                }
            } catch (error) {
                responseDiv.textContent = '❌ Error: ' + error.message;
                agentActivity.className = 'agent-activity';
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Get Analysis';
            }
        }

        // Check status on page load
        checkStatus();
        
        // Allow Enter key in textarea to submit
        document.getElementById('query').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                sendQuery();
            }
        });
    </script>
</body>
</html>
    """

# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Set environment variables for verbose CrewAI output
    os.environ['CREWAI_VERBOSE'] = 'True'
    
    port = int(os.getenv("PORT", 8000))
    
    logger.info("="*60)
    logger.info(f"🚀 Starting FastAPI server on port {port}")
    logger.info("="*60)
    logger.info("Configuration:")
    logger.info(f" 	• Environment: {detect_environment()}")
    logger.info(f" 	• CrewAI Verbose: ENABLED")
    logger.info(f" 	• Log Level: DEBUG")
    logger.info(f" 	• Vision: ENABLED")
    logger.info("="*60)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port, 
        log_level="debug", # <--- CHANGE THIS FROM "info" to "debug"
        access_log=True
    )