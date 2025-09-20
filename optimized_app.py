import sys
import os
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from coordinator_agent import CoordinatorAgent

# Add current directory to Python path for local imports
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Import local modules (NOT stuttgart_reg_agent.*)
try:
    from schemas import (
        BuildingSearchRequest, 
        BuildingSearchResponse, 
        ChatRequest, 
        ChatResponse,
        HealthResponse
    )
    from tools.groq_client import GroqClient
    from precomputed_rag import PrecomputedRAGSystem
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current directory: {current_dir}")
    print("Files in current directory:")
    for f in current_dir.glob("*"):
        print(f"  {f.name}")
    raise

# Environment detection and loading
def detect_environment():
    """Detect if running on Railway or locally"""
    railway_vars = ['RAILWAY_ENVIRONMENT', 'RAILWAY_PROJECT_ID', 'RAILWAY_SERVICE_ID', 'RAILWAY_DEPLOYMENT_ID']
    is_railway = any(os.getenv(var) for var in railway_vars)
    return "Railway" if is_railway else "Local"

def load_environment():
    """Load environment variables based on deployment context"""
    environment = detect_environment()
    
    if environment == "Local":
        # Load from .env files for local development
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        env_files = [PROJECT_ROOT / ".env1", PROJECT_ROOT / ".env"]
        
        loaded_from = None
        for env_file in env_files:
            if env_file.exists():
                load_dotenv(dotenv_path=env_file)
                loaded_from = env_file
                break
        
        return environment, loaded_from
    else:
        # Railway environment - variables should be available directly
        return environment, "Railway Environment Variables"

# Load environment
env_type, env_source = load_environment()
print(f"Environment: {env_type}")
if env_source:
    print(f"Loaded from: {env_source}")

# Get API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not found in environment variables")
    available_vars = [k for k in os.environ.keys() if any(term in k.upper() for term in ['GROQ', 'API', 'KEY'])]
    print(f"Available API-related variables: {available_vars}")
else:
    print(f"GROQ_API_KEY loaded: {GROQ_API_KEY[:8]}...")

# Global variables
groq_client: Optional[GroqClient] = None
rag_system: Optional[PrecomputedRAGSystem] = None
coordinator: Optional[CoordinatorAgent] = None  # Add coordinator here

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with detailed error debugging"""
    global groq_client, rag_system, coordinator  # Add coordinator to global
    
    print("="*50)
    print("STARTING LIFESPAN INITIALIZATION")
    print("="*50)
    
    try:
        print("Step 1: Starting Stuttgart Building Agent...")
        
        # Initialize GroqClient
        print("Step 2: Initializing Groq client...")
        if GROQ_API_KEY:
            groq_client = GroqClient(
                api_key=GROQ_API_KEY,
                api_url=GROQ_API_URL
            )
            print("✅ Groq client initialized successfully")
        else:
            print("⚠️ WARNING: Groq client not initialized - missing API key")
        
        # Initialize RAG system with extensive debugging
        print("Step 3: Initializing RAG system...")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Embeddings directory exists: {(Path(__file__).parent / 'embeddings').exists()}")
        
        try:
            rag_system = PrecomputedRAGSystem()
            print("✅ RAG system object created")
        except Exception as e:
            print(f"❌ Failed to create RAG system object: {e}")
            raise
        
        print("Step 4: Calling RAG system initialize...")
        try:
            # RAG system initializes itself in __init__ - no separate initialize() call needed
            print("✅ RAG system initialize() completed")
        except Exception as e:
            print(f"❌ RAG system initialize() failed: {e}")
            print(f"Exception type: {type(e).__name__}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            raise
        
        print("Step 5: Checking RAG system readiness...")
        try:
            is_ready = rag_system.is_ready  # Remove () - it's a property, not a method
            print(f"RAG system is_ready: {is_ready}")
            if not is_ready:
                print("⚠️ RAG system reports not ready after initialization")
            else:
                print("✅ RAG system ready!")
        except Exception as e:
            print(f"❌ Error checking RAG readiness: {e}")
        
        # NEW: Initialize coordinator AFTER RAG system is ready
        print("Step 6: Initializing coordinator...")
        try:
            coordinator = CoordinatorAgent(rag_system)
            print("✅ Coordinator initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize coordinator: {e}")
            raise
        
        print("="*50)
        print("✅ APP STARTED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        print("="*50)
        print(f"❌ STARTUP ERROR: {e}")
        print(f"Error type: {type(e).__name__}")
        print("="*50)
        import traceback
        traceback.print_exc()
        
        # Set systems to None so health checks can detect the failure
        groq_client = None
        rag_system = None
        coordinator = None  # Add this
        
        # Don't raise - let the app start so we can see the error via health endpoint
        print("⚠️ Continuing startup with disabled systems for debugging")
    
    yield
    
    print("🔥 Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Stuttgart Building Registration Agent",
    description="AI Agent for Stuttgart building registration process",
    version="1.0.0",
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


@app.get("/debug/rag-status")
async def debug_rag_status():
    """Debug endpoint to check RAG system status"""
    if not rag_system:
        return {"error": "RAG system not initialized"}
    
    return {
        "is_ready": rag_system.is_ready,
        "embeddings_dir": str(rag_system.embeddings_dir),
        "documents_count": len(rag_system.documents),
        "embeddings_shape": rag_system.embeddings.shape if rag_system.embeddings is not None else None,
        "embeddings_dir_exists": rag_system.embeddings_dir.exists(),
        "files_in_embeddings_dir": [f.name for f in rag_system.embeddings_dir.glob("*")] if rag_system.embeddings_dir.exists() else []
    }


    
# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        groq_ready = groq_client is not None and GROQ_API_KEY is not None
        rag_ready = rag_system is not None and rag_system.is_ready  # Remove () - it's a property
        
        status = "healthy" if (groq_ready and rag_ready) else "degraded"
        
        return HealthResponse(
            status=status,
            timestamp=datetime.now().isoformat(),
            environment=detect_environment(),
            api_ready=groq_ready,
            rag_ready=rag_ready,
            components={
                "groq_client": "ready" if groq_ready else "not ready",
                "rag_system": "ready" if rag_ready else "not ready"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Building search endpoint
@app.post("/search", response_model=BuildingSearchResponse)
async def search_buildings(request: BuildingSearchRequest):
    """Search for buildings based on user query"""
    try:
        if not rag_system:
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        results = rag_system.search(  # Remove await - it's not async
            query=request.query,
            top_k=request.top_k
        )
        
        return BuildingSearchResponse(
            results=results,
            query=request.query,
            total_results=len(results),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Chat endpoint

# Replace your chat endpoint in optimized_app.py with this enhanced version:

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Enhanced PropTech chat with detailed citations"""
    try:
        if not groq_client:
            raise HTTPException(status_code=503, detail="Groq client not initialized")
        if not rag_system:
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        if not coordinator:
            raise HTTPException(status_code=503, detail="Coordinator not initialized")

        # Detect district mentions for targeted search
        stuttgart_districts = ['Zuffenhausen', 'Feuerbach', 'Weilimdorf', 'Bad Cannstatt', 'Stammheim']
        mentioned_district = None
        for district in stuttgart_districts:
            if district.lower() in request.message.lower():
                mentioned_district = district
                break

        # Use coordinator for intelligent routing
        routing = coordinator.route(request.message)
        
        # Get enhanced context with citations
        if mentioned_district:
            context_results = rag_system.search_by_district(request.message, mentioned_district, top_k=4)
            context = f"DISTRICT-SPECIFIC RESULTS FOR {mentioned_district.upper()}:\n\n"
            context += rag_system.get_context_for_query(request.message, max_tokens=2500, include_citations=True)
        else:
            context = rag_system.get_context_for_query(request.message, max_tokens=2500, include_citations=True)

        # Enhanced system prompt for PropTech focus
        system_context = f"""You are a specialized Stuttgart Building Regulations Assistant for PropTech applications.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using the provided Stuttgart documents below
2. ALWAYS include specific citations (document name, page, section)
3. Include form numbers, official IDs, and legal references when available
4. Highlight district-specific rules when relevant
5. If the answer isn't in the documents, state: "This information is not available in the current Stuttgart regulations database."

Agent Context: {routing['agent']} | Task: {routing['task']}

STUTTGART BUILDING REGULATIONS DATABASE:
{context}

FORMAT YOUR RESPONSE AS:
1. Direct answer with specific citations
2. Relevant forms/documents needed
3. District-specific considerations (if applicable)
4. Legal references and section numbers
"""

        full_prompt = f"{system_context}\n\nUser Question: {request.message}\n\nDetailed Response:"

        groq_response = await groq_client.complete_async(
            prompt=full_prompt,
            max_tokens=800,  # Increased for detailed responses
            temperature=0.1
        )

        if 'choices' in groq_response and len(groq_response['choices']) > 0:
            response_text = groq_response['choices'][0]['message']['content']
        else:
            response_text = "I'm sorry, I couldn't generate a response at this time."

        # Count context sources for metadata
        context_sources = len(context.split("[Source")) - 1 if "[Source" in context else 0

        return ChatResponse(
            message=response_text,
            timestamp=datetime.now().isoformat(),
            context_used=context_sources,
            conversation_id=getattr(request, 'conversation_id', None)
        )

    except Exception as e:
        print(f"Enhanced chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

# Add new endpoints for PropTech functionality
@app.get("/forms/{process_type}")
async def get_forms_for_process(process_type: str):
    """Get relevant forms for a building process"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    forms = rag_system.get_forms_for_process(process_type)
    
    return {
        "process_type": process_type,
        "relevant_forms": forms,
        "total_found": len(forms)
    }

@app.get("/districts/{district_name}/regulations")
async def get_district_regulations(district_name: str):
    """Get district-specific regulations"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    # Search for district-specific content
    results = rag_system.search_by_district("building regulations requirements", district_name, top_k=5)
    
    return {
        "district": district_name,
        "regulations": [
            {
                "content_preview": result.content[:300] + "...",
                "citation": result.get_detailed_citation(),
                "score": result.score,
                "document_type": result.metadata.get("document_type", "Unknown")
            }
            for result in results
        ],
        "total_found": len(results)
    }


# Static files and frontend
try:
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
except Exception as e:
    print(f"Warning: Could not mount static files: {e}")

@app.get("/", response_class=HTMLResponse)
async def frontend():
    """Serve the frontend"""
    try:
        static_dir = Path(__file__).parent / "static"
        index_file = static_dir / "index.html"
        
        if index_file.exists():
            return HTMLResponse(content=index_file.read_text(encoding='utf-8'))
        else:
            # Simple fallback HTML
            return HTMLResponse(content="""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Stuttgart Building Agent</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                    .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
                    .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
                    .healthy { background: #d4edda; color: #155724; }
                    .error { background: #f8d7da; color: #721c24; }
                    button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
                    button:hover { background: #0056b3; }
                    #response { margin-top: 20px; padding: 15px; background: white; border-radius: 5px; white-space: pre-wrap; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Stuttgart Building Registration Agent</h1>
                    <div id="status" class="status">Checking system status...</div>
                    
                    <h2>Test the API</h2>
                    <input type="text" id="query" placeholder="Ask about building registration..." style="width: 60%; padding: 10px;">
                    <button onclick="sendQuery()">Send</button>
                    
                    <div id="response"></div>
                </div>

                <script>
                    const API_BASE = window.location.origin;

                    async function checkStatus() {
                        try {
                            const response = await fetch(`${API_BASE}/health`);
                            const data = await response.json();
                            const statusDiv = document.getElementById('status');
                            
                            if (data.status === 'healthy') {
                                statusDiv.className = 'status healthy';
                                statusDiv.innerHTML = `System Status: ${data.status.toUpperCase()}<br>
                                    Environment: ${data.environment}<br>
                                    API Ready: ${data.api_ready ? 'Yes' : 'No'}<br>
                                    RAG Ready: ${data.rag_ready ? 'Yes' : 'No'}`;
                            } else {
                                statusDiv.className = 'status error';
                                statusDiv.innerHTML = `System Status: ${data.status.toUpperCase()}<br>
                                    Some components may not be ready.`;
                            }
                        } catch (error) {
                            document.getElementById('status').innerHTML = 'Error checking status: ' + error.message;
                            document.getElementById('status').className = 'status error';
                        }
                    }

                    async function sendQuery() {
                        const query = document.getElementById('query').value;
                        const responseDiv = document.getElementById('response');
                        
                        if (!query) return;
                        
                        responseDiv.textContent = 'Processing...';
                        
                        try {
                            const response = await fetch(`${API_BASE}/chat`, {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ message: query })
                            });
                            
                            const data = await response.json();
                            responseDiv.textContent = data.message;
                        } catch (error) {
                            responseDiv.textContent = 'Error: ' + error.message;
                        }
                    }

                    // Check status on page load
                    checkStatus();
                </script>
            </body>
            </html>
            """)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Frontend error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Railway sets $PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
