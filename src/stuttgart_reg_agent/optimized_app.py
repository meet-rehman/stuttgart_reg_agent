# src/stuttgart_reg_agent/optimized_app.py

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import yaml
from dotenv import load_dotenv

from stuttgart_reg_agent.schemas import (
    PlotDetails,
    ZoningRecommendation,
    BuildingCodeCompliance,
    AccessibilityAnalysis,
)
from stuttgart_reg_agent.tools.groq_client import GroqClient
from stuttgart_reg_agent.optimized_rag import OptimizedRAGSystem

# -----------------------------------------------------------------------------
# Load environment
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_FILE = PROJECT_ROOT / ".env1"

if DOTENV_FILE.exists():
    load_dotenv(dotenv_path=DOTENV_FILE)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")

# Directories
CONFIG_DIR = Path(__file__).parent / "config"
DATA_DIR = PROJECT_ROOT / "data"
MEMORY_DIR = PROJECT_ROOT / "memory"

# Global variables for app state
groq_client = None
rag_system = None
tasks_config = None

# -----------------------------------------------------------------------------
# Startup and Shutdown
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app startup and shutdown"""
    global groq_client, rag_system, tasks_config
    
    print("🚀 Starting Stuttgart Building Agent...")
    
    # Initialize Groq client
    groq_client = GroqClient(
        api_key=GROQ_API_KEY, 
        api_url=GROQ_API_URL, 
        model="llama-3.1-8b-instant"  # Faster model
    )
    
    # Load tasks configuration
    with open(CONFIG_DIR / "tasks.yaml", "r", encoding="utf-8") as f:
        tasks_config = yaml.safe_load(f)
    
    # Initialize RAG system (async)
    rag_system = OptimizedRAGSystem(
        data_dir=DATA_DIR,
        memory_dir=MEMORY_DIR
    )
    
    # Load existing embeddings or create new ones in background
    await rag_system.initialize()
    
    print("✅ App started successfully!")
    
    yield  # App is running
    
    print("🛑 Shutting down...")

# -----------------------------------------------------------------------------
# Input/Output Models
# -----------------------------------------------------------------------------
class PlotInput(BaseModel):
    location: str
    size_m2: float
    building_type: str
    floors: int
    height_m: float

class QuestionInput(BaseModel):
    question: str

class AnalysisResponse(BaseModel):
    zoning: Optional[ZoningRecommendation] = None
    building_code: Optional[BuildingCodeCompliance] = None
    accessibility: Optional[AccessibilityAnalysis] = None
    processing_time_seconds: float
    status: str

class QuestionResponse(BaseModel):
    answer: str
    sources: List[str]
    processing_time_seconds: float

# -----------------------------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Stuttgart Building Agent",
    description="Fast API for Stuttgart building regulations analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def build_optimized_prompt(task_name: str, task_spec: dict, plot_data: dict, schema_model) -> str:
    """Build an optimized prompt for faster processing"""
    
    # Get relevant context from RAG system
    context = ""
    if rag_system and rag_system.is_ready:
        relevant_docs = rag_system.search(f"{task_name} {plot_data.get('location', '')} {plot_data.get('building_type', '')}")
        context = "\n".join(relevant_docs[:3])  # Limit context to avoid token limits
    
    # Create schema template
    schema_fields = {k: str(v.annotation) for k, v in schema_model.model_fields.items()}
    example_json = {}
    
    for k, typ in schema_fields.items():
        typ_lower = typ.lower()
        if "bool" in typ_lower:
            example_json[k] = False
        elif "list" in typ_lower:
            example_json[k] = []
        elif "int" in typ_lower or "float" in typ_lower:
            example_json[k] = 0
        else:
            example_json[k] = "<value>"
    
    prompt = f"""You are a Stuttgart building regulations expert. Analyze the plot data and provide a response.

Task: {task_name}
Description: {task_spec.get('description', '')}

Plot Data:
{json.dumps(plot_data, indent=2)}

{"Relevant Regulations:" if context else ""}
{context}

Return ONLY a valid JSON object matching this schema:
{json.dumps(schema_fields, indent=2)}

Example format:
{json.dumps(example_json, indent=2)}

Response:"""
    
    return prompt

async def process_task_async(task_name: str, plot_data: dict) -> dict:
    """Process a single task asynchronously"""
    task_spec = tasks_config.get(task_name, {})
    
    # Map task to schema
    model_map = {
        "analyze_zoning": ZoningRecommendation,
        "check_building_code": BuildingCodeCompliance,
        "assess_accessibility": AccessibilityAnalysis,
    }
    
    model = model_map.get(task_name)
    if not model:
        return {"error": f"Unknown task: {task_name}"}
    
    try:
        prompt = build_optimized_prompt(task_name, task_spec, plot_data, model)
        
        # Use async HTTP client for non-blocking calls
        response = await asyncio.to_thread(
            groq_client.complete, 
            prompt, 
            max_tokens=400,  # Reduced for faster response
            temperature=0.1
        )
        
        # Extract response
        if "choices" in response and response["choices"]:
            content = response["choices"][0].get("message", {}).get("content", "")
            
            # Parse JSON
            try:
                # Extract JSON from response
                start = content.find("{")
                end = content.rfind("}") + 1
                if start != -1 and end > start:
                    json_str = content[start:end]
                    data = json.loads(json_str)
                    validated = model.model_validate(data)
                    return validated.model_dump()
            except Exception as e:
                return {"error": f"JSON parsing error: {str(e)}", "raw_response": content}
        
        return {"error": "No response from AI"}
        
    except Exception as e:
        return {"error": str(e)}

# Frontend HTML
FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stuttgart Building Regulations Assistant</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🏢</text></svg>">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary-color: #2563eb;
            --primary-hover: #1d4ed8;
            --success-color: #059669;
            --error-color: #dc2626;
            --warning-color: #d97706;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --border-color: #e2e8f0;
            --shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-color);
            color: var(--text-primary);
            line-height: 1.6;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 40px 20px;
            background: linear-gradient(135deg, var(--primary-color), #3b82f6);
            color: white;
            border-radius: 16px;
            box-shadow: var(--shadow-lg);
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 700;
        }

        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }

        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-top: 20px;
            padding: 8px 16px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            font-size: 0.9rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }

        .card {
            background: var(--card-bg);
            border-radius: 16px;
            padding: 30px;
            box-shadow: var(--shadow);
            border: 1px solid var(--border-color);
            transition: all 0.3s ease;
        }

        .card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }

        .card h2 {
            font-size: 1.5rem;
            margin-bottom: 20px;
            color: var(--primary-color);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .icon {
            width: 24px;
            height: 24px;
            fill: currentColor;
        }

        .form-group {
            margin-bottom: 20px;
        }

        .form-group label {
            display: block;
            margin-bottom: 6px;
            font-weight: 500;
            color: var(--text-primary);
        }

        .form-group input, 
        .form-group select,
        .form-group textarea {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid var(--border-color);
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.3s ease;
            background: white;
        }

        .form-group input:focus,
        .form-group select:focus,
        .form-group textarea:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }

        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }

        .btn {
            padding: 14px 28px;
            font-size: 1rem;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            min-height: 48px;
        }

        .btn-primary {
            background: var(--primary-color);
            color: white;
        }

        .btn-primary:hover:not(:disabled) {
            background: var(--primary-hover);
            transform: translateY(-1px);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid transparent;
            border-top: 2px solid currentColor;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .results {
            background: var(--card-bg);
            border-radius: 16px;
            padding: 30px;
            box-shadow: var(--shadow);
            border: 1px solid var(--border-color);
            grid-column: 1 / -1;
        }

        .results h2 {
            color: var(--primary-color);
            margin-bottom: 20px;
        }

        .result-section {
            margin-bottom: 25px;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid var(--primary-color);
        }

        .result-section h3 {
            margin-bottom: 15px;
            color: var(--text-primary);
        }

        .result-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid var(--border-color);
        }

        .result-item:last-child {
            border-bottom: none;
        }

        .status-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }

        .status-compliant {
            background: rgba(5, 150, 105, 0.1);
            color: var(--success-color);
        }

        .status-non-compliant {
            background: rgba(220, 38, 38, 0.1);
            color: var(--error-color);
        }

        .violations-list, .recommendations-list {
            list-style: none;
            margin-top: 10px;
        }

        .violations-list li, .recommendations-list li {
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 6px;
            background: rgba(220, 38, 38, 0.05);
            border-left: 3px solid var(--error-color);
        }

        .recommendations-list li {
            background: rgba(5, 150, 105, 0.05);
            border-left-color: var(--success-color);
        }

        .processing-time {
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-top: 15px;
        }

        .error-message {
            background: rgba(220, 38, 38, 0.1);
            color: var(--error-color);
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border: 1px solid rgba(220, 38, 38, 0.2);
        }

        .question-answer {
            background: rgba(37, 99, 235, 0.05);
            padding: 20px;
            border-radius: 8px;
            margin-top: 15px;
            border: 1px solid rgba(37, 99, 235, 0.1);
        }

        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
                gap: 20px;
            }
            
            .form-row {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .container {
                padding: 15px;
            }
            
            .card {
                padding: 20px;
            }
        }

        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Stuttgart Building Regulations Assistant</h1>
            <p>Get instant analysis of zoning, building codes, and accessibility compliance</p>
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span id="apiStatus">Checking API status...</span>
            </div>
        </div>

        <div class="main-content">
            <!-- Plot Analysis Form -->
            <div class="card">
                <h2>
                    <svg class="icon" viewBox="0 0 24 24">
                        <path d="M3 21h18v-2H3v2zM3 8v6h18V8H3zm0-3v2h18V5H3z"/>
                    </svg>
                    Plot Analysis
                </h2>
                <form id="analysisForm">
                    <div class="form-group">
                        <label for="location">Location</label>
                        <select id="location" required>
                            <option value="">Select location</option>
                            <option value="Stuttgart-Mitte">Stuttgart-Mitte</option>
                            <option value="Stuttgart-Nord">Stuttgart-Nord</option>
                            <option value="Stuttgart-Süd">Stuttgart-Süd</option>
                            <option value="Stuttgart-Ost">Stuttgart-Ost</option>
                            <option value="Stuttgart-West">Stuttgart-West</option>
                        </select>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label for="size">Size (m²)</label>
                            <input type="number" id="size" step="0.1" min="1" required>
                        </div>
                        <div class="form-group">
                            <label for="buildingType">Building Type</label>
                            <select id="buildingType" required>
                                <option value="">Select type</option>
                                <option value="residential">Residential</option>
                                <option value="commercial">Commercial</option>
                                <option value="mixed-use">Mixed-use</option>
                                <option value="industrial">Industrial</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label for="floors">Number of Floors</label>
                            <input type="number" id="floors" min="1" max="20" required>
                        </div>
                        <div class="form-group">
                            <label for="height">Height (m)</label>
                            <input type="number" id="height" step="0.1" min="1" required>
                        </div>
                    </div>
                    
                    <button type="submit" class="btn btn-primary" id="analyzeBtn">
                        <span id="analyzeBtnText">Analyze Plot</span>
                        <div class="spinner hidden" id="analyzeSpinner"></div>
                    </button>
                </form>
            </div>

            <!-- Question Form -->
            <div class="card">
                <h2>
                    <svg class="icon" viewBox="0 0 24 24">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 17h-2v-2h2v2zm2.07-7.75l-.9.92C13.45 12.9 13 13.5 13 15h-2v-.5c0-1.1.45-2.1 1.17-2.83l1.24-1.26c.37-.36.59-.86.59-1.41 0-1.1-.9-2-2-2s-2 .9-2 2H8c0-2.21 1.79-4 4-4s4 1.79 4 4c0 .88-.36 1.68-.93 2.25z"/>
                    </svg>
                    Ask Questions
                </h2>
                <form id="questionForm">
                    <div class="form-group">
                        <label for="question">Your Question</label>
                        <textarea 
                            id="question" 
                            rows="4" 
                            placeholder="Ask about building regulations, height limits, accessibility requirements, etc."
                            required
                        ></textarea>
                    </div>
                    
                    <button type="submit" class="btn btn-primary" id="questionBtn">
                        <span id="questionBtnText">Ask Question</span>
                        <div class="spinner hidden" id="questionSpinner"></div>
                    </button>
                </form>
                
                <div id="questionResult" class="hidden">
                    <div class="question-answer">
                        <h4>Answer:</h4>
                        <div id="answerText"></div>
                        <div class="processing-time" id="questionTime"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Results Section -->
        <div id="results" class="results hidden">
            <h2>Analysis Results</h2>
            
            <div id="zoningResult" class="result-section">
                <h3>Zoning Analysis</h3>
                <div id="zoningContent"></div>
            </div>
            
            <div id="buildingCodeResult" class="result-section">
                <h3>Building Code Compliance</h3>
                <div id="buildingCodeContent"></div>
            </div>
            
            <div id="accessibilityResult" class="result-section">
                <h3>Accessibility Analysis</h3>
                <div id="accessibilityContent"></div>
            </div>
            
            <div class="processing-time" id="processingTime"></div>
        </div>
    </div>

    <script>
        const API_BASE = 'http://127.0.0.1:8000';
        
        // Check API status on load
        async function checkApiStatus() {
            try {
                const response = await fetch(`${API_BASE}/health`);
                const data = await response.json();
                
                document.getElementById('apiStatus').textContent = 
                    `API Ready • ${data.documents_loaded || 0} documents loaded`;
                
                if (!data.rag_ready) {
                    document.getElementById('apiStatus').textContent += ' • RAG indexing...';
                }
            } catch (error) {
                document.getElementById('apiStatus').textContent = 'API Connection Failed';
                document.querySelector('.status-dot').style.background = '#ef4444';
            }
        }

        // Plot Analysis Form Handler
        document.getElementById('analysisForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const analyzeBtn = document.getElementById('analyzeBtn');
            const analyzeBtnText = document.getElementById('analyzeBtnText');
            const analyzeSpinner = document.getElementById('analyzeSpinner');
            
            // Show loading state
            analyzeBtn.disabled = true;
            analyzeBtnText.textContent = 'Analyzing...';
            analyzeSpinner.classList.remove('hidden');
            
            // Hide previous results
            document.getElementById('results').classList.add('hidden');
            
            try {
                const formData = {
                    location: document.getElementById('location').value,
                    size_m2: parseFloat(document.getElementById('size').value),
                    building_type: document.getElementById('buildingType').value,
                    floors: parseInt(document.getElementById('floors').value),
                    height_m: parseFloat(document.getElementById('height').value)
                };
                
                const response = await fetch(`${API_BASE}/analyze`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData)
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const result = await response.json();
                displayAnalysisResults(result);
                
            } catch (error) {
                console.error('Analysis error:', error);
                displayError('Failed to analyze plot. Please check if the API server is running.');
            } finally {
                // Reset button state
                analyzeBtn.disabled = false;
                analyzeBtnText.textContent = 'Analyze Plot';
                analyzeSpinner.classList.add('hidden');
            }
        });

        // Question Form Handler
        document.getElementById('questionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const questionBtn = document.getElementById('questionBtn');
            const questionBtnText = document.getElementById('questionBtnText');
            const questionSpinner = document.getElementById('questionSpinner');
            
            // Show loading state
            questionBtn.disabled = true;
            questionBtnText.textContent = 'Searching...';
            questionSpinner.classList.remove('hidden');
            
            // Hide previous answer
            document.getElementById('questionResult').classList.add('hidden');
            
            try {
                const question = document.getElementById('question').value;
                
                const response = await fetch(`${API_BASE}/ask`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ question: question })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const result = await response.json();
                displayQuestionResult(result);
                
            } catch (error) {
                console.error('Question error:', error);
                displayError('Failed to get answer. Please check if the API server is running.');
            } finally {
                // Reset button state
                questionBtn.disabled = false;
                questionBtnText.textContent = 'Ask Question';
                questionSpinner.classList.add('hidden');
            }
        });

        function displayAnalysisResults(result) {
            // Show results section
            document.getElementById('results').classList.remove('hidden');
            
            // Display zoning results
            if (result.zoning) {
                const zoningContent = document.getElementById('zoningContent');
                zoningContent.innerHTML = `
                    <div class="result-item">
                        <span>Allowed Building Type:</span>
                        <strong>${result.zoning.allowed_building_type || 'N/A'}</strong>
                    </div>
                    <div class="result-item">
                        <span>Max Floors:</span>
                        <strong>${result.zoning.max_floors || 'N/A'}</strong>
                    </div>
                    <div class="result-item">
                        <span>Max Height:</span>
                        <strong>${result.zoning.max_height_m ? result.zoning.max_height_m + 'm' : 'N/A'}</strong>
                    </div>
                    ${result.zoning.notes ? `<div class="result-item"><span>Notes:</span><span>${result.zoning.notes}</span></div>` : ''}
                `;
            }
            
            // Display building code results
            if (result.building_code) {
                const buildingCodeContent = document.getElementById('buildingCodeContent');
                const isCompliant = result.building_code.compliant;
                
                buildingCodeContent.innerHTML = `
                    <div class="result-item">
                        <span>Compliance Status:</span>
                        <span class="status-badge ${isCompliant ? 'status-compliant' : 'status-non-compliant'}">
                            ${isCompliant ? 'Compliant' : 'Non-Compliant'}
                        </span>
                    </div>
                    ${result.building_code.violations && result.building_code.violations.length > 0 ? 
                        `<div class="result-item">
                            <span>Violations:</span>
                            <ul class="violations-list">
                                ${result.building_code.violations.map(v => `<li>${v}</li>`).join('')}
                            </ul>
                        </div>` : ''
                    }
                `;
            }
            
            // Display accessibility results
            if (result.accessibility) {
                const accessibilityContent = document.getElementById('accessibilityContent');
                const isCompliant = result.accessibility.compliant;
                
                accessibilityContent.innerHTML = `
                    <div class="result-item">
                        <span>Compliance Status:</span>
                        <span class="status-badge ${isCompliant ? 'status-compliant' : 'status-non-compliant'}">
                            ${isCompliant ? 'Compliant' : 'Needs Improvements'}
                        </span>
                    </div>
                    ${result.accessibility.recommendations && result.accessibility.recommendations.length > 0 ? 
                        `<div class="result-item">
                            <span>Recommendations:</span>
                            <ul class="recommendations-list">
                                ${result.accessibility.recommendations.map(r => `<li>${r}</li>`).join('')}
                            </ul>
                        </div>` : ''
                    }
                `;
            }
            
            // Display processing time
            if (result.processing_time_seconds) {
                document.getElementById('processingTime').textContent = 
                    `Analysis completed in ${result.processing_time_seconds} seconds`;
            }
        }

        function displayQuestionResult(result) {
            document.getElementById('questionResult').classList.remove('hidden');
            document.getElementById('answerText').innerHTML = result.answer || 'No answer available.';
            document.getElementById('questionTime').textContent = 
                `Answered in ${result.processing_time_seconds || 0} seconds`;
        }

        function displayError(message) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message';
            errorDiv.textContent = message;
            
            // Insert error message before results
            const container = document.querySelector('.container');
            const results = document.getElementById('results');
            container.insertBefore(errorDiv, results);
            
            // Remove error message after 5 seconds
            setTimeout(() => {
                errorDiv.remove();
            }, 5000);
        }

        // Initialize
        checkApiStatus();
    </script>
</body>
</html>
"""

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML interface"""
    return FRONTEND_HTML

@app.get("/api", response_class=HTMLResponse)
async def api_info():
    """API information endpoint (for backwards compatibility)"""
    return {
        "message": "Stuttgart Building Agent API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "POST - Analyze plot for zoning, building code, and accessibility",
            "/ask": "POST - Ask questions about building regulations",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        },
        "rag_ready": rag_system.is_ready if rag_system else False
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "groq_connected": groq_client is not None,
        "rag_ready": rag_system.is_ready if rag_system else False,
        "documents_loaded": len(rag_system.documents) if rag_system else 0
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_plot(plot: PlotInput):
    """Analyze plot for all regulations - optimized for speed"""
    import time
    start_time = time.time()
    
    plot_data = plot.model_dump()
    
    # Process all tasks concurrently
    tasks = ["analyze_zoning", "check_building_code", "assess_accessibility"]
    results = await asyncio.gather(
        *[process_task_async(task, plot_data) for task in tasks],
        return_exceptions=True
    )
    
    # Build response
    response_data = {}
    for i, task in enumerate(tasks):
        result = results[i]
        if isinstance(result, Exception):
            response_data[task.replace("analyze_", "").replace("check_", "").replace("assess_", "")] = {
                "error": str(result)
            }
        else:
            response_data[task.replace("analyze_", "").replace("check_", "").replace("assess_", "")] = result
    
    processing_time = time.time() - start_time
    
    return AnalysisResponse(
        zoning=response_data.get("zoning"),
        building_code=response_data.get("building_code"),
        accessibility=response_data.get("accessibility"),
        processing_time_seconds=round(processing_time, 2),
        status="completed"
    )

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(question_input: QuestionInput):
    """Ask questions about building regulations"""
    import time
    start_time = time.time()
    
    if not rag_system or not rag_system.is_ready:
        raise HTTPException(status_code=503, detail="RAG system not ready")
    
    try:
        # Search for relevant documents
        relevant_docs = rag_system.search(question_input.question, top_k=3)
        
        # Build context-aware prompt
        context = "\n\n".join(relevant_docs)
        prompt = f"""Based on Stuttgart building regulations, answer this question:

Question: {question_input.question}

Relevant regulations:
{context}

Provide a clear, specific answer based on the regulations above."""

        # Get AI response
        response = await asyncio.to_thread(
            groq_client.complete,
            prompt,
            max_tokens=500,
            temperature=0.2
        )
        
        answer = "No answer available"
        if "choices" in response and response["choices"]:
            answer = response["choices"][0].get("message", {}).get("content", answer)
        
        processing_time = time.time() - start_time
        
        return QuestionResponse(
            answer=answer,
            sources=relevant_docs[:2],  # Return top 2 sources
            processing_time_seconds=round(processing_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reindex")
async def reindex_documents(background_tasks: BackgroundTasks):
    """Reindex documents in the background"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    background_tasks.add_task(rag_system.reindex_documents)
    return {"message": "Reindexing started in background"}

# -----------------------------------------------------------------------------
# Run with: python -m uvicorn optimized_app:app --reload --host 0.0.0.0 --port 8000
# -----------------------------------------------------------------------------