# src/stuttgart_reg_agent/app.py
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import yaml

from stuttgart_reg_agent.schemas import (
    PlotDetails,
    ZoningRecommendation,
    BuildingCodeCompliance,
    AccessibilityAnalysis,
)
from stuttgart_reg_agent.main import build_task_prompt, validate_and_parse
from stuttgart_reg_agent.tools.groq_client import GroqClient
from sentence_transformers import SentenceTransformer
from stuttgart_reg_agent.new_crew import DocumentCrew

# -------------------------------------------------------------------------
# Load environment
# -------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_FILE = PROJECT_ROOT / ".env1"
if DOTENV_FILE.exists():
    load_dotenv(dotenv_path=DOTENV_FILE)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL")

# Directories
CONFIG_DIR = Path(__file__).parent / "config"
MEMORY_DIR = PROJECT_ROOT / "memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Load tasks.yaml
# -------------------------------------------------------------------------
with open(CONFIG_DIR / "tasks.yaml", "r", encoding="utf-8") as f:
    tasks = yaml.safe_load(f)

# Initialize GroqClient and embedding model
groq = GroqClient(api_key=GROQ_API_KEY, api_url=GROQ_API_URL, model="openai/gpt-oss-20b")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Map task names to Pydantic models
model_map = {
    "analyze_zoning": ZoningRecommendation,
    "check_building_code": BuildingCodeCompliance,
    "assess_accessibility": AccessibilityAnalysis,
}

# -------------------------------------------------------------------------
# Initialize PDF crew for document Q&A
# -------------------------------------------------------------------------
pdf_crew = DocumentCrew(pdf_dir="data")
# ingestion will now be via endpoint
crew = pdf_crew.create_crew()

# -------------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------------
app = FastAPI(title="Stuttgart Building Agent")

# -------------------------------------------------------------------------
# Input schemas
# -------------------------------------------------------------------------
class PlotInput(BaseModel):
    location: str
    size_m2: float
    building_type: str
    floors: int
    height_m: float

class QuestionInput(BaseModel):
    question: str

# -------------------------------------------------------------------------
# Root endpoint
# -------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Welcome to Stuttgart Building Agent API! Available endpoints: /analyze, /ask_pdf, /ingest_pdfs"}

# -------------------------------------------------------------------------
# Endpoint: analyze plot
# -------------------------------------------------------------------------
@app.post("/analyze")
def analyze_plot(plot: PlotInput):
    plot_data = plot.model_dump()
    results = {}

    for task_name, task_spec in tasks.items():
        model = model_map.get(task_name)
        if not model:
            continue

        prompt = build_task_prompt(task_name, task_spec, plot_data, model)
        resp_json = groq.complete(prompt, max_tokens=512, temperature=0.0)

        # Extract text output
        text_output = None
        if "choices" in resp_json and len(resp_json["choices"]) > 0:
            choice = resp_json["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                text_output = choice["message"]["content"]
            elif "text" in choice:
                text_output = choice["text"]
            else:
                text_output = json.dumps(choice)
        else:
            text_output = json.dumps(resp_json)

        # Parse and validate
        try:
            parsed = validate_and_parse(model, text_output)
            results[task_name] = parsed.model_dump()
        except Exception as e:
            results[task_name] = {"error": str(e), "raw_output": text_output}

        # Save embedding
        embedding = embedder.encode(text_output)
        em_record = {
            "task": task_name,
            "embedding": embedding.tolist(),
            "result": results[task_name]
        }
        with open(MEMORY_DIR / "embeddings.jsonl", "a", encoding="utf-8") as ef:
            ef.write(json.dumps(em_record, ensure_ascii=False) + "\n")

    return results

# -------------------------------------------------------------------------
# Endpoint: ask PDFs
# -------------------------------------------------------------------------
@app.post("/ask_pdf")
def ask_pdf(question_input: QuestionInput):
    agent = crew.agents[0]  # PDF agent
    response = agent.respond(question_input.question)
    return {"answer": response}

# -------------------------------------------------------------------------
# Endpoint: ingest PDFs
# -------------------------------------------------------------------------
@app.post("/ingest_pdfs")
def ingest_pdfs():
    pdf_crew.ingest_pdfs()
    return {"message": "PDF ingestion completed successfully."}
