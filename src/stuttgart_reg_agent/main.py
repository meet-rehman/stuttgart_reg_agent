# src/stuttgart_reg_agent/main.py
"""
Run StuttgartRegAgent tasks using Groq (chat API for text generation)
and Hugging Face sentence-transformers (for embeddings).

Each task prompt includes:
- Task description
- Input plot data
- JSON schema for expected output
- Example template to ensure valid JSON
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
import yaml

from stuttgart_reg_agent.schemas import (
    PlotDetails,
    ZoningRecommendation,
    BuildingCodeCompliance,
    AccessibilityAnalysis,
)
from stuttgart_reg_agent.tools.groq_client import GroqClient
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------------------------
# Load environment variables
# -------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_FILE = PROJECT_ROOT / ".env1"
if DOTENV_FILE.exists():
    load_dotenv(dotenv_path=DOTENV_FILE)
    print(f"Loaded environment variables from {DOTENV_FILE}")
else:
    print("⚠️ Warning: .env1 file not found!")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")

# Directories
CONFIG_DIR = Path(__file__).parent / "config"
MEMORY_DIR = PROJECT_ROOT / "memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def extract_json(text: str) -> str:
    """
    Extract the first valid JSON object or array from text.
    """
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return text.strip()


def load_tasks_yaml() -> dict:
    """
    Load tasks.yaml as a dictionary.
    """
    with open(CONFIG_DIR / "tasks.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_task_prompt(task_name: str, task_spec: dict, plot: dict, schema_model) -> str:
    """
    Build a prompt for Groq/OpenAI-compatible chat API, explicitly requesting JSON output.
    Forces AI to return valid JSON matching the Pydantic schema.
    """
    description = task_spec.get("description", "")
    input_fields = task_spec.get("parameters", {})

    input_fields_str = "\n".join([f"- {k}: {v}" for k, v in input_fields.items()])

    # Convert model_fields to serializable dict: {field_name: type_name}
    serializable_fields = {k: str(v.annotation) for k, v in schema_model.model_fields.items()}

    # JSON schema template for reference
    example_json = {}
    # Add safe example defaults depending on type
    for k, typ in serializable_fields.items():
        typ_lower = typ.lower()
        if "bool" in typ_lower:
            example_json[k] = False
        elif "list" in typ_lower:
            example_json[k] = []
        elif "int" in typ_lower or "float" in typ_lower:
            example_json[k] = 0
        else:
            example_json[k] = "<value>"

    prompt = f"""
You are an expert AI assistant for Stuttgart building regulations.

Task: {task_name}
Description: {description}

Input plot fields:
{input_fields_str}

Plot data:
{json.dumps(plot, indent=2, ensure_ascii=False)}

Instructions:
1. Return a single valid JSON object matching exactly this schema:
{json.dumps(serializable_fields, indent=2, ensure_ascii=False)}
2. Do NOT include any text, explanation, or characters outside the JSON object.
3. Use correct types: booleans for true/false, arrays for lists, numbers for numeric fields.
4. If any data is missing, provide default values:
   - For booleans: false
   - For lists: []
   - For numbers: 0
   - For strings: ""
5. Here is an example template to follow:
{json.dumps(example_json, indent=2, ensure_ascii=False)}

Return only the JSON object. Do not truncate or add extra commentary.
""".strip()

    return prompt


def validate_and_parse(schema_model, json_text: str):
    """
    Parse JSON and validate against Pydantic model.
    """
    try:
        data = json.loads(json_text)
    except Exception:
        candidate = extract_json(json_text)
        data = json.loads(candidate)
    return schema_model.model_validate(data)


# -------------------------------------------------------------------------
# Main workflow
# -------------------------------------------------------------------------
def main():
    tasks = load_tasks_yaml()
    groq = GroqClient(api_key=GROQ_API_KEY, api_url=GROQ_API_URL, model="openai/gpt-oss-20b")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Example plot input
    plot = PlotDetails(
        location="Stuttgart-Mitte",
        size_m2=500.0,
        building_type="residential",
        floors=3,
        height_m=10.0
    ).model_dump()

    results = {}
    model_map = {
        "analyze_zoning": ZoningRecommendation,
        "check_building_code": BuildingCodeCompliance,
        "assess_accessibility": AccessibilityAnalysis,
    }

    for task_name, task_spec in tasks.items():
        print(f"\n--- Running task: {task_name} ---")
        model = model_map.get(task_name)
        if not model:
            print(f"⚠️ No Pydantic model defined for task {task_name}, skipping.")
            continue

        prompt = build_task_prompt(task_name, task_spec, plot, model)
        resp_json = groq.complete(prompt, max_tokens=512, temperature=0.0)

        # Extract content from Groq/OpenAI chat response
        if "choices" in resp_json and isinstance(resp_json["choices"], list) and len(resp_json["choices"]) > 0:
            choice = resp_json["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                text_output = choice["message"]["content"]
            elif "text" in choice:
                text_output = choice["text"]
            else:
                text_output = json.dumps(choice)
        else:
            text_output = json.dumps(resp_json)

        # Parse and validate JSON
        json_candidate = extract_json(text_output)
        try:
            parsed = validate_and_parse(model, json_candidate)
            results[task_name] = parsed.model_dump()
            print(f"✅ {task_name} result:\n{json.dumps(parsed.model_dump(), indent=2, ensure_ascii=False)}")
        except Exception as e:
            print(f"❌ Validation error for {task_name}: {e}")
            print("Raw model output:\n", text_output)
            # Fallback: return empty/default object for schema
            fallback = {}
            for k, v in model.model_fields.items():
                typ = str(v.annotation).lower()
                if "bool" in typ:
                    fallback[k] = False
                elif "list" in typ:
                    fallback[k] = []
                elif "int" in typ or "float" in typ:
                    fallback[k] = 0
                else:
                    fallback[k] = ""
            results[task_name] = {"error": str(e), "raw_output": text_output, "fallback": fallback}

        # Save embedding for the output
        text_for_embedding = text_output or json.dumps(results[task_name])
        embedding = embedder.encode(text_for_embedding)
        em_record = {
            "task": task_name,
            "embedding": embedding.tolist(),
            "result": results[task_name]
        }
        with open(MEMORY_DIR / "embeddings.jsonl", "a", encoding="utf-8") as ef:
            ef.write(json.dumps(em_record, ensure_ascii=False) + "\n")

    # Save all results to file
    with open(MEMORY_DIR / "last_run_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n🎉 All tasks finished. Results saved to:", MEMORY_DIR / "last_run_results.json")


if __name__ == "__main__":
    main()
