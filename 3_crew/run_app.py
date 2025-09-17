# stuttgart_reg_agent/run_app.py
import sys
from pathlib import Path
import uvicorn

# Add src/ to sys.path
project_root = Path(__file__).parent / "src"
sys.path.append(str(project_root))

if __name__ == "__main__":
    uvicorn.run(
        "stuttgart_reg_agent.app:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True
    )
