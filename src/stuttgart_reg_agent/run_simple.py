# run_simple.py

import os
import sys
from pathlib import Path

# Add src/ to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

import uvicorn

if __name__ == "__main__":
    print("Starting simple Stuttgart Building Agent...")
    print("This version works without PDF processing for quick testing")
    
    uvicorn.run(
        "stuttgart_reg_agent.simple_app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,  # No reload for stability
        log_level="info"
    )