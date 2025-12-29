# run_local.py
import sys
from pathlib import Path

# Add parent directory to path so imports work
sys.path.append(str(Path(__file__).parent.parent))

if __name__ == "__main__":
    import uvicorn
    from stuttgart_reg_agent.optimized_app import app
    
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)