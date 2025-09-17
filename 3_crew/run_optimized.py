# run_optimized.py - Railway.app compatible version

import os
import sys
from pathlib import Path

# Add src/ to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

import uvicorn

if __name__ == "__main__":
    # Railway provides PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    
    # Production settings
    uvicorn.run(
        "stuttgart_reg_agent.optimized_app:app",
        host="0.0.0.0",  # Railway requires 0.0.0.0
        port=port,
        reload=False,    # No reload in production
        workers=1,
        log_level="info"
    )