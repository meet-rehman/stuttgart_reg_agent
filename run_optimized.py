import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "stuttgart_reg_agent.optimized_app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1,
        log_level="info"
    )