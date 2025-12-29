#!/usr/bin/env python3
"""
Simple runner script for the Stuttgart Building Agent
This avoids module import issues by running directly from the app directory
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Verify all required files exist
required_files = [
    'optimized_app.py',
    'schemas.py', 
    'precomputed_rag.py',
    'tools/groq_client.py'
]

missing_files = []
for file_path in required_files:
    full_path = current_dir / file_path
    if not full_path.exists():
        missing_files.append(file_path)

if missing_files:
    print("Missing required files:")
    for file in missing_files:
        print(f"  - {file}")
    print(f"Current directory: {current_dir}")
    sys.exit(1)

# Import the app
try:
    from optimized_app import app
    print("Successfully imported the FastAPI app")
except ImportError as e:
    print(f"Failed to import app: {e}")
    sys.exit(1)

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Stuttgart Building Agent...")
    print(f"Current working directory: {current_dir}")
    
    # Run the app
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(current_dir)]
    )