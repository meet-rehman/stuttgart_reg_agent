# debug_env.py - Run this locally to test your environment variables

import os
from pathlib import Path
from dotenv import load_dotenv

print("🔍 Debugging Environment Variables\n")

# Check current working directory
print(f"Current directory: {os.getcwd()}")

# Find project root
PROJECT_ROOT = Path(__file__).resolve().parents[2] if Path(__file__).resolve().parents[2].exists() else Path.cwd()
print(f"Project root: {PROJECT_ROOT}")

# Check for .env1 file
env1_path = PROJECT_ROOT / ".env1"
print(f".env1 path: {env1_path}")
print(f".env1 exists: {env1_path.exists()}")

if env1_path.exists():
    print(f".env1 file size: {env1_path.stat().st_size} bytes")
    
    # Try to read the file content (without showing sensitive data)
    try:
        with open(env1_path, 'r') as f:
            content = f.read()
            lines = content.strip().split('\n')
            print(f".env1 has {len(lines)} lines")
            
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('#'):
                    key = line.split('=')[0] if '=' in line else line
                    print(f"  Line {i+1}: {key}=***")
    except Exception as e:
        print(f"Error reading .env1: {e}")

# Load environment variables
print("\n🔄 Loading environment variables...")
load_dotenv(dotenv_path=env1_path)

# Check if variables are loaded
groq_key = os.getenv("GROQ_API_KEY")
groq_url = os.getenv("GROQ_API_URL")

print(f"\nGroq API Key loaded: {'✅ Yes' if groq_key else '❌ No'}")
if groq_key:
    print(f"Key preview: {groq_key[:10]}...{groq_key[-4:]} (length: {len(groq_key)})")

print(f"Groq API URL loaded: {'✅ Yes' if groq_url else '❌ No'}")
if groq_url:
    print(f"URL: {groq_url}")

# Test API connection
print("\n🔌 Testing API connection...")
try:
    import requests
    
    if not groq_key:
        print("❌ Cannot test - no API key")
    else:
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json"
        }
        
        # Simple test request
        test_data = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10
        }
        
        response = requests.post(groq_url, headers=headers, json=test_data, timeout=10)
        
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print("✅ API connection successful!")
        else:
            print(f"❌ API error: {response.text}")
            
except ImportError:
    print("❌ requests library not installed - run: pip install requests")
except Exception as e:
    print(f"❌ Connection error: {e}")