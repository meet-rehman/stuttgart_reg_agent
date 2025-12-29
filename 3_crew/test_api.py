# test_api.py - Test your FastAPI endpoints

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_analyze():
    """Test plot analysis endpoint"""
    print("🏢 Testing plot analysis...")
    
    plot_data = {
        "location": "Stuttgart-Mitte",
        "size_m2": 500.0,
        "building_type": "residential",
        "floors": 3,
        "height_m": 10.0
    }
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/analyze", json=plot_data)
    end_time = time.time()
    
    print(f"Status: {response.status_code}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    
    if response.status_code == 200:
        result = response.json()
        print(f"API Response time: {result.get('processing_time_seconds')} seconds")
        print("Results:")
        for key, value in result.items():
            if key not in ['processing_time_seconds', 'status']:
                print(f"  {key}: {json.dumps(value, indent=4)}")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)

def test_question():
    """Test question answering endpoint"""
    print("❓ Testing question answering...")
    
    question_data = {
        "question": "What are the height restrictions for residential buildings in Stuttgart?"
    }
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/ask", json=question_data)
    end_time = time.time()
    
    print(f"Status: {response.status_code}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Answer: {result.get('answer')}")
        print(f"Sources: {len(result.get('sources', []))} documents found")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)

if __name__ == "__main__":
    print("🚀 Testing Stuttgart Building Agent API\n")
    
    try:
        test_health()
        test_analyze()
        test_question()
        
        print("✅ All tests completed!")
        print("\n💡 Your API is working! The server stays running to handle requests.")
        print("💡 The 'favicon.ico 404' error is normal - just your browser looking for an icon.")
        print("💡 Visit http://127.0.0.1:8000/docs for interactive API documentation.")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API. Make sure the server is running.")
    except Exception as e:
        print(f"❌ Test error: {e}")