import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize model
model = genai.GenerativeModel('gemini-1.0-pro')  # Stable version

# First query
response = model.generate_content("Write a haiku about coding in Python")
print(response.text)