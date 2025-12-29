#!/usr/bin/env python3
"""
Minimal test to check if vision API works at all
"""

import os
import base64
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()

def test_minimal_vision():
    """Bare minimum vision test"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No API key")
        return
    
    client = OpenAI(api_key=api_key)
    
    # Find an image
    image_path = Path("data/raw/Landuse Plans/Stuttgart Nord/Nordbahnhofstarsse_Freidhofsstrasse (strasse)/images/Flächennutzungsplan Nordbahnhofstarsse_page_1.png")
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"📄 Loading: {image_path.name}")
    
    # Load and encode image
    img = Image.open(image_path)
    print(f"   Size: {img.size}, Mode: {img.mode}")
    
    # Convert to RGB if needed
    if img.mode in ('RGBA', 'LA', 'P'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
        img = background
    
    # Resize if too large
    max_dim = 2048
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        print(f"   Resized to: {new_size}")
    
    # Encode to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.read()).decode('utf-8')
    
    size_kb = len(base64_image) * 3 / 4 / 1024
    print(f"   Encoded: {size_kb:.1f} KB")
    
    # Test with multiple models
    models_to_test = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-vision-preview"
    ]
    
    for model in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing with: {model}")
        print('='*60)
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What do you see in this image? Describe any text, numbers, or labels visible."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0
            )
            
            result = response.choices[0].message.content
            print(f"\n✅ SUCCESS with {model}")
            print(f"Response:\n{result[:300]}...")
            
            # Check if it's a refusal
            if "unable to" in result.lower() or "cannot analyze" in result.lower():
                print("⚠️ Model refused to analyze")
            else:
                print("✅ Model analyzed the image!")
            
            break  # Stop after first successful model
            
        except Exception as e:
            print(f"❌ Error with {model}: {e}")
            continue

if __name__ == "__main__":
    test_minimal_vision()