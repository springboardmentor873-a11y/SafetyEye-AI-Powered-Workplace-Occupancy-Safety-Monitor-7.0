"""
Test script for YOLO Detection API
Run this after starting the FastAPI server
"""

import requests
import base64
import json
from pathlib import Path
from PIL import Image
import io

API_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{API_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_predict_with_image(image_path: str):
    """Test prediction endpoint with an image"""
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"Image file not found: {image_path}")
        return
    
    print(f"Testing prediction with image: {image_path}")
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(f"{API_URL}/predict", files=files)
    
    print(f"Status: {response.status_code}")
    data = response.json()
    
    if data['success']:
        print(f"✓ Predictions: {data['num_detections']} detections found")
        for i, pred in enumerate(data['predictions']):
            print(f"  Detection {i+1}: {pred['class_name']} (confidence: {pred['confidence']:.2%})")
        print(f"✓ Image encoded: {len(data['image'])} characters")
        
        # Save the annotated image
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes))
        output_path = image_path.parent / f"annotated_{image_path.name}"
        img.save(output_path)
        print(f"✓ Annotated image saved to: {output_path}")
    else:
        print(f"✗ Error: {data}")
    
    print()

def test_predict_raw(image_path: str):
    """Test raw prediction endpoint (without image)"""
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"Image file not found: {image_path}")
        return
    
    print(f"Testing raw prediction with image: {image_path}")
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(f"{API_URL}/predict-raw", files=files)
    
    print(f"Status: {response.status_code}")
    data = response.json()
    
    if data['success']:
        print(f"✓ Predictions: {data['num_detections']} detections found")
        print(json.dumps(data['predictions'], indent=2))
    else:
        print(f"✗ Error: {data}")
    
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("YOLO Detection API Test Suite")
    print("=" * 60 + "\n")
    
    # Test health
    try:
        test_health()
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        print("Make sure the server is running with: python main.py\n")
        exit(1)
    
    # Test with a sample image if available
    # Uncomment below and provide your test image path
    # test_predict_with_image("path/to/test_image.jpg")
    # test_predict_raw("path/to/test_image.jpg")
    
    print("=" * 60)
    print("Basic tests passed! Server is ready.")
    print("=" * 60)
