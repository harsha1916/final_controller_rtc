#!/usr/bin/env python3
"""
Test script for the RFID Access Control System Web Interface
"""

import os
import json
import time
import requests
from datetime import datetime

def test_interface():
    """Test the web interface endpoints"""
    
    base_url = "http://localhost:5001"
    
    print("🧪 Testing RFID Access Control System Web Interface")
    print("=" * 60)
    
    # Test 1: Main interface
    print("\n1. Testing main interface...")
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            print("✅ Main interface accessible")
        else:
            print(f"❌ Main interface failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Main interface error: {e}")
    
    # Test 2: Get transactions
    print("\n2. Testing transactions endpoint...")
    try:
        response = requests.get(f"{base_url}/get_transactions", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Transactions endpoint working - {len(data)} transactions")
        else:
            print(f"❌ Transactions endpoint failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Transactions endpoint error: {e}")
    
    # Test 3: Get images
    print("\n3. Testing images endpoint...")
    try:
        response = requests.get(f"{base_url}/get_images", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Images endpoint working:")
            print(f"   - Total images: {data.get('total', 0)}")
            print(f"   - Displayed: {len(data.get('images', []))}")
            print(f"   - Uploaded: {data.get('uploaded', 0)}")
            print(f"   - Pending: {data.get('pending', 0)}")
            print(f"   - Failed: {data.get('failed', 0)}")
        else:
            print(f"❌ Images endpoint failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Images endpoint error: {e}")
    
    # Test 4: System status
    print("\n4. Testing system status...")
    try:
        response = requests.get(f"{base_url}/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ System status endpoint working:")
            print(f"   - System: {data.get('system', 'unknown')}")
            print(f"   - Firebase: {data.get('components', {}).get('firebase', False)}")
            print(f"   - Internet: {data.get('components', {}).get('internet', False)}")
        else:
            print(f"❌ System status endpoint failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ System status endpoint error: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Interface testing completed!")

def create_test_images():
    """Create some test images for demonstration"""
    
    print("\n📸 Creating test images...")
    
    # Ensure images directory exists
    images_dir = "images"
    os.makedirs(images_dir, exist_ok=True)
    
    # Create a simple test image (1x1 pixel JPEG)
    test_image_data = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9'
    
    # Create test images with different upload statuses
    test_cases = [
        {"card": "123456789", "uploaded": True, "s3_location": "https://s3.amazonaws.com/test-bucket/test1.jpg"},
        {"card": "987654321", "uploaded": False, "s3_location": None},
        {"card": "555666777", "uploaded": None, "s3_location": None},
    ]
    
    for i, test_case in enumerate(test_cases):
        timestamp = int(time.time()) - i * 3600  # Different timestamps
        filename = f"{test_case['card']}_{timestamp}.jpg"
        filepath = os.path.join(images_dir, filename)
        
        # Create the image file
        with open(filepath, 'wb') as f:
            f.write(test_image_data)
        
        # Create upload sidecar if uploaded
        if test_case['uploaded'] is True:
            sidecar_data = {
                "uploaded_at": timestamp,
                "s3_location": test_case['s3_location']
            }
            with open(filepath + ".uploaded.json", 'w') as f:
                json.dump(sidecar_data, f, indent=2)
        
        print(f"   Created: {filename} (uploaded: {test_case['uploaded']})")
    
    print(f"✅ Created {len(test_cases)} test images in {images_dir}/")

if __name__ == "__main__":
    print("🚀 RFID Access Control System - Interface Test")
    print("Make sure the Flask application is running on localhost:5001")
    print()
    
    # Create test images first
    create_test_images()
    
    # Test the interface
    test_interface()
    
    print("\n💡 To view the interface, open: http://localhost:5001/")
    print("💡 Make sure the Flask app is running with: python integrated_access_camera.py")
