#!/usr/bin/env python3
"""
Simple test script to verify the Flask API endpoints work correctly
"""

import requests
import json

# Test data
test_comments = [
    {
        "text": "This is a great video!",
        "timestamp": "2023-01-01T00:00:00Z"
    },
    {
        "text": "I hate this content",
        "timestamp": "2023-01-02T00:00:00Z"
    },
    {
        "text": "This is okay I guess",
        "timestamp": "2023-01-03T00:00:00Z"
    }
]

def test_predict_with_timestamps():
    """Test the predict_with_timestamps endpoint"""
    url = "http://localhost:5000/predict_with_timestamps"
    payload = {"comments": test_comments}
    
    try:
        print("Testing /predict_with_timestamps endpoint...")
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Success!")
            for item in data:
                print(f"Comment: {item['comment'][:50]}... -> Sentiment: {item['sentiment']}")
        else:
            print("❌ Failed!")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask server. Is it running on localhost:5000?")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_generate_chart():
    """Test the generate_chart endpoint"""
    url = "http://localhost:5000/generate_chart"
    payload = {"sentiment_counts": {"1": 5, "0": 3, "-1": 2}}
    
    try:
        print("\nTesting /generate_chart endpoint...")
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Chart generated successfully!")
            print(f"Content-Type: {response.headers.get('Content-Type')}")
        else:
            print(f"❌ Failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_predict_with_timestamps()
    test_generate_chart()
