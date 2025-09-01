#!/usr/bin/env python3
"""
Test script to verify model signature and compatibility
"""
import pytest
import sys
import os
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import joblib
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError as e:
    pytest.skip(f"Required packages not available: {e}")

def test_model_signature():
    """Test that model signature is compatible with expected input/output"""
    try:
        # Get the project root directory
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        # Load the model and vectorizer
        model_path = os.path.join(root_dir, "lgbm_model.pkl")
        vectorizer_path = os.path.join(root_dir, "tfidf_vectorizer.pkl")
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        # Test data
        test_comments = [
            "This is a great video!",
            "I hate this content",
            "This is okay I guess"
        ]
        
        # Transform test comments
        transformed_comments = vectorizer.transform(test_comments)
        
        # Make predictions
        predictions = model.predict(transformed_comments)
        
        # Verify output format
        assert isinstance(predictions, np.ndarray), "Predictions should be numpy array"
        assert len(predictions) == len(test_comments), "Should have one prediction per comment"
        
        # Verify predictions are in expected range (assuming sentiment classes 0, 1, 2)
        unique_predictions = np.unique(predictions)
        for pred in unique_predictions:
            assert pred in [0, 1, 2], f"Prediction {pred} not in expected range [0, 1, 2]"
        
        print("✅ Model signature test passed")
        
    except Exception as e:
        pytest.fail(f"Model signature test failed: {e}")

def test_vectorizer_compatibility():
    """Test that vectorizer processes text correctly"""
    try:
        # Get the project root directory
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        # Load the vectorizer
        vectorizer_path = os.path.join(root_dir, "tfidf_vectorizer.pkl")
        vectorizer = joblib.load(vectorizer_path)
        
        # Test data
        test_comments = [
            "This is a test comment",
            "Another test comment with different words"
        ]
        
        # Transform comments
        transformed = vectorizer.transform(test_comments)
        
        # Verify output format
        assert hasattr(transformed, 'toarray'), "Vectorizer output should be sparse matrix"
        assert transformed.shape[0] == len(test_comments), "Should have one row per comment"
        assert transformed.shape[1] > 0, "Should have features"
        
        print("✅ Vectorizer compatibility test passed")
        
    except Exception as e:
        pytest.fail(f"Vectorizer compatibility test failed: {e}")

if __name__ == "__main__":
    test_model_signature()
    test_vectorizer_compatibility()
    print("All signature tests completed!")
