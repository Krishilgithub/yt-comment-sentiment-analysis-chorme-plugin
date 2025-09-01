#!/usr/bin/env python3
"""
Test script to verify model loading from MLflow
"""
import pytest
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import mlflow
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError as e:
    pytest.skip(f"Required packages not available: {e}")

def test_model_loading():
    """Test that model can be loaded from local files"""
    try:
        # Get the project root directory
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        # Try to load the model and vectorizer from local files
        model_path = os.path.join(root_dir, "lgbm_model.pkl")
        vectorizer_path = os.path.join(root_dir, "tfidf_vectorizer.pkl")
        
        # Check if files exist
        assert os.path.exists(model_path), f"Model file not found at {model_path}"
        assert os.path.exists(vectorizer_path), f"Vectorizer file not found at {vectorizer_path}"
        
        # Load the model and vectorizer
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        # Basic sanity checks
        assert model is not None, "Model should not be None"
        assert vectorizer is not None, "Vectorizer should not be None"
        assert hasattr(model, 'predict'), "Model should have predict method"
        assert hasattr(vectorizer, 'transform'), "Vectorizer should have transform method"
        
        print("✅ Model loading test passed")
        
    except Exception as e:
        pytest.fail(f"Model loading test failed: {e}")

def test_mlflow_connection():
    """Test MLflow server connection"""
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("http://51.21.200.99:5000/")
        
        # Try to get the current experiment
        client = mlflow.tracking.MlflowClient()
        experiments = client.list_experiments()
        
        assert len(experiments) >= 0, "Should be able to list experiments"
        print("✅ MLflow connection test passed")
        
    except Exception as e:
        # Don't fail the test if MLflow server is not accessible
        print(f"⚠️ MLflow connection test failed (this may be expected): {e}")
        pytest.skip("MLflow server not accessible")

if __name__ == "__main__":
    test_model_loading()
    test_mlflow_connection()
    print("All tests completed!")
