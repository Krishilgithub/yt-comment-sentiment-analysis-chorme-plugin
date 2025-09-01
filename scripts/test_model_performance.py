#!/usr/bin/env python3
"""
Test script to verify model performance meets minimum thresholds
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
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError as e:
    pytest.skip(f"Required packages not available: {e}")

# Minimum performance thresholds
MIN_ACCURACY = 0.60  # 60% minimum accuracy
MIN_F1_SCORE = 0.50  # 50% minimum F1 score

def test_model_performance():
    """Test that model meets minimum performance thresholds"""
    try:
        # Get the project root directory
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        # Load test data
        test_data_path = os.path.join(root_dir, "data", "interim", "test_processed.csv")
        
        if not os.path.exists(test_data_path):
            pytest.skip("Test data not available for performance testing")
        
        # Load the model and vectorizer
        model_path = os.path.join(root_dir, "lgbm_model.pkl")
        vectorizer_path = os.path.join(root_dir, "tfidf_vectorizer.pkl")
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        # Load test data
        test_df = pd.read_csv(test_data_path)
        
        # Assume the test data has 'text' and 'label' columns
        if 'text' not in test_df.columns or 'label' not in test_df.columns:
            pytest.skip("Test data doesn't have expected columns (text, label)")
        
        # Take a sample if dataset is large
        if len(test_df) > 1000:
            test_df = test_df.sample(n=1000, random_state=42)
        
        # Prepare data
        X_test = test_df['text'].values
        y_test = test_df['label'].values
        
        # Transform and predict
        X_test_transformed = vectorizer.transform(X_test)
        predictions = model.predict(X_test_transformed)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        
        print(f"Model Performance:")
        print(f"  Accuracy: {accuracy:.3f} (minimum: {MIN_ACCURACY})")
        print(f"  F1-Score: {f1:.3f} (minimum: {MIN_F1_SCORE})")
        
        # Assert minimum performance
        assert accuracy >= MIN_ACCURACY, f"Accuracy {accuracy:.3f} below threshold {MIN_ACCURACY}"
        assert f1 >= MIN_F1_SCORE, f"F1-score {f1:.3f} below threshold {MIN_F1_SCORE}"
        
        print("✅ Model performance test passed")
        
    except Exception as e:
        pytest.fail(f"Model performance test failed: {e}")

def test_model_consistency():
    """Test that model produces consistent results"""
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
            "This is amazing! I love it!",  # Should be positive
            "This is terrible! I hate it!",  # Should be negative
            "This is okay, not great but not bad"  # Should be neutral
        ]
        
        # Transform and predict multiple times
        transformed_comments = vectorizer.transform(test_comments)
        
        predictions1 = model.predict(transformed_comments)
        predictions2 = model.predict(transformed_comments)
        
        # Check consistency
        assert np.array_equal(predictions1, predictions2), "Model should produce consistent results"
        
        print("✅ Model consistency test passed")
        
    except Exception as e:
        pytest.fail(f"Model consistency test failed: {e}")

if __name__ == "__main__":
    test_model_performance()
    test_model_consistency()
    print("All performance tests completed!")
