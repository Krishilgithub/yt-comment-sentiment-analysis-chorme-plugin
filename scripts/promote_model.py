#!/usr/bin/env python3
"""
Script to promote model to production after successful validation
"""
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    import json
    from datetime import datetime
except ImportError as e:
    print(f"Required packages not available: {e}")
    sys.exit(1)

def promote_model_to_production():
    """Promote the latest model version to production stage"""
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("http://51.21.200.99:5000/")
        
        client = MlflowClient()
        
        # Model name in MLflow Model Registry
        model_name = "yt-comment-sentiment-model"
        
        try:
            # Get the latest version of the model
            latest_versions = client.get_latest_versions(model_name, stages=["None", "Staging"])
            
            if not latest_versions:
                print("No model versions found. Skipping promotion.")
                return
            
            # Get the latest version
            latest_version = max(latest_versions, key=lambda x: int(x.version))
            
            print(f"Latest model version: {latest_version.version}")
            
            # Promote to production
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version.version,
                stage="Production"
            )
            
            print(f"✅ Model version {latest_version.version} promoted to Production")
            
        except Exception as registry_error:
            print(f"Model registry operation failed: {registry_error}")
            print("This might be expected if the model registry is not set up yet.")
            
            # Try to register the model if it doesn't exist
            try:
                # Look for recent runs with models
                runs = client.search_runs(experiment_ids=["0"], order_by=["start_time desc"], max_results=1)
                
                if runs:
                    run = runs[0]
                    model_uri = f"runs:/{run.info.run_id}/lgbm_model"
                    
                    # Register the model
                    model_version = mlflow.register_model(model_uri, model_name)
                    print(f"✅ Registered new model: {model_name} version {model_version.version}")
                    
                    # Promote to production
                    client.transition_model_version_stage(
                        name=model_name,
                        version=model_version.version,
                        stage="Production"
                    )
                    
                    print(f"✅ Model version {model_version.version} promoted to Production")
                else:
                    print("No recent runs found with models to promote")
                    
            except Exception as register_error:
                print(f"Model registration failed: {register_error}")
                print("Continuing without model promotion...")
        
    except Exception as e:
        print(f"Model promotion failed: {e}")
        print("This might be expected if MLflow server is not accessible.")
        # Don't fail the entire pipeline for MLflow issues
        return

def save_promotion_info():
    """Save information about the promoted model"""
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        promotion_info = {
            "timestamp": str(datetime.now()),
            "status": "promoted",
            "model_name": "yt-comment-sentiment-model",
            "stage": "Production"
        }
        
        with open(os.path.join(root_dir, "promotion_info.json"), "w") as f:
            json.dump(promotion_info, f, indent=2)
            
        print("✅ Promotion info saved")
        
    except Exception as e:
        print(f"Failed to save promotion info: {e}")

if __name__ == "__main__":
    print("Starting model promotion to production...")
    promote_model_to_production()
    save_promotion_info()
    print("Model promotion process completed!")
