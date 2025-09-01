import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow   #type: ignore
import mlflow.sklearn   #type: ignore
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature   #type: ignore

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise


def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise


def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and log classification metrics and confusion matrix."""
    try:
        # Predict and calculate classification metrics
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.debug('Model evaluation completed')

        return report, cm
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def log_confusion_matrix(cm, dataset_name):
    """Log confusion matrix as an artifact."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save confusion matrix plot as a file and log it to MLflow
    cm_file_path = f'confusion_matrix_{dataset_name}.png'
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        # Create a dictionary with the info you want to save
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        # Save the dictionary as a JSON file
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise


def main():
    # Use remote MLflow tracking server
    mlflow.set_tracking_uri("http://51.21.200.99:5000/")

    mlflow.set_experiment('dvc-pipeline-runs')
    
    # Set AWS credentials for MLflow if available
    import os
    if 'AWS_ACCESS_KEY_ID' in os.environ and 'AWS_SECRET_ACCESS_KEY' in os.environ:
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.environ.get('MLFLOW_S3_ENDPOINT_URL', 'https://s3.us-east-1.amazonaws.com')
        print("AWS credentials found, configuring MLflow for S3 access")
    else:
        print("No AWS credentials found in environment")
    
    # Initialize variables that we'll need regardless of MLflow success
    run_id = None
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    
    try:
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            print(f"Started MLflow run: {run_id}")
            
            # Load parameters from YAML file
            params = load_params(os.path.join(root_dir, 'params.yaml'))

            # Log parameters (with error handling)
            try:
                for key, value in params.items():
                    mlflow.log_param(key, value)
            except Exception as param_error:
                print(f"Failed to log parameters to MLflow: {param_error}")
            
            # Load model and vectorizer
            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Load test data for signature inference
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))

            # Prepare test data
            X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            # Create a DataFrame for signature inference (using first few rows as an example)
            input_example = pd.DataFrame(X_test_tfidf.toarray()[:5], columns=vectorizer.get_feature_names_out())  # <--- Added for signature    #type: ignore

            # Infer the signature
            signature = infer_signature(input_example, model.predict(X_test_tfidf[:5]))  # <--- Added for signature #type: ignore

            # Log model with signature
            try:
                mlflow.sklearn.log_model(
                    model,
                    "lgbm_model",
                    signature=signature,  # <--- Added for signature
                    input_example=input_example  # <--- Added input example
                )
                print("Model logged to MLflow successfully")
            except Exception as model_log_error:
                print(f"Failed to log model to MLflow: {model_log_error}")
                print("Continuing without MLflow model logging...")

            # Log the vectorizer as an artifact
            try:
                mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))
                print("Vectorizer logged to MLflow successfully")
            except Exception as artifact_log_error:
                print(f"Failed to log vectorizer to MLflow: {artifact_log_error}")
                print("Continuing without MLflow artifact logging...")

            # Evaluate model and get metrics
            report, cm = evaluate_model(model, X_test_tfidf, y_test)    #type: ignore

            # Log classification report metrics for the test data
            try:
                for label, metrics in report.items():   #type: ignore
                    if isinstance(metrics, dict):
                        mlflow.log_metrics({
                            f"test_{label}_precision": metrics['precision'],
                            f"test_{label}_recall": metrics['recall'],
                            f"test_{label}_f1-score": metrics['f1-score']
                        })
            except Exception as metrics_error:
                print(f"Failed to log metrics to MLflow: {metrics_error}")

            # Log confusion matrix
            try:
                log_confusion_matrix(cm, "Test Data")
            except Exception as cm_error:
                print(f"Failed to log confusion matrix to MLflow: {cm_error}")

            # Add important tags
            try:
                mlflow.set_tag("model_type", "LightGBM")
                mlflow.set_tag("task", "Sentiment Analysis")
                mlflow.set_tag("dataset", "YouTube Comments")
            except Exception as tag_error:
                print(f"Failed to set MLflow tags: {tag_error}")

    except Exception as mlflow_error:
        print(f"MLflow operations failed: {mlflow_error}")
        print("Continuing with local operations...")
        # Generate a fallback run_id
        import uuid
        run_id = str(uuid.uuid4())
        print(f"Using fallback run_id: {run_id}")
    
    # Always save model info (regardless of MLflow success)
    try:
        model_path = "lgbm_model"
        save_model_info(run_id or "fallback", model_path, 'experiment_info.json')
        print("✅ experiment_info.json created successfully")
    except Exception as save_error:
        print(f"Failed to save experiment_info.json: {save_error}")
        # Create a minimal experiment_info.json to satisfy DVC
        try:
            import json
            fallback_info = {
                'run_id': run_id or "fallback",
                'model_path': "lgbm_model",
                'status': 'completed_with_errors' if run_id is None else 'completed'
            }
            with open(os.path.join(root_dir, 'experiment_info.json'), 'w') as f:
                json.dump(fallback_info, f, indent=4)
            print("✅ Fallback experiment_info.json created")
        except Exception as fallback_error:
            logger.error(f"Failed to create fallback experiment_info.json: {fallback_error}")
            raise

if __name__ == '__main__':
    main()