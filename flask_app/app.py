# app.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

# Load the model and vectorizer from MLflow with fallback to local storage
def load_model_and_vectorizer_with_mlflow_fallback():
    """Load model and vectorizer from MLflow with fallback to local files"""
    import os
    
    # First, try to load from MLflow
    try:
        print("Attempting to load model from MLflow server...")
        # Set MLflow tracking URI to your server with timeout
        mlflow.set_tracking_uri("http://51.21.200.99:5000/")
        
        # Set timeout for MLflow operations (if supported)
        import os as mlflow_os
        mlflow_os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "30"  # 30 seconds timeout
        
        # Load experiment info
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        experiment_info_path = os.path.join(project_root, 'experiment_info.json')
        
        with open(experiment_info_path, 'r') as f:
            experiment_info = json.load(f)
        
        run_id = experiment_info['run_id']
        model_path = experiment_info['model_path']
        
        # Try to load model from MLflow
        model_uri = f"runs:/{run_id}/{model_path}"
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Load vectorizer from local storage (as it's not stored in MLflow)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        vectorizer_path = os.path.join(project_root, "tfidf_vectorizer.pkl")
        vectorizer = joblib.load(vectorizer_path)
        
        print("Successfully loaded model from MLflow and vectorizer from local storage")
        return model, vectorizer
        
    except Exception as mlflow_error:
        print(f"Failed to load from MLflow: {mlflow_error}")
        print("Falling back to local model files...")
        
        # Fallback to local model files
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            
            full_model_path = os.path.join(project_root, "lgbm_model.pkl")
            full_vectorizer_path = os.path.join(project_root, "tfidf_vectorizer.pkl")
            
            model = joblib.load(full_model_path)
            vectorizer = joblib.load(full_vectorizer_path)
            
            print("Successfully loaded model and vectorizer from local storage")
            return model, vectorizer
            
        except Exception as local_error:
            print(f"Failed to load from local storage: {local_error}")
            raise Exception(f"Failed to load models from both MLflow and local storage. MLflow error: {mlflow_error}, Local error: {local_error}")

# Initialize the model and vectorizer
model, vectorizer = load_model_and_vectorizer_with_mlflow_fallback()

@app.route('/')
def home():
    return "Welcome to our flask api"

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        print(f"Received {len(comments_data)} comments for prediction")
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # Preprocess each comment before vectorizing
        print("Preprocessing comments...")
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        print("Vectorizing comments...")
        transformed_comments = vectorizer.transform(preprocessed_comments)
        print(f"Transformed comments shape: {transformed_comments.shape}")
        
        # Make predictions - handle different model types
        print("Making predictions...")
        print(f"Model type: {type(model)}")
        
        # Convert the TF-IDF features to the format expected by the model
        if hasattr(transformed_comments, 'toarray'):
            # If it's a sparse matrix, convert to dense array
            features = transformed_comments.toarray()
        else:
            features = transformed_comments
        
        print(f"Features shape for model: {features.shape}")
        print(f"Features type: {type(features)}")
        
        # For MLflow models, convert to DataFrame with proper feature names
        if 'mlflow' in str(type(model)).lower():
            try:
                # Get feature names from the vectorizer
                feature_names = vectorizer.get_feature_names_out()
                print(f"Number of feature names: {len(feature_names)}")
                
                # Create DataFrame with proper column names
                import pandas as pd
                features_df = pd.DataFrame(features, columns=feature_names)
                print(f"Created DataFrame with shape: {features_df.shape}")
                
                # Make predictions using DataFrame
                predictions = model.predict(features_df)
                print(f"MLflow model predictions successful")
            except Exception as mlflow_error:
                print(f"MLflow DataFrame prediction failed: {mlflow_error}")
                # Fallback to regular array prediction
                predictions = model.predict(features)
        else:
            # Regular sklearn model
            predictions = model.predict(features)
        
        print(f"Raw predictions shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}")
        print(f"Raw predictions sample: {predictions[:5] if len(predictions) > 5 else predictions}")
        
        # Convert predictions to list and then to strings
        if hasattr(predictions, 'tolist'):
            predictions = predictions.tolist()
        predictions = [str(pred) for pred in predictions]
        print(f"Final predictions (first 5): {predictions[:5]}")
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500
    
    # Return the response with original comments, predicted sentiments, and timestamps
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Convert the TF-IDF features to the format expected by the model
        if hasattr(transformed_comments, 'toarray'):
            features = transformed_comments.toarray()
        else:
            features = transformed_comments
        
        # For MLflow models, convert to DataFrame with proper feature names
        if 'mlflow' in str(type(model)).lower():
            try:
                # Get feature names from the vectorizer
                feature_names = vectorizer.get_feature_names_out()
                
                # Create DataFrame with proper column names
                import pandas as pd
                features_df = pd.DataFrame(features, columns=feature_names)
                
                # Make predictions using DataFrame
                predictions = model.predict(features_df)
            except Exception as mlflow_error:
                print(f"MLflow DataFrame prediction failed: {mlflow_error}")
                # Fallback to regular array prediction
                predictions = model.predict(features)
        else:
            # Regular sklearn model
            predictions = model.predict(features)
        
        # Convert predictions to list and then to strings
        if hasattr(predictions, 'tolist'):
            predictions = predictions.tolist()
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments and predicted sentiments
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)