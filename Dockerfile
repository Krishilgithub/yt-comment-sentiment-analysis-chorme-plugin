FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1

# Copy requirements.txt first for better Docker layer caching
COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

COPY flask_app/ /app/

COPY lgbm_model.pkl /app/lgbm_model.pkl
COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

CMD ["python", "app.py"]