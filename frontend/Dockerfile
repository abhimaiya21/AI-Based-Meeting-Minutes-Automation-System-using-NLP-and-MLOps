# Frontend Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install heavy dependencies individually to prevent OOM
RUN pip install --no-cache-dir \
    tensorflow \
    torch \
    spacy \
    scikit-learn \
    pandas \
    xgboost \
    mlflow

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Frontend Files
COPY app.py /app/

# Expose port
EXPOSE 8501

# Run Streamlit
# We use shell form to allow variable expansion if needed, but array form is safer.
# Using --server.address 0.0.0.0 to be accessible externally
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
