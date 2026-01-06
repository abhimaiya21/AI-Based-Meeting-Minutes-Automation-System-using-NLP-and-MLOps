# Backend Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (for cv2 or others if needed, keeping minimal for now)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
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

RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy Backend Files
# We copy specific files to avoid clutter
COPY src/ /app/src/
COPY data/ /app/data/
COPY governance_utils.py /app/
COPY main.py /app/

# Create directories for artifacts
RUN mkdir -p reports mlruns models

# Expose port
EXPOSE 8000

# Run uvicorn using the file at /app/main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
