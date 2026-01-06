# Backend Dockerfile
# Use a modern Python version matching the project (3.12)
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
# build-essential for compiling some python packages
# curl for healthchecks (optional but good practice)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
# Install dependencies in stages to prevent timeouts and reduce image size
# 1. Install heavy ML libraries (CPU versions)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir tensorflow-cpu

# 2. Copy requirements and install the rest
COPY requirements.txt .
# Filter out tensorflow and torch from requirements to avoid conflict/redownload
RUN grep -vE "^tensorflow|^torch" requirements.txt > requirements_light.txt \
    && pip install --no-cache-dir -r requirements_light.txt

# Download Spacy model
RUN python -m spacy download en_core_web_sm

# Copy Application Code
# We rely on .dockerignore to exclude venv, git, etc.
COPY . .

# Create directories for artifacts if they don't exist
RUN mkdir -p reports mlruns models

# Expose port 8000
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Run uvicorn using the file at /app/main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]