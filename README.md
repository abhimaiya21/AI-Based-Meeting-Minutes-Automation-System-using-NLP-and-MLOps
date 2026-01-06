# AI-Based Meeting Minutes Automation System (MLOps)

## Project Overview

This project automates the extraction of actionable insights from meeting transcripts using an end-to-end MLOps pipeline. It addresses SDG 9 (Industry, Innovation) by optimizing corporate workflows.

## MLOps Tech Stack

- **Data Versioning:** DVC
- **Experiment Tracking:** MLflow
- **Model Serving:** FastAPI
- **NLP Engine:** Spacy (en_core_web_sm)
- **Monitoring:** Evidently AI
- **CI/CD:** GitHub Actions / Local Batch Pipeline
- **Containerization:** Docker

**Where Implemented**

- **Data Versioning (DVC):** [data.dvc](data.dvc#L1) — DVC tracking file for the `data/` folder; also see [.dvcignore](.dvcignore#L1) and `requirements.txt` which includes DVC packages.
- **Experiment Tracking (MLflow):** [src/train_models.py](src/train_models.py#L1-L20), [main.py](main.py#L1-L40), [debug_main.py](debug_main.py#L1-L40) — experiment setup, logging metrics, and artifact logging.
- **Model Serving (FastAPI):** [main.py](main.py#L1-L120) — FastAPI app with `/analyze` and `/chat` endpoints and `uvicorn` launch.
- **NLP Engine (spaCy):** [main.py](main.py#L1-L40), [Dockerfile](Dockerfile#L1-L10) — spaCy model load and sentence/entity extraction; Dockerfile downloads `en_core_web_sm`.
- **Monitoring (Evidently AI):** [src/monitor.py](src/monitor.py#L1-L40) — Evidently `Report` generation and HTML report output.
- **CI/CD (GitHub Actions):** [.github/workflows/mlops_pipeline.yml](.github/workflows/mlops_pipeline.yml#L1-L20) — CI steps to install deps, run tests, generate reports, and build the Docker image.
- **Containerization (Docker):** [Dockerfile](Dockerfile#L1-L20) — container image for running the FastAPI app.

## Project Structure

- `backend/`: Dockerfile for the API service.
- `frontend/`: Dockerfile for the Streamlit UI service.
- `src/`: Core logic and helper scripts.
  - `generate_synthetic_data.py`: Generates training data.
  - `monitor.py`: Drift detection and governance reporting.
  - `train_models.py`: Training pipeline.
  - `xml_parser.py`: Parsing AMI XML corpus.
- `tests/`: Unit tests.
- `data/`: Data storage (tracked by DVC).
- `main.py`: FastAPI entry point.
- `docker-compose.yml`: Container orchestration config.
- `requirements.txt`: Python dependencies.

## Deployment Guide: Run on Another Laptop

### Prerequisites
- **Docker Desktop** installed and running.
- **Git** installed.
- (Optional) Python 3.9+ if running locally without Docker.

### Option A: Run with Docker (Recommended)
This is the easiest way to run the application as it handles all dependencies automatically.

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd Meeting_AI_Project
    ```
2.  **Build and Start Services:**
    ```bash
    docker-compose up --build
    ```
3.  **Access the Application:**
    - **Frontend (Streamlit UI):** [http://localhost:8501](http://localhost:8501)
    - **Backend (FastAPI Docs):** [http://localhost:8000/docs](http://localhost:8000/docs)

### Option B: Run Locally (Python Virtual Env)
Use this if you want to develop or debug without Docker.

1.  **Clone and Setup Environment:**
    ```bash
    git clone <repository_url>
    cd Meeting_AI_Project
    python -m venv venv
    
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Backend (API):**
    Open a terminal and run:
    ```bash
    python main.py
    ```
4.  **Run the Frontend (UI):**
    Open a *new* terminal, activate venv, and run:
    ```bash
    streamlit run app.py
    ```
    *(Note: Ensure `app.py` is the correct entry point for frontend, or `streamlit_app.py` based on file presence)*

    # AI-Based Meeting Minutes Automation System — A → Z Report

    **Project goal:** Automatically extract summaries, topics, action items and a sentiment score from meeting transcripts (AMI corpus XML) using a hybrid AI pipeline (LLM + classical ML), and provide a production-ready API + UI and MLOps tooling.

    **Quick links:**

    - Code entrypoint (API): [main.py](main.py)
    - Streamlit demo: [streamlit_app.py](streamlit_app.py)
    - Training code: [src/train_models.py](src/train_models.py)
    - Data generation: [src/generate_synthetic_data.py](src/generate_synthetic_data.py)
    - Parsing pipeline: [src/xml_parser.py](src/xml_parser.py)
    - Monitoring: [src/monitor.py](src/monitor.py)
    - Requirements: [requirements.txt](requirements.txt)

    **High-level architecture**

    - Input: AMI meeting XML files in `data/` (e.g. `EN2004a.*.words.xml`).
    - Parsing: XML → cleaned text + ML features (`src/xml_parser.py`, `src/xml_to_csv.py`).
    - ML models: classical regressors predict a sentiment score (0–10) from features.
    - LLM summarization / chat: Google GenAI (Gemini) used to generate executive summaries and answer questions.
    - Serving: FastAPI in `main.py` exposes `/analyze` and `/chat` endpoints. UI: Streamlit app posts to the API.
    - MLOps: DVC for data, MLflow for experiment tracking, models persisted in `models/`, governance reports in `governance_report.html`.

    **Detailed A → Z**

    **1) Data**

    - Source: AMI meeting transcripts (XML files) located under the `data/` directory.
    - Synthetic training data: `src/generate_synthetic_data.py` produces `data/training_data.csv` used to train local models when real labels are not available.
    - DVC: `data.dvc` is present if you set up a remote for reproducible dataset versioning.

    **2) Parsing & Feature Extraction**

    - `src/xml_parser.py` / helper in `main.py` parse XML tokens and build:
      - `word_count` — total tokens
      - `positive_count` — heuristic counts (words like "yeah", "right", "good", "okay")
      - `negative_count` — heuristic counts ("no", "sorry", "problem", etc.)
    - The same extractor provides two text outputs: full text (for spaCy/NER/action items) and a filtered text (for Gemini prompts).

    **3) ML Models & Training**

    - Models implemented (see `src/train_models.py`):
      - `LinearRegression` (sklearn)
      - `RandomForestRegressor` (sklearn)
      - `XGBRegressor` (xgboost)
    - Training pipeline:
      - Trains on `data/training_data.csv` (synthetic or real labeled data).
      - Splits data and evaluates using a classification threshold (score > 5 → positive).
      - Metrics logged to MLflow: accuracy, precision, recall, specificity, F1, AUC.
      - ROC plots saved and logged as artifacts; models persisted in `models/{ModelName}.pkl`.

    **4) LLM / AI Integration**

    - The API uses Google GenAI client (`google.genai`) in `main.py` to call `gemini-3-flash-preview` for:
      - generating meeting summaries and topic lists (`/analyze`), and
      - chat-style Q&A over extracted meeting context (`/chat`).
    - The Streamlit UI (`streamlit_app.py`) uploads XML to `/analyze` and shows Gemini summaries alongside ML predictions.

    **5) API & UI**

    - FastAPI endpoints (in [main.py](main.py)):
      - `POST /analyze` — upload XML file; returns `gemini_summary`, `gemini_topics`, `ml_predictions`, `ml_features`, and `action_items`.
      - `POST /chat` — ask a question with `context_text` and `question` and receive an LLM response.
    - Streamlit demo: `streamlit_app.py` provides a web dashboard to upload transcripts, view ML model outputs, and query the LLM.

    **6) MLOps & Observability**

    - Experiment tracking: MLflow (runs stored under `mlruns/`) — `src/train_models.py` sets experiment `Meeting_Sentiment_Analysis`.
    - Monitoring / drift: `src/monitor.py` and generated `governance_report.html` provide dataset & model drift / governance reporting.
    - DVC: dataset versioning via `data.dvc` (configure remotes for true reproducibility).

    **7) Security & Secrets**

    - `main.py` currently shows an `API_KEY` variable used for `google.genai.Client`. Move all secrets into environment variables (recommended):
      - Example: `export GENAI_API_KEY=...` (Windows: `setx GENAI_API_KEY "..."`), then read via `os.environ`.

    **8) How to run locally (minimal reproducible steps)**

    1. Create and activate a Python virtual environment, then install requirements:
       ```powershell
       python -m venv venv
       .\venv\Scripts\Activate.ps1
       pip install -r requirements.txt
       ```
    2. (Optional) Generate synthetic data and train models:
       ```powershell
       python src/generate_synthetic_data.py
       python src/train_models.py
       ```
       Trained models will be saved to `models/`.
    3. Start the API:
       ```powershell
       python main.py
       ```
       The API serves on `http://127.0.0.1:8002` by default (see [main.py](main.py)).
    4. Start Streamlit UI in a separate terminal:
       ```powershell
       streamlit run streamlit_app.py
       ```

    **9) Testing & development tips**

    - Run unit tests:
      ```powershell
      pytest -q
      ```
    - View MLflow UI:
      ```powershell
      mlflow ui --host 127.0.0.1 --port 5000
      ```
    - If using DVC remotes:
      ```powershell
      dvc pull
      ```

    **10) Files & where to look (quick guide)**

    - API / server: [main.py](main.py)
    - Streamlit demo: [streamlit_app.py](streamlit_app.py)
    - Training: [src/train_models.py](src/train_models.py)
    - Data generation: [src/generate_synthetic_data.py](src/generate_synthetic_data.py)
    - XML parsing: [src/xml_parser.py](src/xml_parser.py) and [src/xml_to_csv.py](src/xml_to_csv.py)
    - Monitoring: [src/monitor.py](src/monitor.py) and `governance_report.html`
    - Tests: [tests/test_app.py](tests/test_app.py)
    - Dependencies: [requirements.txt](requirements.txt)

    **11) Suggested next steps**

    - Replace hard-coded API keys with env vars and add `.env` support (`python-dotenv`).
    - Add CI workflow to run tests and `mlflow`/`dvc` checks on PRs.
    - Add a small Docker Compose that runs API + MLflow + Streamlit for local integration testing.
    - Add model explainability (SHAP / ELI5) artifacts to MLflow runs for production explainability.

    ***

    If you'd like, I can:

    - commit this README update, or
    - also create a `CONTRIBUTING.md`, an example `.env.example`, and a `docker-compose.yml` to run API + MLflow + Streamlit locally — tell me which you'd prefer.
