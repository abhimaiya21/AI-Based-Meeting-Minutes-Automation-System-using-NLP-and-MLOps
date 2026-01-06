import uvicorn
import shutil
import os
import joblib
import json
import pandas as pd
import xml.etree.ElementTree as ET
import spacy
import mlflow
from google import genai  # Modern 2025 SDK
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, ConfigDict
from fastapi.middleware.cors import CORSMiddleware

# --- 1. CONFIGURATION ---
# REPLACE WITH YOUR ACTUAL API KEY
API_KEY = "AIzaSyBO934eM6YdfbZheEPqcBY02ea821l3EqM"  

# Initialize the new Client
client = genai.Client(api_key=API_KEY)

# Force MLflow to use the database
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("Hybrid_Meeting_System")

# --- 2. SETUP ---
print("[*] Initializing Hybrid AI System...")
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("[*] Downloading SpaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

app = FastAPI(title="Hybrid AI Meeting System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. LOAD ML MODELS & SCALER ---
ml_models = {}
scaler = None
# Try both possible model locations
MODEL_DIR = "src/models" if os.path.exists("src/models") else "models"

try:
    # Load Models
    lr_path = f"{MODEL_DIR}/Linear_Regression.pkl"
    rf_path = f"{MODEL_DIR}/Random_Forest.pkl"
    xgb_path = f"{MODEL_DIR}/XGBoost.pkl"
    scaler_path = f"{MODEL_DIR}/scaler.pkl"
    
    if not os.path.exists(lr_path):
        print(f"[-] Linear Regression model not found at {lr_path}")
        print(f"    Available paths checked: {MODEL_DIR}")
    else:
        ml_models["Linear_Regression"] = joblib.load(lr_path)
        print("[+] Linear_Regression loaded.")
    
    if os.path.exists(rf_path):
        ml_models["Random_Forest"] = joblib.load(rf_path)
        print("[+] Random_Forest loaded.")
    else:
        print(f"[-] Random_Forest model not found at {rf_path}")
    
    if os.path.exists(xgb_path):
        ml_models["XGBoost"] = joblib.load(xgb_path)
        print("[+] XGBoost loaded.")
    else:
        print(f"[-] XGBoost model not found at {xgb_path}")
    
    # Load Scaler (Critical for Linear Regression accuracy)
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("[+] Scaler loaded successfully.")
    else:
        print(f"[!] Warning: Scaler not found at {scaler_path}")
    
    if ml_models:
        print(f"[+] Loaded {len(ml_models)} ML Models.")
    else:
        print("[-] WARNING: No ML models loaded! MLflow logging will be incomplete.")
except Exception as e:
    print(f"[-] Error loading resources: {e}")
    import traceback
    traceback.print_exc()
    print("   --> Hint: Run 'python src/train_models.py' to generate them.")

# --- 4. HELPER: EXTRACT FEATURES ---
def extract_features(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        all_words, clean_words = [], []
        stop_words = {"mm-hmm", "uh-huh", "hmm", "um", "uh", "ah"}

        for child in root.iter():
            if child.tag.endswith('w') and child.text:
                word = child.text.strip()
                if word:  # Only add non-empty strings
                    all_words.append(word)
                    if word.lower() not in stop_words:
                        clean_words.append(word)
        
        if not all_words:
            print("[!] Warning: No <w> tags found in XML")
            return None, "", ""
        
        full_text = " ".join(all_words)     
        gemini_text = " ".join(clean_words) 
        lower_text = full_text.lower()
        # Basic counts
        word_count = len(all_words)
        positive_count = lower_text.count("yeah") + lower_text.count("right") + lower_text.count("good") + lower_text.count("okay")
        negative_count = lower_text.count("no") + lower_text.count("sorry") + lower_text.count("but") + lower_text.count("problem")

        # Derived features - match training pipeline in src/train_models.py
        positive_negative_ratio = (positive_count + 1) / (negative_count + 1)
        sentiment_intensity = positive_count + negative_count
        # Use a safe normalization for a single-sample inference
        word_count_normalized = word_count / (word_count + 1) if word_count > 0 else 0.0
        engagement_score = (positive_count * 2 - negative_count) / (word_count + 1)

        # Construct feature frame with the same column names used during training (excluding department)
        features = {
            'word_count': word_count,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'positive_negative_ratio': positive_negative_ratio,
            'sentiment_intensity': sentiment_intensity,
            'word_count_normalized': word_count_normalized,
            'engagement_score': engagement_score
        }

        return pd.DataFrame([features]), full_text, gemini_text
    except ET.ParseError as e:
        print(f"XML Parse Error: {e}")
        return None, "", ""
    except Exception as e:
        print(f"Feature Extraction Error: {e}")
        return None, "", ""

# --- 5. ENDPOINTS ---
@app.get("/governance")
async def get_governance_metrics():
    try:
        report_path = "reports/governance_metrics.json"
        if not os.path.exists(report_path):
            return {"error": "Governance report not found. Run src/monitor.py first."}
            
        with open(report_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        return {"error": str(e)}

class ChatRequest(BaseModel):
    # Fix Pydantic 'protected_namespaces' error
    model_config = ConfigDict(protected_namespaces=()) 
    context_text: str
    question: str

@app.post("/analyze")
async def analyze_meeting(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Sanitize filename and ensure it has .xml extension
        safe_filename = file.filename.replace('/', '_').replace('\\', '_')
        if not safe_filename.endswith('.xml'):
            safe_filename += '.xml'
        
        temp_name = f"temp_{safe_filename}"
        with open(temp_name, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        df_features, full_text, gemini_text = extract_features(temp_name)
        
        if df_features is None or full_text == "":
            if os.path.exists(temp_name):
                os.remove(temp_name)
            raise HTTPException(status_code=400, detail="Failed to extract features from XML. Ensure it contains <w> tags with text.")
        
        ml_results, feature_dict = {}, {}
        
        # Always extract feature_dict from features
        feature_dict = df_features.to_dict(orient="records")[0]

        # Initialize AI variables early (before MLflow logging)
        ai_summary, ai_topics = "Summary unavailable", []
        safe_text = gemini_text[:12000] if gemini_text else ""

        # Run Gemini (Summary) FIRST - before MLflow artifact logging
        if safe_text:
            try:
                prompt = f"Analyze this transcript. Output strictly valid JSON with keys: 'summary' (string) and 'topics' (list). Transcript: {safe_text}"
                response = client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=prompt
                )

                # Robust JSON cleaning
                raw_text = response.text
                if "```json" in raw_text:
                    raw_text = raw_text.split("```json")[1].split("```")[0].strip()
                elif "```" in raw_text:
                    raw_text = raw_text.split("```")[1].split("```")[0].strip()

                data_json = json.loads(raw_text)
                ai_summary = data_json.get("summary", "No summary generated.")
                ai_topics = data_json.get("topics", [])
            except json.JSONDecodeError as e:
                ai_summary = f"Summary generation returned invalid JSON: {str(e)}"
            except Exception as e:
                ai_summary = f"Gemini Processing Error: {str(e)}"

        # Run ML Predictions
        if ml_models and df_features is not None:

            # Apply Scaling for Linear Model (robust to feature-name mismatches)
            if scaler:
                try:
                    df_scaled = scaler.transform(df_features)
                except ValueError:
                    # Attempt to align columns to the training scaler's expected features
                    expected_cols = ['word_count', 'positive_count', 'negative_count',
                                     'positive_negative_ratio', 'sentiment_intensity',
                                     'word_count_normalized', 'engagement_score']
                    # Add any missing expected columns with zeros
                    for c in expected_cols:
                        if c not in df_features.columns:
                            df_features[c] = 0.0
                    # Reorder columns to match expected
                    df_for_scale = df_features[expected_cols]
                    try:
                        df_scaled = scaler.transform(df_for_scale)
                    except Exception:
                        # Last-resort: use raw values
                        df_scaled = df_for_scale.values
            else:
                df_scaled = df_features # Fallback

            # Log to MLflow
            with mlflow.start_run(run_name=f"Manual_Analysis_{file.filename}"):
                try:
                    mlflow.log_params(feature_dict)
                except Exception as e:
                    print(f"[!] MLflow Warning - Could not log params: {e}")
                
                for name, model_obj in ml_models.items():
                    try:
                        # Use scaled data for Linear Regression
                        if name == "Linear_Regression" and scaler:
                            pred = model_obj.predict(df_scaled)[0]
                        else:
                            pred = model_obj.predict(df_features)[0]

                        ml_results[name] = round(float(pred), 2)
                        mlflow.log_metric(f"prediction_{name}", ml_results[name])
                        print(f"[+] Logged prediction for {name}: {ml_results[name]}")
                    except Exception as e:
                        print(f"[-] Error predicting with {name}: {e}")
                        ml_results[name] = 0.0

                # Log artifacts: save summary, topics, and predictions as JSON
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    try:
                        # Log AI Summary artifact
                        summary_path = os.path.join(tmpdir, "ai_summary.json")
                        with open(summary_path, "w") as f:
                            json.dump({"summary": ai_summary, "topics": ai_topics}, f, indent=2)
                        mlflow.log_artifact(summary_path, artifact_path="ai_outputs")
                        print(f"[+] Logged AI summary artifact")
                    except Exception as e:
                        print(f"[-] Error logging AI summary: {e}")

                    try:
                        # Log ML predictions artifact
                        pred_path = os.path.join(tmpdir, "ml_predictions.json")
                        with open(pred_path, "w") as f:
                            json.dump(ml_results, f, indent=2)
                        mlflow.log_artifact(pred_path, artifact_path="ml_results")
                        print(f"[+] Logged ML predictions artifact")
                    except Exception as e:
                        print(f"[-] Error logging ML predictions: {e}")

                    try:
                        # Log features artifact
                        features_path = os.path.join(tmpdir, "features.json")
                        with open(features_path, "w") as f:
                            json.dump(feature_dict, f, indent=2)
                        mlflow.log_artifact(features_path, artifact_path="ml_results")
                        print(f"[+] Logged features artifact")
                    except Exception as e:
                        print(f"[-] Error logging features: {e}")

        # Action Items via SpaCy
        action_items = []
        if full_text:
            doc = nlp(full_text[:50000])
            action_items = [s.text for s in doc.sents if any(e.label_ in ["PERSON", "DATE"] for e in s.ents)]

        if os.path.exists(temp_name):
            os.remove(temp_name)

        return {
            "gemini_summary": ai_summary,
            "gemini_topics": ai_topics,
            "ml_predictions": ml_results,
            "ml_features": feature_dict,
            "action_items": action_items[:5],
            "full_text_reference": full_text[:1000]
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("Exception in /analyze:\n", tb)
        return {"error": str(e), "traceback": tb}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=f"Context: {request.context_text[:8000]}\n\nQuestion: {request.question}"
        )
        return {"bot_answer": response.text}
    except Exception as e:
        return {"bot_answer": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)