import pandas as pd
import numpy as np
import joblib
import os
import mlflow
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBRegressor

# Metric Imports
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, mean_squared_error,
    mean_absolute_error, r2_score
)

# Governance Imports (AIF360)
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

# --- 1. CONFIGURATION ---
# Path preservation: Keeping original SQLite and Experiment names
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("Hybrid_Meeting_System")

def train():
    print("[*] Starting Phase 3: Optimized Training & Governance Audit...")
    
    # 1. Define and Create Directories (All original paths preserved)
    data_path = "data/training_data.csv"
    models_dir = "models"
    artifacts_dir = "training_artifacts"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # 2. Check Data
    if not os.path.exists(data_path):
        print("[-] Error: Run Phase 1 (Data Generation) first.")
        return

    # Load Data
    df = pd.read_csv(data_path)
    print(f"[+] Loaded {len(df)} training samples")
    
    # --- Handle categorical department properly ---
    if 'department' in df.columns and df['department'].dtype == 'object':
        print("[+] Encoding categorical department column...")
        le = LabelEncoder()
        df['department_encoded'] = le.fit_transform(df['department'])
        # Preservation: Saving label encoder to models directory
        joblib.dump(le, os.path.join(models_dir, "label_encoder.pkl"))
        protected_attr_col = 'department_encoded'
    else:
        protected_attr_col = 'department'

    # --- ENHANCED FEATURE ENGINEERING ---
    print("[+] Engineering features...")
    df['positive_negative_ratio'] = (df['positive_count'] + 1) / (df['negative_count'] + 1)
    df['sentiment_intensity'] = df['positive_count'] + df['negative_count']
    df['word_count_normalized'] = df['word_count'] / (df['word_count'].max() + 1)
    df['engagement_score'] = (df['positive_count'] * 2 - df['negative_count']) / (df['word_count'] + 1)
    
    feature_cols = ['word_count', 'positive_count', 'negative_count', protected_attr_col,
                    'positive_negative_ratio', 'sentiment_intensity', 'word_count_normalized',
                    'engagement_score']
    
    X = df[feature_cols]
    y = df['sentiment_score']
    
    protected_attr = X[protected_attr_col].copy()
    X_features = X.drop(columns=[protected_attr_col])
    
    # Split Data with stratification
    X_train, X_test, y_train, y_test, dept_train, dept_test = train_test_split(
        X_features, y, protected_attr, test_size=0.2, random_state=42, stratify=protected_attr
    )

    # --- BIAS MITIGATION: Reweighing ---
    y_train_binary = (y_train > 5.0).astype(int)
    train_data_with_dept = X_train.copy()
    train_data_with_dept[protected_attr_col] = dept_train.values
    train_data_with_dept['favorable'] = y_train_binary.values
    
    train_dataset = BinaryLabelDataset(
        df=train_data_with_dept,
        label_names=['favorable'],
        protected_attribute_names=[protected_attr_col]
    )
    
    unique_depts = sorted(dept_train.unique())
    privileged_groups = [{protected_attr_col: unique_depts[-1]}]
    unprivileged_groups = [{protected_attr_col: unique_depts[0]}]
    
    RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    train_dataset_transformed = RW.fit_transform(train_dataset)
    sample_weights = train_dataset_transformed.instance_weights

    # 4. Feature Scaling (Preservation: Saving scaler to models directory)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

    # Enhanced Models
    models = {
        "Linear_Regression": Ridge(alpha=1.0, random_state=42),
        "Random_Forest": RandomForestRegressor(n_estimators=300, max_depth=15, max_features='sqrt', random_state=42, n_jobs=-1),
        "Gradient_Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=400, learning_rate=0.01, max_depth=6, random_state=42, verbosity=0)
    }

    print("\n[*] Training Models & Auditing for Bias...")
    
    for name, model in models.items():
        print(f"[*] Training {name}...")
        
        # Preservation: Using standard run name for Dashboard integration
        with mlflow.start_run(run_name="AI_Governance_Factsheet"):
            
            # A. Train with proper scaling and bias weights
            if name == "Linear_Regression":
                model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
                y_pred_test = model.predict(X_test_scaled)
                y_pred_train = model.predict(X_train_scaled)
            else:
                model.fit(X_train, y_train, sample_weight=sample_weights)
                y_pred_test = model.predict(X_test)
                y_pred_train = model.predict(X_train)
            
            y_pred_binary = (y_pred_test > 5.0).astype(int)
            y_test_binary = (y_test > 5.0).astype(int)
            
            # B. Metrics
            mse = mean_squared_error(y_test, y_pred_test)
            r2 = r2_score(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            overfitting_gap = train_r2 - r2
            
            # Governance Audit (AIF360)
            audit_df = pd.DataFrame({protected_attr_col: dept_test.values, 'favorable_output': y_pred_binary})
            aif_dataset = BinaryLabelDataset(df=audit_df, label_names=['favorable_output'], protected_attribute_names=[protected_attr_col])
            fairness_metric = BinaryLabelDatasetMetric(aif_dataset, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
            
            disp_impact = fairness_metric.disparate_impact()
            stat_parity = fairness_metric.statistical_parity_difference()
            gov_status = "Fair" if 0.8 <= disp_impact <= 1.25 else "Biased"

            # Log to MLflow
            mlflow.log_param("model_type", name)
            mlflow.log_metrics({"r2_score": r2, "disparate_impact": disp_impact, "stat_parity": stat_parity})
            mlflow.set_tag("governance_status", gov_status)

            # C. Artifacts (UTF-8 encoding enabled to prevent Windows crash)
            metrics_report = {"model_name": name, "r2": float(r2), "fairness": {"di": float(disp_impact), "status": gov_status}}
            metrics_json_path = os.path.join(artifacts_dir, f"{name}_metrics.json")
            with open(metrics_json_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_report, f, indent=2)
            
            audit_report = f"MODEL AUDIT REPORT: {name}\nGeneralization: {'✓ Good' if overfitting_gap < 0.1 else '⚠ Warning'}\nFairness: {gov_status}"
            audit_path = os.path.join(artifacts_dir, f"{name}_audit_report.txt")
            with open(audit_path, 'w', encoding='utf-8') as f:
                f.write(audit_report)

            # Preservation: Logging artifacts to original training_artifacts folder
            mlflow.log_artifact(metrics_json_path)
            mlflow.log_artifact(audit_path)
            joblib.dump(model, os.path.join(models_dir, f"{name}.pkl"))

    print("\n✅ Training Complete. All Unicode issues resolved and Metrics logged.")

if __name__ == "__main__":
    train()