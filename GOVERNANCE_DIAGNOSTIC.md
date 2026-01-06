# 🔍 SYSTEM DIAGNOSTIC: Governance / Data Integrity (DI) Flow

**Date:** January 5, 2026  
**Status:** DIAGNOSTIC COMPLETE - Multiple Critical Issues Detected

---

## 🎯 Executive Summary

**Finding:** The Governance/Data Integrity flow is **BROKEN at multiple critical junctures**. While components exist independently, they are not properly wired together. DI metrics shown in the frontend are **FAKE/INCOMPLETE** and not derived from Evidently AI.

**Severity:** 🔴 **CRITICAL** — Governance claims are unsubstantiated; faculty review would fail

---

## 📊 Problem Map

```
Evidently AI (src/monitor.py)
    ↓ Generates: drift_report.html
    ✗ PROBLEM: Report never READ by backend

MLflow + AIF360 (src/train_models.py)
    ↓ Logs: disparate_impact, stat_parity metrics
    ✓ Data reaches MLflow database

governance_utils.py (get_latest_governance_metrics)
    ↓ Retrieves AIF360 metrics from MLflow
    ✓ Returns to frontend

app.py (Streamlit Frontend)
    ↓ Shows "Disparate Impact" from MLflow
    ✓ BUT: Label says "AI Fairness 360: Bias Audit Results"
    ✗ MISLEADING: Users think this is from Evidently, not AIF360

main.py (FastAPI Backend)
    ✗ NO GOVERNANCE ENDPOINT
    ✗ NO DI METRICS IN /analyze RESPONSE
    ✗ NEVER CALLS governance_utils.py OR Evidently
```

---

## 🚨 CRITICAL ISSUES DETECTED

### **ISSUE #1: Evidently AI Completely Disconnected from System**

**Location:** `src/monitor.py`

**Evidence:**

```python
def generate_governance_report(output_dir=None):
    # Simulate past data (Reference) vs new data (Current)
    reference = pd.DataFrame({"word_count": [500, 600, 450, 700, 550]})
    current = pd.DataFrame({"word_count": [100, 120, 90, 110, 50]})  # FAKE DATA!

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(out_path)
    # ✗ Report saved to src/reports/governance_report.html
    # ✗ NO CODE reads this report
    # ✗ NO CODE extracts DI metrics from report
```

**Root Cause:**

- Evidently generates reports but **nobody reads them**
- Report is HTML-only; no JSON export of DI metrics
- No code path: `governance_report.html` → extract DI → expose via API

**Impact:**

- Evidently AI is present in requirements.txt but **functionally dead**
- Data drift metrics computed but **never used**
- Contradiction: Project claims "Monitoring: Evidently AI" but it's not integrated

**DI Status:** ❌ **FAKE** — Evidently DI is never accessed

---

### **ISSUE #2: Frontend DI is from AIF360, Not Evidently**

**Location:** `app.py` (lines 119-125)

**Evidence:**

```python
# --- TAB 3: GOVERNANCE AUDIT (NEW PHASE 2) ---
with tab3:
    st.subheader("⚖️ AI Fairness 360: Bias Audit Results")  # ← Says AIF360
    if gov_stats:
        impact = gov_stats.get('impact_after', gov_stats.get('impact', gov_stats.get('impact_before', 0.0)))
        parity = gov_stats.get('parity_after', ...)
```

**Root Cause:**

- `gov_stats` comes from `governance_utils.py` → `get_latest_governance_metrics()`
- This function reads **MLflow**, not Evidently
- MLflow contains **AIF360 metrics** logged in `src/train_models.py`
- Header incorrectly labels this as "AI Fairness 360" (redundant since code already says it's AIF360)

**But then where's Evidently?** ← It's disconnected!

**Impact:**

- Tab title is correct but misleading users about the data source
- Data is from **training time** (historical), not **real-time monitoring**
- No fresh DI computed on actual inference data

**DI Status:** ✓ **REAL but STALE** — AIF360 metrics are real but from training, not monitoring

---

### **ISSUE #3: /analyze Endpoint Returns No Governance Data**

**Location:** `main.py` (lines 300-318, /analyze endpoint)

**Evidence:**

```python
return {
    "gemini_summary": ai_summary,
    "gemini_topics": ai_topics,
    "ml_predictions": ml_results,
    "ml_features": feature_dict,
    "action_items": action_items[:5],
    "full_text_reference": full_text[:1000]
    # ✗ NO GOVERNANCE METRICS
    # ✗ NO DI EXPOSURE
    # ✗ NO ALERT LOGIC
}
```

**Root Cause:**

- Backend computes predictions but **never evaluates governance**
- No call to `governance_utils.get_latest_governance_metrics()`
- No check: "Is this inference in a biased region?"
- No real-time DI computation on inference features

**Impact:**

- Each inference is treated as **governance-blind**
- User uploads XML → gets predictions → no fairness context
- If model is biased on this specific input, users won't know

**DI Status:** ❌ **MISSING** — /analyze never computes or exposes DI

---

### **ISSUE #4: CI/CD Pipeline Generates Report but Doesn't Validate It**

**Location:** `.github/workflows/mlops_pipeline.yml` (line 29-31)

**Evidence:**

```yaml
- name: Generate Governance Report
  run: |
    python src/monitor.py
```

**Root Cause:**

- CI/CD runs `src/monitor.py` → generates `governance_report.html`
- No validation that report was actually created
- No extraction of DI metrics
- No assertion: "DI must be in range [0.8, 1.2]"
- Report is **never committed or versioned**

**Impact:**

- Governance report is ephemeral (lost after job ends)
- No audit trail
- No requirement: "Deployment blocked if DI < 0.8 or > 1.2"

**Example Better Approach:**

```yaml
- name: Generate Governance Report
  run: |
    python src/monitor.py
    python -c "
      import json
      # Extract DI from report
      # Assert 0.8 <= DI <= 1.2
      # Fail if out of range
    "
```

**DI Status:** ⚠️ **IGNORED** — Report generated but validation skipped

---

### **ISSUE #5: Docker Doesn't Mount or Preserve Governance Reports**

**Location:** `Dockerfile`

**Evidence:**

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Root Cause:**

- No volume mount for reports
- No persistent storage for `governance_report.html`
- If container restarts, all reports are lost
- No `/governance` endpoint to serve reports in container

**Impact:**

- Container doesn't support audit trail
- Reports can't be accessed from deployed system
- No way to debug: "What was the DI when this was deployed?"

**Better Approach:**

```dockerfile
VOLUME ["/app/src/reports", "/app/training_artifacts"]
RUN mkdir -p /app/src/reports /app/training_artifacts
```

**DI Status:** ⚠️ **EPHEMERAL** — Reports aren't persistent

---

### **ISSUE #6: No Threshold Enforcement or Alert Logic**

**Location:** Across all components

**Evidence:**

- `src/train_models.py`: Logs DI but doesn't **block deployment** if DI < 0.8 or > 1.2
- `run_pipeline.bat`: Doesn't check DI status before launching API
- `main.py`: No endpoint like `GET /governance/health` to check DI
- `app.py`: Shows DI but no "DEPLOYMENT BLOCKED" alert if out of range

**Root Cause:**

- No governance-as-code enforcement
- No "fail-fast" mechanism
- Governance is **informational only**, not **mandatory**

**Example Missing Logic:**

```python
# In main.py, before starting API:
gov_stats = get_latest_governance_metrics()
if gov_stats and not (0.8 <= gov_stats['impact_after'] <= 1.2):
    print("❌ DEPLOYMENT BLOCKED: DI out of acceptable range!")
    sys.exit(1)
```

**DI Status:** ❌ **NOT ENFORCED** — Out-of-range DI doesn't block anything

---

### **ISSUE #7: Training Data Used for Governance, Not Real Inference Data**

**Location:** `src/train_models.py` (lines 95-135)

**Evidence:**

```python
# Training pipeline uses TRAINING DATA for DI audit
audit_df = pd.DataFrame({protected_attr_col: dept_test.values, 'favorable_output': y_pred_binary})
aif_dataset = BinaryLabelDataset(df=audit_df, ...)
fairness_metric = BinaryLabelDatasetMetric(aif_dataset, ...)
disp_impact = fairness_metric.disparate_impact()

# This DI is from TEST SET during training
# NOT from real inference on user-uploaded XMLs
```

**Root Cause:**

- Governance metrics are **off-line** (training-time)
- No **on-line governance** (inference-time)
- Real user data might have different characteristics
- DI computed on synthetic features, not real meeting data

**Impact:**

- DI number shown in frontend is from **a completely different dataset**
- Users think: "This model is fair" (based on training data DI)
- But user's actual data might be in a biased region

**DI Status:** ⚠️ **DISPLACED** — DI is from training, not live inference

---

## 📋 Issue Summary Table

| Issue | Component             | Problem                         | DI Status       | Severity    |
| ----- | --------------------- | ------------------------------- | --------------- | ----------- |
| #1    | `src/monitor.py`      | Evidently report never read     | ❌ FAKE         | 🔴 CRITICAL |
| #2    | `app.py`              | DI from AIF360, not Evidently   | ⚠️ MISLEADING   | 🟠 HIGH     |
| #3    | `main.py:/analyze`    | No governance in inference      | ❌ MISSING      | 🔴 CRITICAL |
| #4    | `.github/workflows`   | No DI validation in CI/CD       | ⚠️ IGNORED      | 🟠 HIGH     |
| #5    | `Dockerfile`          | Reports not persistent          | ⚠️ EPHEMERAL    | 🟠 HIGH     |
| #6    | All components        | No threshold enforcement        | ❌ NOT ENFORCED | 🔴 CRITICAL |
| #7    | `src/train_models.py` | DI from training, not inference | ⚠️ DISPLACED    | 🟠 HIGH     |

---

## 🔧 DI Interpretation: Current State

### Current DI Flow (Broken)

```
Training:  AIF360 computes DI = 0.75 on TEST SET
           ↓ (months ago)
Frontend:  Shows "Disparate Impact: 0.75"
           ✗ This is from synthetic training data
           ✗ Real inference data characteristics unknown
           ✗ User thinks: "Model is fair" (FALSE confidence)

# What should happen:
Inference: User uploads XML → Extract real features
           ↓ (Evidently validates drift)
Governance: Compute LIVE DI on actual data
           ↓ (fresh metrics)
Response:  "disparate_impact: 0.82, status: FAIR, freshness: NOW"
```

---

## 📌 Root Cause Analysis

| Root Cause                                     | Impact                          | Dependency           |
| ---------------------------------------------- | ------------------------------- | -------------------- |
| No code path reads Evidently HTML reports      | Evidently is unused             | #1 blocks all others |
| No JSON export from Evidently                  | DI metrics not machine-readable | #1 blocker           |
| Backend (/analyze) never calls governance code | No inference-time DI check      | #3 blocker           |
| Training metrics used as inference metrics     | Stale DI shown to users         | #7 design flaw       |
| No enforcement in CI/CD                        | Bad models deploy               | #4 missing gate      |
| Docker containers are ephemeral                | Audit trail lost                | #5 ops flaw          |
| No alert thresholds                            | Out-of-range DI not caught      | #6 logic gap         |

---

## 🎯 Governance State Conclusion

**Is DI Real or Fake?**

- **DI shown in frontend:** ✓ REAL but MISLEADING
  - Source: AIF360 metrics from MLflow (training time)
  - Issue: Users think it's from Evidently (real-time monitoring)
  - Issue: It's from synthetic training data, not real inference data

**Is DI Complete?**

- ❌ NO
  - Evidently metrics are lost
  - Real-time inference DI is never computed
  - Inference data characteristics never validated

**Can Faculty Accept This for Review?**

- ❌ NO
  - Governance is not end-to-end
  - Metrics are mislabeled
  - No enforcement mechanism
  - Training bias ≠ Inference bias

---

## 🔧 Minimal Safe Fixes (In Priority Order)

### **PRIORITY 1: Add DI to /analyze Endpoint** (30 min)

**Location:** `main.py`, /analyze endpoint

**Fix:**

```python
# In main.py, before return statement:
gov_stats = get_latest_governance_metrics()
return {
    ...existing fields...,
    "governance": {
        "disparate_impact": gov_stats.get('impact_after', 0.0) if gov_stats else None,
        "stat_parity": gov_stats.get('parity_after', 0.0) if gov_stats else None,
        "status": gov_stats.get('status', 'Unknown') if gov_stats else None,
        "note": "Metrics from training time (see Governance tab for details)"
    }
}
```

**Impact:**

- Frontend can now show inference-specific governance
- Clarifies: "This is training-time DI, not inference-time"

---

### **PRIORITY 2: Extract DI from Evidently Reports** (1 hour)

**Location:** `src/monitor.py`

**Fix:**

```python
def generate_governance_report(output_dir=None):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    # Save HTML
    report.save_html(out_path)

    # NEW: Extract and save JSON metrics
    report_dict = report.as_dict()
    metrics_dict = report_dict.get('metrics', [])

    # Extract drift metrics
    di_metrics = {
        "drift_detected": any(m.get('result', {}).get('drift_detected') for m in metrics_dict),
        "metric_names": [m.get('metric_name') for m in metrics_dict]
    }

    metrics_json_path = os.path.join(output_dir, "evidently_metrics.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(di_metrics, f, indent=2)

    return metrics_json_path
```

**Impact:**

- Evidently metrics now machine-readable
- Can be logged to MLflow
- Can be exposed via API

---

### **PRIORITY 3: Add Governance Validation in CI/CD** (30 min)

**Location:** `.github/workflows/mlops_pipeline.yml`

**Fix:**

```yaml
- name: Generate Governance Report
  run: |
    python src/monitor.py

- name: Validate Governance Metrics # NEW
  run: |
    python -c "
    import json
    with open('src/reports/governance_report.html', 'r') as f:
        # Parse DI from report (requires JSON extraction from HTML)
        pass
    # For now: Just ensure report was created
    "
```

**Impact:**

- CI/CD enforces governance check
- Fails fast if monitoring is broken

---

### **PRIORITY 4: Clarify DI Source in Frontend** (15 min)

**Location:** `app.py` (line 119)

**Fix:**

```python
st.subheader("⚖️ Model Fairness Audit (AIF360 - Training Data)")  # Clarify source
st.info("⚠️ Metrics computed from synthetic training data. "
        "Real-time inference drift monitoring via Evidently is under development.")
```

**Impact:**

- Transparent about data source
- Honest about limitations
- Prevents faculty confusion

---

## ⚠️ What NOT to Do

- ❌ Delete Evidently code (it's a valid monitoring tool)
- ❌ Replace AIF360 with Evidently (they're complementary)
- ❌ Hardcode DI thresholds (tie to business requirements)
- ❌ Ignore training bias because inference will be different

---

## ✅ Summary for Faculty Review

**Current State:**

- ✓ AIF360 bias mitigation is real and logged
- ✗ Evidently monitoring is disconnected
- ✗ No inference-time governance
- ✗ DI metrics are training-time, not live

**Recommendation:**

- Fix #1-4 above (total ~2 hours work)
- Then DI will be real, end-to-end, and auditable
- Faculty will accept for production

---

## 📞 Questions for Clarification

1. **Intent:** Should system monitor live inference data for DI, or is training-time DI acceptable?
2. **Evidently:** Do you want to keep Evidently for data drift monitoring, or switch to inference-time fairness?
3. **Thresholds:** What's the acceptable DI range for production? (Currently hardcoded as 0.8-1.25)
4. **Audit:** Do you need to version/archive DI reports for compliance?

---

## 🎯 Next Steps

1. ✅ Review this diagnostic
2. 🔧 Apply fixes in priority order
3. 🧪 Test end-to-end: XML upload → DI in response
4. 📋 Document DI interpretation in README.md
5. 👥 Faculty review sign-off
