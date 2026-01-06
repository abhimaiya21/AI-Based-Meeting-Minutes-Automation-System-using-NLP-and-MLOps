import json
import sys
from pathlib import Path

def validate():
    root = Path(__file__).resolve().parents[1]
    report_path = root / "reports" / "governance_metrics.json"
    
    if not report_path.exists():
        print("❌ Governance Report not found! Run src/monitor.py first.")
        sys.exit(1)
        
    with open(report_path, "r") as f:
        data = json.load(f)
        
    di = data.get("disparate_impact", 0.0)
    
    print(f"[*] Validating Governance Metrics (DI={di})...")
    
    if di < 0.8:
        print("❌ FAILED: Disparate Impact < 0.8 (Possible Drift/Bias)")
        sys.exit(1)
    elif di > 1.2:
        print("❌ FAILED: Disparate Impact > 1.2 (Possible Leakage/Overfitting)")
        sys.exit(1)
        
    print("✅ SUCCESS: Governance Checks Passed.")
    sys.exit(0)

if __name__ == "__main__":
    validate()
