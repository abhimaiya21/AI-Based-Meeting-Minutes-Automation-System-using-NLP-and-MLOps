import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

# --- CONFIGURATION ---
# Prefer project-local mlruns DB when present
ROOT = Path(__file__).resolve().parents[0]
DEFAULT_DB = f"sqlite:///{ROOT / 'mlruns.db'}"
EXPERIMENT_NAME = "Hybrid_Meeting_System"

client = MlflowClient(tracking_uri=DEFAULT_DB)


def get_latest_governance_metrics():
    """Retrieve the latest governance run metrics from MLflow.

    Looks specifically for runs named 'AI_Governance_Factsheet' and returns
    standardized metric keys used by `src/governance.py`.
    """
    try:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            return None

        # Use correct tag filter syntax: `tag.<key> = 'value'`
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tag.mlflow.runName = 'AI_Governance_Factsheet'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if not runs:
            return None

        latest_run = runs[0]
        metrics = latest_run.data.metrics
        tags = latest_run.data.tags

        # Map metric names to a friendly structure
        return {
            "impact_before": round(metrics.get("disparate_impact_before", 0.0), 4),
            "parity_before": round(metrics.get("stat_parity_before", metrics.get("statistical_parity_before", 0.0)), 4),
            "impact_after": round(metrics.get("disparate_impact_after", 0.0), 4),
            "parity_after": round(metrics.get("stat_parity_after", metrics.get("statistical_parity_after", 0.0)), 4),
            "status": tags.get("governance_status", "Unknown"),
            "run_id": latest_run.info.run_id,
            "run_name": tags.get("mlflow.runName", "AI_Governance_Factsheet"),
            "timestamp": latest_run.info.start_time,
        }

    except Exception as e:
        print(f"❌ Governance Data Fetch Error: {e}")
        return None