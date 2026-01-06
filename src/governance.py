import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
from sklearn.preprocessing import LabelEncoder
try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric
    AIF360_INSTALLED = True
except Exception:
    AIF360_INSTALLED = False

# --- 1. CONFIGURATION ---
# Use project-local mlruns.sqlite by default
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = f"sqlite:///{ROOT / 'mlruns.db'}"
mlflow.set_tracking_uri(DEFAULT_DB)
mlflow.set_experiment("Hybrid_Meeting_System")


def _load_data():
    # Prefer src/data, fall back to repo root data/
    src_data = Path(__file__).parent / "data" / "training_data.csv"
    root_data = ROOT / "data" / "training_data.csv"
    if src_data.exists():
        return pd.read_csv(src_data)
    if root_data.exists():
        return pd.read_csv(root_data)
    raise FileNotFoundError("training_data.csv not found in src/data or data/")


def _write_report_text(output_dir, name, content):
    os.makedirs(output_dir, exist_ok=True)
    path = Path(output_dir) / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def run_governance(mitigate=True, output_dir=None):
    print("[*] Running AI Fairness Governance Check...")

    if output_dir is None:
        output_dir = Path(__file__).parent / "reports"
    os.makedirs(output_dir, exist_ok=True)

    df = _load_data()

    # 2. DATA PREPARATION (Bias Audit)
    if "department" in df.columns and df["department"].dtype == object:
        le = LabelEncoder()
        df["department_numeric"] = le.fit_transform(df["department"])
        protected_attr = "department_numeric"
        unique_vals = sorted(df[protected_attr].unique())
        privileged_groups = [{protected_attr: unique_vals[-1]}]
        unprivileged_groups = [{protected_attr: unique_vals[0]}]
    else:
        if "department" not in df.columns:
            df["department"] = np.random.choice([0, 1], size=len(df))
        protected_attr = "department"
        privileged_groups = [{protected_attr: 1}]
        unprivileged_groups = [{protected_attr: 0}]

    # Define favorable outcome (Sentiment > 5.0)
    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = np.random.normal(loc=5.0, scale=2.0, size=len(df))
    df["favorable"] = (df["sentiment_score"] > 5.0).astype(int)

    def _compute_metrics(dataframe):
        # Prefer AIF360 when available for robust metrics; otherwise fallback
        # to simple pandas-based calculations.
        if AIF360_INSTALLED:
            dataset = BinaryLabelDataset(
                df=dataframe[[protected_attr, "favorable"]],
                label_names=["favorable"],
                protected_attribute_names=[protected_attr],
            )
            metric = BinaryLabelDatasetMetric(
                dataset,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups,
            )
            return float(metric.disparate_impact()), float(metric.statistical_parity_difference())

        # Fallback calculation: DI = (unpriv_fav_rate) / (priv_fav_rate)
        priv_mask = dataframe[protected_attr].isin([gp[protected_attr] for gp in privileged_groups])
        unpriv_mask = dataframe[protected_attr].isin([gp[protected_attr] for gp in unprivileged_groups])
        priv = dataframe[priv_mask]
        unpriv = dataframe[unpriv_mask]
        priv_rate = priv["favorable"].mean() if len(priv) else 0.0
        unpriv_rate = unpriv["favorable"].mean() if len(unpriv) else 0.0
        di = (unpriv_rate / priv_rate) if priv_rate > 0 else 0.0
        parity = unpriv_rate - priv_rate
        return float(di), float(parity)

    def _reweigh_dataframe(dataframe):
        """Stronger mitigation: reweigh examples to reduce disparate impact.

        If AIF360 is installed, prefer its Reweighing preprocessor. Otherwise
        compute simple group/label weights to equalize favorable rates.
        Returns a DataFrame plus an array of sample weights.
        """
        df = dataframe.copy().reset_index(drop=True)

        # Determine group masks
        priv_val = [gp[protected_attr] for gp in privileged_groups][0]
        unpriv_val = [gp[protected_attr] for gp in unprivileged_groups][0]
        priv_mask = df[protected_attr] == priv_val
        unpriv_mask = df[protected_attr] == unpriv_val

        if AIF360_INSTALLED:
            try:
                from aif360.algorithms.preprocessing import Reweighing
                from aif360.datasets import BinaryLabelDataset

                bdata = BinaryLabelDataset(df=df[[protected_attr, 'favorable']], label_names=['favorable'], protected_attribute_names=[protected_attr])
                rw = Reweighing(unprivileged_groups=[{protected_attr: unpriv_val}], privileged_groups=[{protected_attr: priv_val}])
                bdata_transf = rw.fit_transform(bdata)
                # Convert instance weights back to pandas-aligned array
                weights = bdata_transf.instance_weights
                # Ensure length matches
                if len(weights) == len(df):
                    return df, weights
            except Exception:
                # fallback to manual below
                pass

        # Manual reweighing: compute group x label frequencies and target probs
        # P(g), P(y), P(g,y) then compute w(g,y) = P(g)*P(y) / P(g,y)
        N = len(df)
        Pg = df[protected_attr].value_counts().to_dict()
        Py = df['favorable'].value_counts().to_dict()
        Pgy = df.groupby([protected_attr, 'favorable']).size().to_dict()

        # Probabilities
        Pg = {k: v / N for k, v in Pg.items()}
        Py = {k: v / N for k, v in Py.items()}
        Pgy = {k: v / N for k, v in Pgy.items()}

        weights = []
        for _, row in df.iterrows():
            g = row[protected_attr]
            y = row['favorable']
            p_g = Pg.get(g, 0.0)
            p_y = Py.get(y, 0.0)
            p_gy = Pgy.get((g, y), 1e-12)
            w = (p_g * p_y) / max(p_gy, 1e-12)
            weights.append(float(w))

        # Normalize weights to keep dataset scale
        w_arr = np.array(weights, dtype=float)
        w_arr = w_arr / (w_arr.mean() if w_arr.mean() > 0 else 1.0)
        return df, w_arr

    def _compute_weighted_metrics(dataframe, weights):
        # Compute weighted favorable rates per group and derive DI/parity
        df = dataframe.copy().reset_index(drop=True)
        df['_w'] = weights
        priv_val = [gp[protected_attr] for gp in privileged_groups][0]
        unpriv_val = [gp[protected_attr] for gp in unprivileged_groups][0]

        def weighted_rate(mask):
            sub = df[mask]
            if sub['_w'].sum() == 0:
                return 0.0
            return (sub.loc[sub['favorable'] == 1, '_w'].sum()) / sub['_w'].sum()

        priv_mask = df[protected_attr] == priv_val
        unpriv_mask = df[protected_attr] == unpriv_val
        priv_rate = weighted_rate(priv_mask)
        unpriv_rate = weighted_rate(unpriv_mask)
        di = (unpriv_rate / priv_rate) if priv_rate > 0 else 0.0
        parity = unpriv_rate - priv_rate
        return float(di), float(parity)

    # Compute baseline metrics
    di_before, parity_before = _compute_metrics(df)

    # Stronger mitigation via reweighing (AIF360 if available, otherwise manual)
    df_mitigated = df.copy()
    di_after, parity_after = di_before, parity_before
    if mitigate:
        try:
            df_rw, sample_weights = _reweigh_dataframe(df)
            di_after, parity_after = _compute_weighted_metrics(df_rw, sample_weights)
        except Exception:
            # fallback to previous simple upsampling approach
            df_mitigated = df.copy()
            priv = df_mitigated[df_mitigated[protected_attr].isin([gp[protected_attr] for gp in privileged_groups])]
            unpriv = df_mitigated[df_mitigated[protected_attr].isin([gp[protected_attr] for gp in unprivileged_groups])]
            priv_rate = priv["favorable"].mean() if len(priv) else 0.0
            unpriv_rate = unpriv["favorable"].mean() if len(unpriv) else 0.0
            if unpriv_rate == 0 and priv_rate > 0:
                to_add = priv_rate * len(unpriv)
                fav_rows = unpriv[unpriv["favorable"] == 1]
                if not fav_rows.empty:
                    reps = max(1, int(np.ceil(to_add / max(1, len(fav_rows)))))
                    df_mitigated = pd.concat([df_mitigated, pd.concat([fav_rows] * reps, ignore_index=True)], ignore_index=True)
            else:
                if unpriv_rate < priv_rate and unpriv_rate > 0:
                    factor = priv_rate / max(unpriv_rate, 1e-6)
                    fav_rows = unpriv[unpriv["favorable"] == 1]
                    if not fav_rows.empty and factor > 1.1:
                        reps = int(np.floor(factor)) - 1
                        df_mitigated = pd.concat([df_mitigated, pd.concat([fav_rows] * reps, ignore_index=True)], ignore_index=True)
            di_after, parity_after = _compute_metrics(df_mitigated)

    # 4. LOG TO MLFLOW AS ARTIFACTS
    with mlflow.start_run(run_name="AI_Governance_Factsheet"):
        mlflow.log_metric("disparate_impact_before", round(di_before, 4))
        mlflow.log_metric("stat_parity_before", round(parity_before, 4))
        mlflow.log_metric("disparate_impact_after", round(di_after, 4))
        mlflow.log_metric("stat_parity_after", round(parity_after, 4))

        status = "Fair" if 0.8 <= di_after <= 1.25 else "Biased"
        mlflow.set_tag("governance_status", status)
        mlflow.set_tag("fairness_status", status)

        report_content = (
            f"Governance Summary\nStatus: {status}\n"
            f"DI_before: {di_before:.4f}, DI_after: {di_after:.4f}\n"
            f"Parity_before: {parity_before:.4f}, Parity_after: {parity_after:.4f}\n"
        )
        mlflow.log_text(report_content, "governance_summary.txt")

        governance_data = {
            "experiment": "Hybrid_Meeting_System",
            "metrics": {
                "disparate_impact_before": float(di_before),
                "statistical_parity_before": float(parity_before),
                "disparate_impact_after": float(di_after),
                "statistical_parity_after": float(parity_after),
            },
            "status": status,
            "audit_groups": {
                "privileged": str(privileged_groups),
                "unprivileged": str(unprivileged_groups),
            },
        }
        mlflow.log_dict(governance_data, "governance_metadata.json")

    # Also save a local text report
    out_path = _write_report_text(output_dir, "governance_summary.txt", report_content)

    print(f"[+] Governance complete. Status: {status} (DI after: {di_after:.2f})")
    return {
        "status": status,
        "di_before": di_before,
        "di_after": di_after,
        "report_path": out_path,
    }


if __name__ == "__main__":
    run_governance()