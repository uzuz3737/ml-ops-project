"""Scheduled production data-drift and delayed-label quality monitoring."""
import hashlib
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.models.evaluate import compute_metrics
from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def population_stability_index(reference: pd.Series, current: pd.Series, categorical: bool = False) -> float:
    """Compare reference and current feature distributions using PSI."""
    ref = pd.Series(reference).dropna()
    cur = pd.Series(current).dropna()
    if ref.empty or cur.empty:
        return 0.0

    if categorical:
        buckets = sorted(set(ref.astype(str)) | set(cur.astype(str)))
        ref_counts = ref.astype(str).value_counts().reindex(buckets, fill_value=0).to_numpy(dtype=float)
        cur_counts = cur.astype(str).value_counts().reindex(buckets, fill_value=0).to_numpy(dtype=float)
    else:
        edges = np.histogram_bin_edges(ref.to_numpy(dtype=float), bins="sturges")
        if len(edges) < 2 or np.allclose(edges[0], edges[-1]):
            center = float(ref.iloc[0])
            edges = np.array([center - 0.5, center + 0.5])
        ref_counts = np.histogram(ref.to_numpy(dtype=float), bins=edges)[0].astype(float)
        # Keep out-of-reference-range observations in the boundary bins so
        # severe shifts are counted instead of silently discarded.
        cur_values = np.clip(cur.to_numpy(dtype=float), edges[0], edges[-1])
        cur_counts = np.histogram(cur_values, bins=edges)[0].astype(float)

    epsilon = 1e-6
    ref_share = np.clip(ref_counts / ref_counts.sum(), epsilon, None)
    cur_share = np.clip(cur_counts / cur_counts.sum(), epsilon, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def monitor_serving(config: dict = None) -> dict:
    """Log one MLflow monitoring run from recent predictions and available labels."""
    config = config or load_config()
    root = get_project_root()
    reference_path = root / config["data"]["train_data_file"]
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference training data not found: {reference_path}")

    prediction_path = root / config["serving"]["prediction_log_file"]
    feedback_path = root / config["serving"]["feedback_log_file"]
    cutoff = datetime.now(timezone.utc) - timedelta(days=config["monitoring"].get("window_days", 7))
    quality_cutoff = datetime.now(timezone.utc) - timedelta(
        days=config["monitoring"].get("quality_window_days", 180)
    )
    all_predictions = _load_jsonl(prediction_path)
    predictions = [
        event for event in all_predictions
        if datetime.fromisoformat(event["timestamp"]) >= cutoff
    ]
    feedback = [
        item for item in _load_jsonl(feedback_path)
        if datetime.fromisoformat(item["timestamp"]) >= quality_cutoff
    ]
    prediction_by_id = {event["prediction_id"]: event for event in all_predictions}
    labeled = [
        (prediction_by_id[item["prediction_id"]], item)
        for item in feedback if item["prediction_id"] in prediction_by_id
    ]

    reference = pd.read_parquet(reference_path)
    current = pd.DataFrame([event["features"] for event in predictions])
    report = {
        "window_start_utc": cutoff.isoformat(),
        "window_end_utc": datetime.now(timezone.utc).isoformat(),
        "quality_window_start_utc": quality_cutoff.isoformat(),
        "prediction_count": len(predictions),
        "labeled_feedback_count": len(labeled),
        "features": {},
        "quality_metrics": {},
    }

    threshold = float(config["monitoring"].get("data_drift_psi_threshold", 0.2))
    drift_values = {}
    for col in config["features"]["categorical_cols"] + config["features"]["numerical_cols"]:
        if col in reference.columns and col in current.columns and not current.empty:
            psi = population_stability_index(
                reference[col], current[col], categorical=col in config["features"]["categorical_cols"]
            )
            drift_values[col] = psi
            report["features"][col] = {"psi": psi, "alert": psi >= threshold}

    if labeled:
        y_true = np.asarray([item["actual_default"] for _, item in labeled], dtype=int)
        y_pred = np.asarray([event["prediction"] for event, _ in labeled], dtype=int)
        probabilities = np.asarray([event["probability"] for event, _ in labeled], dtype=float)
        quality = {
            "accuracy": float(np.mean(y_true == y_pred)),
            "precision": float(np.sum((y_true == 1) & (y_pred == 1)) / max(np.sum(y_pred == 1), 1)),
            "recall": float(np.sum((y_true == 1) & (y_pred == 1)) / max(np.sum(y_true == 1), 1)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        if len(np.unique(y_true)) == 2:
            full_metrics = compute_metrics(y_true, probabilities)
            quality.update({"roc_auc": full_metrics["roc_auc"], "pr_auc": full_metrics["pr_auc"]})
        report["quality_metrics"] = quality

    minimum_feedback = int(config["monitoring"].get("min_labeled_feedback", 20))
    quality_ready = len(labeled) >= minimum_feedback and "roc_auc" in report["quality_metrics"]
    report["quality_status"] = "ready" if quality_ready else "insufficient_labeled_feedback_or_class_coverage"

    drift_alert = bool(drift_values) and max(drift_values.values()) >= threshold
    quality_alert = (
        quality_ready
        and report["quality_metrics"]["roc_auc"] < float(config["monitoring"].get("min_roc_auc", 0.70))
    )
    report["alerts"] = {
        "data_drift": drift_alert,
        "prediction_quality": quality_alert,
        "drift_threshold_psi": threshold,
        "quality_min_roc_auc": float(config["monitoring"].get("min_roc_auc", 0.70)),
        "quality_min_feedback": int(config["monitoring"].get("min_labeled_feedback", 20)),
    }

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"].get("tracking_uri", "http://localhost:5000"))
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = config["monitoring"].get(
        "experiment_name", "credit_card_default_production_monitoring"
    )
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="serving_monitoring_window") as run:
        mlflow.log_params({
            "window_days": config["monitoring"].get("window_days", 7),
            "quality_window_days": config["monitoring"].get("quality_window_days", 180),
            "reference_data_sha256": _sha256(reference_path),
            "data_drift_psi_threshold": threshold,
            "min_labeled_feedback": config["monitoring"].get("min_labeled_feedback", 20),
            "min_roc_auc": config["monitoring"].get("min_roc_auc", 0.70),
        })
        metrics = {
            "prediction_count": len(predictions),
            "labeled_feedback_count": len(labeled),
            "drifted_feature_count": sum(value >= threshold for value in drift_values.values()),
            "max_feature_psi": max(drift_values.values(), default=0.0),
            "data_drift_alert": int(drift_alert),
            "prediction_quality_alert": int(quality_alert),
            "prediction_quality_monitor_ready": int(quality_ready),
        }
        if labeled:
            metrics.update({f"quality_{name}": value for name, value in report["quality_metrics"].items()})
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        mlflow.set_tags({
            "monitoring_type": "production_data_drift_and_delayed_label_quality",
            "model_family": config.get("serving", {}).get("model_family", "xgboost"),
            "data_drift_alert": str(drift_alert).lower(),
            "prediction_quality_alert": str(quality_alert).lower(),
            "quality_monitor_status": report["quality_status"],
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / "monitoring_report.json"
            artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(artifact), artifact_path="monitoring")

    logger.info("MLflow monitoring run %s logged: %s", run.info.run_id, metrics)
    return {"run_id": run.info.run_id, "metrics": metrics, "report": report}


if __name__ == "__main__":
    monitor_serving()
