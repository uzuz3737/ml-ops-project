"""Tests for serving outcome capture and MLflow monitoring reports."""
import json

import pandas as pd
from mlflow.tracking import MlflowClient

from src.models.monitor import monitor_serving
from src.utils.config import load_config
from serving.monitoring import record_feedback, record_prediction


def test_serving_drift_and_quality_are_logged_to_mlflow(tmp_path, monkeypatch):
    from tests.test_features import sample_dataframe
    import serving.monitoring as serving_monitoring
    import src.models.monitor as model_monitoring

    monkeypatch.setattr(serving_monitoring, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(model_monitoring, "get_project_root", lambda: tmp_path)
    config = load_config()
    config["mlflow"]["tracking_uri"] = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    config["monitoring"].update(
        experiment_name="production-monitoring-test",
        data_drift_psi_threshold=0.1,
        min_labeled_feedback=2,
        min_roc_auc=0.99,
    )

    training_data = pd.concat([sample_dataframe()] * 20, ignore_index=True)
    reference_path = tmp_path / config["data"]["train_data_file"]
    reference_path.parent.mkdir(parents=True)
    training_data.to_parquet(reference_path, index=False)

    ids = []
    for probability, actual in ((0.8, 0), (0.7, 1), (0.6, 0), (0.9, 1)):
        features = training_data.iloc[0].drop(config["data"]["target_col"]).to_dict()
        features["LIMIT_BAL"] = 1_000_000.0  # deliberate feature drift from reference
        prediction_id = record_prediction(features, probability, 1, "test-model", config)
        record_feedback(prediction_id, actual, config)
        ids.append(prediction_id)

    result = monitor_serving(config)
    assert result["report"]["alerts"]["data_drift"]
    assert result["report"]["alerts"]["prediction_quality"]
    assert result["metrics"]["prediction_count"] == 4
    assert result["metrics"]["labeled_feedback_count"] == 4

    client = MlflowClient(tracking_uri=config["mlflow"]["tracking_uri"])
    experiment = client.get_experiment_by_name("production-monitoring-test")
    run = client.get_run(result["run_id"])
    assert run.info.experiment_id == experiment.experiment_id
    assert run.data.metrics["data_drift_alert"] == 1
    assert run.data.metrics["prediction_quality_alert"] == 1
    artifacts = client.list_artifacts(result["run_id"], "monitoring")
    assert any(item.path.endswith("monitoring_report.json") for item in artifacts)
    report_path = client.download_artifacts(result["run_id"], "monitoring/monitoring_report.json")
    assert json.loads(open(report_path, encoding="utf-8").read())["labeled_feedback_count"] == len(ids)
