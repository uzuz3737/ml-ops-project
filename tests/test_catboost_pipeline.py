"""Local, deterministic coverage for the CatBoost MLflow training lifecycle."""
import pandas as pd
from mlflow.tracking import MlflowClient

from src.models.train_catboost import train_catboost
from src.utils.config import load_config


def test_catboost_training_logs_metrics_artifacts_and_registers(tmp_path, monkeypatch):
    from tests.test_features import sample_dataframe
    import src.models.train_catboost as training
    import src.data.validation as validation

    monkeypatch.setattr(training, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(validation, "get_project_root", lambda: tmp_path)
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    rows = pd.concat([sample_dataframe()] * 20, ignore_index=True)
    rows["default_payment_next_month"] = [index % 2 for index in range(len(rows))]
    rows.iloc[:30].to_parquet(train_path, index=False)
    rows.iloc[30:].to_parquet(val_path, index=False)

    config = load_config()
    config["data"]["train_data_file"] = "train.parquet"
    config["data"]["val_data_file"] = "val.parquet"
    config["mlflow"]["tracking_uri"] = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    config["catboost_model"]["params"].update(iterations=8, depth=2)
    config["catboost_mlflow"].update(
        experiment_name="catboost-test", registered_model_name="CatBoostPipelineTest", min_roc_auc=0.0
    )

    is_valid, anomalies = validation.validate_with_fallback(rows.iloc[:30], rows.iloc[30:], config)
    assert is_valid, anomalies

    _, metrics = train_catboost(config)
    assert {"roc_auc", "pr_auc", "precision", "recall", "f1"}.issubset(metrics)

    client = MlflowClient(tracking_uri=config["mlflow"]["tracking_uri"])
    experiment = client.get_experiment_by_name("catboost-test")
    run = client.search_runs([experiment.experiment_id])[0]
    assert all(name in run.data.metrics for name in ("roc_auc", "pr_auc", "precision", "recall", "f1"))
    artifacts = {item.path for item in client.list_artifacts(run.info.run_id, "evaluation_plots")}
    assert "evaluation_plots/confusion_matrix.png" in artifacts
    assert "evaluation_plots/roc_curve.png" in artifacts
    assert client.get_latest_versions("CatBoostPipelineTest", stages=["Production"])

    import serving.app as serving
    monkeypatch.setattr(serving, "get_project_root", lambda: tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    config["serving"]["model_family"] = "catboost"
    serving.load_model(config)
    assert state_is_catboost(serving.state["model"])


def state_is_catboost(model):
    """Avoid depending on a CatBoost concrete estimator subclass name."""
    return model is not None and model.__class__.__module__.startswith("catboost")
