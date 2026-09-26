"""Train and track CatBoost on the shared UCI credit default pipeline."""

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import catboost
import mlflow
import mlflow.catboost
import pandas as pd
from mlflow.models.signature import infer_signature

from src.features.engineering import prepare_features_and_target
from src.data.validation import validate_with_fallback
from src.models.evaluate import compute_metrics, plot_confusion_matrix, plot_roc_curve
from src.models.registry import register_and_promote_model
from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _sha256(path: Path) -> str:
    """Return a stable content hash for a dataset or source file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_sha256(config: dict, config_path: Path) -> str:
    """Hash the source config when available, otherwise its canonical values."""
    if config_path.is_file():
        return _sha256(config_path)
    canonical_config = json.dumps(
        config, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical_config.encode("utf-8")).hexdigest()


def train_catboost(config: dict = None):
    """Fit CatBoost on train, evaluate on validation, track, and conditionally register."""
    config = config or load_config()
    root = get_project_root()
    train_path = root / config["data"]["train_data_file"]
    val_path = root / config["data"]["val_data_file"]
    if not train_path.exists() or not val_path.exists():
        from src.data.ingestion import ingest_data

        ingest_data(config)

    train_df, val_df = pd.read_parquet(train_path), pd.read_parquet(val_path)
    is_valid, validation_issues = validate_with_fallback(train_df, val_df, config)
    if not is_valid:
        raise ValueError(
            f"Training/validation data failed schema validation: {validation_issues}"
        )

    X_train, y_train = prepare_features_and_target(train_df, config)
    X_val, y_val = prepare_features_and_target(val_df, config)
    params = config["catboost_model"]["params"].copy()
    model = catboost.CatBoostClassifier(**params)

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        config["mlflow"].get("tracking_uri", "http://localhost:8000"),
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["catboost_mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="catboost_validation_run") as run:
        mlflow.log_params(params)
        mlflow.log_params(
            {
                "features_count": X_train.shape[1],
                "train_samples": X_train.shape[0],
                "val_samples": X_val.shape[0],
                "random_state": config["data"]["random_state"],
                "catboost_version": catboost.__version__,
                "data_train_sha256": _sha256(train_path),
                "data_validation_sha256": _sha256(val_path),
                "config_sha256": _config_sha256(
                    config, root / "config" / "config.yaml"
                ),
                "code_train_sha256": _sha256(Path(__file__).resolve()),
                "code_features_sha256": _sha256(
                    Path(__file__).resolve().parents[1] / "features" / "engineering.py"
                ),
            }
        )
        try:
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = "unknown"
        mlflow.log_param("git_commit", revision)
        mlflow.log_param("python_version", os.sys.version.split()[0])

        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        probabilities = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val.to_numpy(), probabilities)
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plot_confusion_matrix(
                y_val.to_numpy(),
                (
                    probabilities >= config["evaluation"]["classification_threshold"]
                ).astype(int),
                str(tmp_path / "confusion_matrix.png"),
            )
            plot_roc_curve(
                y_val.to_numpy(), probabilities, str(tmp_path / "roc_curve.png")
            )
            mlflow.log_artifacts(str(tmp_path), artifact_path="evaluation_plots")
            environment = tmp_path / "environment.txt"
            environment.write_text(
                subprocess.check_output(
                    [os.sys.executable, "-m", "pip", "freeze"], text=True
                ),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(environment), artifact_path="reproducibility")

        artifact_path = config["catboost_mlflow"].get("artifact_path", "catboost_model")
        mlflow.catboost.log_model(
            cb_model=model,
            artifact_path=artifact_path,
            signature=infer_signature(
                X_train.head(10), model.predict(X_train.head(10))
            ),
            input_example=X_train.head(2),
        )
        mlflow.set_tags(
            {"model_family": "catboost", "validation_split": "shared_configured_split"}
        )

        threshold = config["catboost_mlflow"].get("min_roc_auc", 0.70)
        if metrics["roc_auc"] >= threshold:
            try:
                register_and_promote_model(
                    run_id=run.info.run_id,
                    model_name=config["catboost_mlflow"]["registered_model_name"],
                    artifact_path=artifact_path,
                    target_stage="Production",
                    config=config,
                )
            except Exception as exc:
                logger.warning(
                    "CatBoost met the AUC criterion but registry promotion failed: %s",
                    exc,
                )
        else:
            logger.info(
                "CatBoost ROC-AUC %.4f is below registry threshold %.4f",
                metrics["roc_auc"],
                threshold,
            )

        logger.info(
            "CatBoost MLflow run %s validation metrics: %s", run.info.run_id, metrics
        )
    return model, metrics


if __name__ == "__main__":
    train_catboost()
