import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature
from pathlib import Path

from src.utils.config import load_config, get_project_root
from src.utils.logger import get_logger
from src.features.engineering import prepare_features_and_target
from src.models.evaluate import compute_metrics, plot_confusion_matrix, plot_roc_curve
from src.models.registry import register_and_promote_model

logger = get_logger(__name__)


def setup_mlflow(config: dict):
    """Initializes MLflow tracking URI and experiment."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"].get("tracking_uri", "http://localhost:5000"))
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = config["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow configured with URI: {tracking_uri}, Experiment: {experiment_name}")


def train_model(config: dict = None) -> tuple[xgb.XGBClassifier, dict]:
    """Trains XGBoost model, tracks run in MLflow, and registers production model."""
    if config is None:
        config = load_config()

    root = get_project_root()
    train_path = root / config["data"]["train_data_file"]
    val_path = root / config["data"]["val_data_file"]

    if not train_path.exists():
        from src.data.ingestion import ingest_data
        ingest_data(config)

    logger.info("Loading processed training and validation sets...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    logger.info("Applying feature engineering...")
    X_train, y_train = prepare_features_and_target(train_df, config)
    X_val, y_val = prepare_features_and_target(val_df, config)

    # Initialize XGBoost model
    model_params = config["model"]["params"].copy()
    logger.info(f"Instantiating XGBClassifier with params: {model_params}")
    model = xgb.XGBClassifier(**model_params)

    # Setup MLflow
    try:
        setup_mlflow(config)
    except Exception as e:
        logger.warning(f"Could not connect to MLflow server: {e}. Local file store will be used.")

    # Run MLflow Tracking
    with mlflow.start_run(run_name="xgboost_baseline_run") as run:
        # Log parameters
        mlflow.log_params(model_params)
        mlflow.log_param("features_count", X_train.shape[1])
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("val_samples", X_val.shape[0])

        logger.info("Fitting XGBoost model...")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        # Predictions & Probabilities
        val_pred_prob = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val.to_numpy(), val_pred_prob)
        logger.info(f"Validation Metrics: {metrics}")

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)

        # Generate and log evaluation plots
        temp_dir = root / "models" / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        cm_path = temp_dir / "confusion_matrix.png"
        roc_path = temp_dir / "roc_curve.png"

        plot_confusion_matrix(y_val.to_numpy(), (val_pred_prob >= 0.5).astype(int), str(cm_path))
        plot_roc_curve(y_val.to_numpy(), val_pred_prob, str(roc_path))

        mlflow.log_artifact(str(cm_path), artifact_path="evaluation_plots")
        mlflow.log_artifact(str(roc_path), artifact_path="evaluation_plots")

        # Save model locally for fallback serving
        saved_models_dir = root / "models" / "saved"
        saved_models_dir.mkdir(parents=True, exist_ok=True)
        local_model_path = saved_models_dir / "xgb_model.joblib"
        joblib.dump(model, local_model_path)
        logger.info(f"Model saved locally to {local_model_path}")

        # Log model to MLflow with signature
        signature = infer_signature(X_train.head(10), model.predict(X_train.head(10)))
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="xgboost_model",
            signature=signature,
            input_example=X_train.head(2),
        )
        logger.info(f"MLflow run completed with run_id: {run.info.run_id}")

        # Model Registry Promotion Criteria (e.g. ROC AUC >= 0.70)
        min_auc_threshold = 0.70
        if metrics["roc_auc"] >= min_auc_threshold:
            reg_name = config["mlflow"]["registered_model_name"]
            try:
                register_and_promote_model(
                    run_id=run.info.run_id,
                    model_name=reg_name,
                    artifact_path="xgboost_model",
                    target_stage="Production",
                )
            except Exception as e:
                logger.warning(f"Could not register model into MLflow registry: {e}")
        else:
            logger.warning(
                f"Model ROC AUC ({metrics['roc_auc']:.3f}) did not meet criteria ({min_auc_threshold}). "
                "Skipping registry promotion."
            )

    return model, metrics


if __name__ == "__main__":
    train_model()
