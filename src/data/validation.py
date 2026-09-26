import json
from pathlib import Path
import pandas as pd
import numpy as np
from src.utils.config import load_config, get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_with_tfdv(train_df: pd.DataFrame, eval_df: pd.DataFrame, config: dict):
    """Data validation using TensorFlow Data Validation (TFDV)."""
    # Apply the explicit project contract first; TFDV complements it with
    # schema statistics and distribution checks rather than replacing it.
    is_contract_valid, contract_anomalies = validate_with_fallback(train_df, eval_df, config)
    if not is_contract_valid:
        return False, contract_anomalies

    import tensorflow_data_validation as tfdv
    from google.protobuf import text_format

    root = get_project_root()
    stats_dir = root / config["data"]["stats_dir"]
    schema_dir = root / config["data"]["schema_dir"]
    stats_dir.mkdir(parents=True, exist_ok=True)
    schema_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Computing statistics using TensorFlow Data Validation (TFDV)...")
    train_stats = tfdv.generate_statistics_from_dataframe(train_df)
    eval_stats = tfdv.generate_statistics_from_dataframe(eval_df)

    # Save stats
    tfdv.write_stats_text(train_stats, str(stats_dir / "train_stats.pbtxt"))
    tfdv.write_stats_text(eval_stats, str(stats_dir / "eval_stats.pbtxt"))

    # Infer schema from train stats
    logger.info("Inferring schema from training statistics...")
    schema = tfdv.infer_schema(statistics=train_stats)
    tfdv.write_schema_text(schema, str(schema_dir / "schema.pbtxt"))

    # Check for anomalies in eval data
    logger.info("Validating evaluation dataset against inferred schema...")
    anomalies = tfdv.validate_statistics(statistics=eval_stats, schema=schema)

    anomaly_file = root / config["data"]["anomalies_file"]
    anomaly_file.parent.mkdir(parents=True, exist_ok=True)
    with open(anomaly_file, "w", encoding="utf-8") as f:
        f.write(text_format.MessageToString(anomalies))

    if anomalies.anomaly_info:
        logger.warning(f"TFDV detected anomalies: {len(anomalies.anomaly_info)} anomaly conditions found.")
        for col, info in anomalies.anomaly_info.items():
            logger.warning(f"Anomaly in {col}: {info.description}")
        return False, anomalies
    else:
        logger.info("TFDV validation PASSED! No anomalies detected.")
        return True, anomalies


def validate_with_fallback(train_df: pd.DataFrame, eval_df: pd.DataFrame, config: dict):
    """Validate required columns, types, nulls, labels, and categorical domains."""
    logger.info("Running statistical schema validator...")
    root = get_project_root()
    schema_dir = root / config["data"]["schema_dir"]
    schema_dir.mkdir(parents=True, exist_ok=True)

    anomalies = []
    target_col = config["data"]["target_col"]
    feature_cols = config["features"]["categorical_cols"] + config["features"]["numerical_cols"]
    required_cols = set(feature_cols + [target_col])

    if train_df.empty:
        anomalies.append("Training dataset is empty")
    if eval_df.empty:
        anomalies.append("Evaluation dataset is empty")

    for name, frame in (("train", train_df), ("evaluation", eval_df)):
        missing = required_cols - set(frame.columns)
        if missing:
            anomalies.append(f"{name} dataset is missing required columns: {sorted(missing)}")

    missing_eval = set(train_df.columns) - set(eval_df.columns)
    extra_eval = set(eval_df.columns) - set(train_df.columns)
    if missing_eval:
        anomalies.append(f"Evaluation dataset is missing training columns: {sorted(missing_eval)}")
    if extra_eval:
        anomalies.append(f"Evaluation dataset has unexpected columns: {sorted(extra_eval)}")

    for name, frame in (("train", train_df), ("evaluation", eval_df)):
        nulls = frame.isnull().sum()
        if nulls.sum() > 0:
            anomalies.append(f"{name} dataset contains null values: {nulls[nulls > 0].to_dict()}")

        for col in feature_cols + [target_col]:
            if col not in frame.columns:
                continue
            if not pd.api.types.is_numeric_dtype(frame[col]):
                anomalies.append(f"{name}.{col} must be numeric, got {frame[col].dtype}")
                continue
            values = frame[col].to_numpy()
            if not np.isfinite(values).all():
                anomalies.append(f"{name}.{col} contains non-finite values")

        if target_col in frame.columns:
            labels = set(frame[target_col].dropna().unique())
            invalid_labels = labels - {0, 1}
            if invalid_labels:
                anomalies.append(f"{name}.{target_col} must contain only 0/1 labels; found {sorted(invalid_labels)}")

    allowed_values = config["data"].get("allowed_values", {})
    for col, allowed in allowed_values.items():
        for name, frame in (("train", train_df), ("evaluation", eval_df)):
            if col in frame.columns:
                invalid = set(frame[col].dropna().unique()) - set(allowed)
                if invalid:
                    anomalies.append(f"{name}.{col} contains unsupported values: {sorted(invalid)}")

    for col, bounds in config["data"].get("valid_ranges", {}).items():
        minimum, maximum = bounds
        for name, frame in (("train", train_df), ("evaluation", eval_df)):
            if col in frame.columns and pd.api.types.is_numeric_dtype(frame[col]):
                outside = frame[col].notna() & ((frame[col] < minimum) | (frame[col] > maximum))
                if outside.any():
                    observed = frame.loc[outside, col].agg(["min", "max"]).to_dict()
                    anomalies.append(
                        f"{name}.{col} has values outside [{minimum}, {maximum}]: {observed}"
                    )

    # New category levels are a schema mismatch even when they happen to be
    # inside a broad legal range; log them distinctly as potential drift.
    for col in config["features"]["categorical_cols"]:
        if col in train_df.columns and col in eval_df.columns:
            unseen = set(eval_df[col].dropna().unique()) - set(train_df[col].dropna().unique())
            if unseen:
                anomalies.append(f"Evaluation has unseen category values in {col}: {sorted(unseen)}")

    # Save schema metadata
    schema = {
        "columns": {col: str(dtype) for col, dtype in train_df.dtypes.items()},
        "shape_train": list(train_df.shape),
        "shape_eval": list(eval_df.shape),
        "required_columns": sorted(required_cols),
        "allowed_values": allowed_values,
    }
    schema_file = schema_dir / "schema.json"
    with open(schema_file, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    anomaly_file = root / config["data"]["anomalies_file"]
    anomaly_file.parent.mkdir(parents=True, exist_ok=True)
    with open(anomaly_file, "w", encoding="utf-8") as f:
        f.write("\n".join(anomalies) if anomalies else "NO_ANOMALIES_DETECTED")

    if anomalies:
        logger.error("Schema validation FAILED with %s issue(s): %s", len(anomalies), anomalies)
        return False, anomalies
    else:
        logger.info("Schema validation PASSED! No anomalies detected.")
        return True, []


def run_validation(config: dict = None) -> bool:
    """Entry point for data validation."""
    if config is None:
        config = load_config()

    root = get_project_root()
    train_path = root / config["data"]["train_data_file"]
    test_path = root / config["data"]["test_data_file"]

    if not train_path.exists() or not test_path.exists():
        from src.data.ingestion import ingest_data
        train_df, _, test_df = ingest_data(config)
    else:
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)

    try:
        import tensorflow_data_validation
        return validate_with_tfdv(train_df, test_df, config)
    except ImportError:
        logger.info("TFDV not installed in current environment. Using statistical schema validator.")
        return validate_with_fallback(train_df, test_df, config)


if __name__ == "__main__":
    run_validation()
