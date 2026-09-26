import os
import json
from pathlib import Path
import pandas as pd
from src.utils.config import load_config, get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_with_tfdv(train_df: pd.DataFrame, eval_df: pd.DataFrame, config: dict):
    """Data validation using TensorFlow Data Validation (TFDV)."""
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
    """Statistical and schema validation engine (used when TFDV is unavailable in local OS)."""
    logger.info("Running statistical schema validator...")
    root = get_project_root()
    schema_dir = root / config["data"]["schema_dir"]
    schema_dir.mkdir(parents=True, exist_ok=True)

    anomalies = []

    # 1. Missing columns
    missing_cols = set(train_df.columns) - set(eval_df.columns)
    if missing_cols:
        anomalies.append(f"Missing columns in evaluation dataset: {missing_cols}")

    # 2. Null values
    train_nulls = train_df.isnull().sum()
    eval_nulls = eval_df.isnull().sum()
    if eval_nulls.sum() > 0:
        anomalies.append(f"Unexpected null values found in eval set: {eval_nulls[eval_nulls > 0].to_dict()}")

    # 3. Categorical range checks
    for cat_col in config["features"]["categorical_cols"]:
        if cat_col in train_df.columns and cat_col in eval_df.columns:
            train_vals = set(train_df[cat_col].unique())
            eval_vals = set(eval_df[cat_col].unique())
            unexpected = eval_vals - train_vals
            if unexpected:
                anomalies.append(f"Categorical drift/unseen values in {cat_col}: {unexpected}")

    # 4. Numerical range/outlier checks
    for num_col in config["features"]["numerical_cols"]:
        if num_col in train_df.columns and num_col in eval_df.columns:
            min_val = train_df[num_col].quantile(0.001)
            max_val = train_df[num_col].quantile(0.999)
            eval_out_of_bounds = ((eval_df[num_col] < min_val * 2) | (eval_df[num_col] > max_val * 2)).sum()
            if eval_out_of_bounds > 0:
                logger.debug(f"{eval_out_of_bounds} extreme values in {num_col}")

    # Save schema metadata
    schema = {
        "columns": {col: str(dtype) for col, dtype in train_df.dtypes.items()},
        "shape_train": list(train_df.shape),
        "shape_eval": list(eval_df.shape),
    }
    schema_file = schema_dir / "schema.json"
    with open(schema_file, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    anomaly_file = root / config["data"]["anomalies_file"]
    with open(anomaly_file, "w", encoding="utf-8") as f:
        f.write("\n".join(anomalies) if anomalies else "NO_ANOMALIES_DETECTED")

    if anomalies:
        logger.warning(f"Schema validator found {len(anomalies)} warnings: {anomalies}")
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
