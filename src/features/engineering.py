import pandas as pd
from typing import Tuple
from src.utils.config import load_config


def clean_categories(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """Standardizes undocumented category levels in the UCI credit dataset."""
    config = config or load_config()
    df = df.copy()
    for column, mapping in config["features"].get("category_mappings", {}).items():
        if column in df.columns:
            df[column] = df[column].replace(mapping)
    return df


def engineer_features(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """Creates domain-specific financial features."""
    config = config or load_config()
    df = clean_categories(df, config)
    feature_config = config["feature_engineering"]

    # 1. Credit utilization ratio for the most recent month
    utilization = feature_config["utilization"]
    bill_column = utilization["bill_column"]
    limit_column = utilization["limit_column"]
    if bill_column in df.columns and limit_column in df.columns:
        df["UTILIZATION_RATE"] = (
            df[bill_column] / (df[limit_column] + utilization["denominator_epsilon"])
        ).clip(utilization["clip_min"], utilization["clip_max"])

    # 2. Total recent bill statements (last 3 months) vs total recent payments
    recent_months = feature_config["recent_months"]
    bill_cols = [f"BILL_AMT{i}" for i in recent_months if f"BILL_AMT{i}" in df.columns]
    pay_cols = [f"PAY_AMT{i}" for i in recent_months if f"PAY_AMT{i}" in df.columns]

    if bill_cols and pay_cols:
        df["SUM_BILL_3M"] = df[bill_cols].sum(axis=1)
        df["SUM_PAY_3M"] = df[pay_cols].sum(axis=1)
        ratio = feature_config["pay_to_bill_ratio"]
        df["PAY_TO_BILL_RATIO"] = (
            df["SUM_PAY_3M"] / (df["SUM_BILL_3M"].abs() + ratio["denominator_epsilon"])
        ).clip(ratio["clip_min"], ratio["clip_max"])

    # 3. Delinquency frequency: Number of months with payment delay (> 0)
    delay_cols = [
        column
        for column in feature_config["payment_delay_columns"]
        if column in df.columns
    ]
    if delay_cols:
        df["MONTHS_DELAYED"] = (df[delay_cols] > 0).sum(axis=1)
        df["MAX_DELAY_MONTHS"] = df[delay_cols].max(axis=1)

    return df


def prepare_features_and_target(
    df: pd.DataFrame, config: dict = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Applies feature engineering and separates features and target."""
    if config is None:
        config = load_config()

    target_col = config["data"]["target_col"]
    df_engineered = engineer_features(df, config)

    if target_col in df_engineered.columns:
        y = df_engineered[target_col]
        X = df_engineered.drop(columns=[target_col])
    else:
        y = None
        X = df_engineered

    # Drop ID if present
    id_col = config["data"].get("id_col", "ID")
    if id_col in X.columns:
        X = X.drop(columns=[id_col])

    return X, y
