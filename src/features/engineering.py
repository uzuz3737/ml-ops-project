import numpy as np
import pandas as pd
from typing import Tuple
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes undocumented category levels in the UCI credit dataset."""
    df = df.copy()
    if "EDUCATION" in df.columns:
        # 0, 4, 5, 6 -> 4 (others)
        df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    if "MARRIAGE" in df.columns:
        # 0 -> 3 (others)
        df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates domain-specific financial features."""
    df = clean_categories(df)

    # 1. Credit utilization ratio for the most recent month
    if "BILL_AMT1" in df.columns and "LIMIT_BAL" in df.columns:
        df["UTILIZATION_RATE"] = (df["BILL_AMT1"] / (df["LIMIT_BAL"] + 1e-5)).clip(-1.0, 5.0)

    # 2. Total recent bill statements (last 3 months) vs total recent payments
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 4) if f"BILL_AMT{i}" in df.columns]
    pay_cols = [f"PAY_AMT{i}" for i in range(1, 4) if f"PAY_AMT{i}" in df.columns]
    
    if bill_cols and pay_cols:
        df["SUM_BILL_3M"] = df[bill_cols].sum(axis=1)
        df["SUM_PAY_3M"] = df[pay_cols].sum(axis=1)
        df["PAY_TO_BILL_RATIO"] = (df["SUM_PAY_3M"] / (df["SUM_BILL_3M"].abs() + 1.0)).clip(0.0, 10.0)

    # 3. Delinquency frequency: Number of months with payment delay (> 0)
    delay_cols = [f"PAY_{i}" for i in [0, 2, 3, 4, 5, 6] if f"PAY_{i}" in df.columns]
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
    df_engineered = engineer_features(df)

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
