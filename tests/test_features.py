import pandas as pd
from src.features.engineering import clean_categories, engineer_features, prepare_features_and_target


def sample_dataframe():
    return pd.DataFrame({
        "LIMIT_BAL": [50000.0, 20000.0],
        "SEX": [1, 2],
        "EDUCATION": [0, 5],
        "MARRIAGE": [0, 1],
        "AGE": [30, 25],
        "PAY_0": [0, 2],
        "PAY_2": [0, 2],
        "PAY_3": [0, -1],
        "PAY_4": [0, -1],
        "PAY_5": [0, -2],
        "PAY_6": [0, -2],
        "BILL_AMT1": [10000.0, 5000.0],
        "BILL_AMT2": [8000.0, 4000.0],
        "BILL_AMT3": [6000.0, 3000.0],
        "BILL_AMT4": [0.0, 0.0],
        "BILL_AMT5": [0.0, 0.0],
        "BILL_AMT6": [0.0, 0.0],
        "PAY_AMT1": [2000.0, 0.0],
        "PAY_AMT2": [2000.0, 500.0],
        "PAY_AMT3": [2000.0, 0.0],
        "PAY_AMT4": [0.0, 0.0],
        "PAY_AMT5": [0.0, 0.0],
        "PAY_AMT6": [0.0, 0.0],
        "default_payment_next_month": [0, 1],
    })


def test_clean_categories():
    df = sample_dataframe()
    cleaned = clean_categories(df)
    assert set(cleaned["EDUCATION"].unique()).issubset({1, 2, 3, 4})
    assert set(cleaned["MARRIAGE"].unique()).issubset({1, 2, 3})


def test_engineer_features():
    df = sample_dataframe()
    eng = engineer_features(df)
    assert "UTILIZATION_RATE" in eng.columns
    assert "SUM_BILL_3M" in eng.columns
    assert "SUM_PAY_3M" in eng.columns
    assert "PAY_TO_BILL_RATIO" in eng.columns
    assert "MONTHS_DELAYED" in eng.columns
    assert eng["MONTHS_DELAYED"].iloc[1] >= 1


def test_prepare_features_and_target():
    df = sample_dataframe()
    X, y = prepare_features_and_target(df)
    assert "default_payment_next_month" not in X.columns
    assert len(y) == 2
    assert list(y) == [0, 1]
