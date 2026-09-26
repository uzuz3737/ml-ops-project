"""Data contract validation tests for clean and intentionally corrupt batches."""
import pandas as pd

from src.data.validation import validate_with_fallback
from src.utils.config import load_config


def test_schema_validation_accepts_valid_train_and_evaluation(tmp_path, monkeypatch):
    from tests.test_features import sample_dataframe
    import src.data.validation as validation

    monkeypatch.setattr(validation, "get_project_root", lambda: tmp_path)
    clean = pd.concat([sample_dataframe()] * 4, ignore_index=True)

    is_valid, anomalies = validate_with_fallback(clean.iloc[:4], clean.iloc[4:], load_config())

    assert is_valid
    assert anomalies == []
    assert (tmp_path / "data/validation/schema/schema.json").exists()
    assert (tmp_path / "data/validation/anomalies.txt").read_text(encoding="utf-8") == "NO_ANOMALIES_DETECTED"


def test_schema_validation_rejects_bad_values_and_missing_columns(tmp_path, monkeypatch):
    from tests.test_features import sample_dataframe
    import src.data.validation as validation

    monkeypatch.setattr(validation, "get_project_root", lambda: tmp_path)
    train = pd.concat([sample_dataframe()] * 3, ignore_index=True)
    evaluation = train.copy()
    evaluation.loc[0, "SEX"] = 8
    evaluation.loc[1, "default_payment_next_month"] = 2
    evaluation.loc[2, "AGE"] = 500
    evaluation.loc[3, "PAY_AMT1"] = None
    evaluation = evaluation.drop(columns=["LIMIT_BAL"])

    is_valid, anomalies = validate_with_fallback(train, evaluation, load_config())

    assert not is_valid
    assert any("missing required columns" in item for item in anomalies)
    assert any("unsupported values" in item for item in anomalies)
    assert any("only 0/1 labels" in item for item in anomalies)
    assert any("null values" in item for item in anomalies)
    assert any("outside [18, 120]" in item for item in anomalies)
    saved_anomalies = (tmp_path / "data/validation/anomalies.txt").read_text(encoding="utf-8")
    assert "NO_ANOMALIES_DETECTED" not in saved_anomalies
