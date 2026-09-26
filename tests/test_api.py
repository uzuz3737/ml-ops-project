from fastapi.testclient import TestClient
import numpy as np
import xgboost as xgb
import pytest

from serving.app import app, state


@pytest.fixture(autouse=True)
def mock_serving_model():
    # Setup a mock trained XGBoost model for API testing
    # Generate 27 features (original 23 + 4 engineered features)
    from src.features.engineering import prepare_features_and_target
    import pandas as pd
    
    dummy_input = {
        "LIMIT_BAL": 20000.0,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 24,
        "PAY_0": 2,
        "PAY_2": 2,
        "PAY_3": -1,
        "PAY_4": -1,
        "PAY_5": -2,
        "PAY_6": -2,
        "BILL_AMT1": 3913.0,
        "BILL_AMT2": 3102.0,
        "BILL_AMT3": 689.0,
        "BILL_AMT4": 0.0,
        "BILL_AMT5": 0.0,
        "BILL_AMT6": 0.0,
        "PAY_AMT1": 0.0,
        "PAY_AMT2": 689.0,
        "PAY_AMT3": 0.0,
        "PAY_AMT4": 0.0,
        "PAY_AMT5": 0.0,
        "PAY_AMT6": 0.0,
    }
    df = pd.DataFrame([dummy_input])
    from src.utils.config import load_config
    cfg = load_config()
    X, _ = prepare_features_and_target(df, cfg)
    
    mock_model = xgb.XGBClassifier(n_estimators=3, max_depth=2, random_state=42)
    # Fit on random data with same feature shape
    X_fake = np.random.randn(20, X.shape[1])
    y_fake = np.random.randint(0, 2, size=20)
    mock_model.fit(X_fake, y_fake)
    
    state["model"] = mock_model
    state["model_version"] = "test-mock-v1"
    state["model_source"] = "mock"
    state["config"] = cfg


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_metrics_endpoint():
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text or "python_gc_objects_collected_total" in response.text


def test_single_predict_endpoint():
    client = TestClient(app)
    payload = {
        "LIMIT_BAL": 20000.0,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 24,
        "PAY_0": 2,
        "PAY_2": 2,
        "PAY_3": -1,
        "PAY_4": -1,
        "PAY_5": -2,
        "PAY_6": -2,
        "BILL_AMT1": 3913.0,
        "BILL_AMT2": 3102.0,
        "BILL_AMT3": 689.0,
        "BILL_AMT4": 0.0,
        "BILL_AMT5": 0.0,
        "BILL_AMT6": 0.0,
        "PAY_AMT1": 0.0,
        "PAY_AMT2": 689.0,
        "PAY_AMT3": 0.0,
        "PAY_AMT4": 0.0,
        "PAY_AMT5": 0.0,
        "PAY_AMT6": 0.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert "default_prediction" in res
    assert "default_probability" in res
    assert res["default_prediction"] in [0, 1]
