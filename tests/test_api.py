"""FastAPI contract tests using a fixed predictor, with no estimator fitting."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from serving.app import app, state
from src.utils.config import load_config


CUSTOMER = {
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


class FixedProbabilityModel:
    """Return stable class probabilities without fitting a model in tests."""

    def predict_proba(self, features):
        return np.tile(np.array([[0.25, 0.75]]), (len(features), 1))


@pytest.fixture(autouse=True)
def mock_serving_model(tmp_path, monkeypatch):
    import serving.monitoring as serving_monitoring

    monkeypatch.setattr(serving_monitoring, "get_project_root", lambda: tmp_path)
    state.update(
        model=FixedProbabilityModel(),
        model_version="test-fixed-v1",
        model_source="test-fixture",
        config=load_config(),
    )


def test_health_endpoint():
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_metrics_endpoint():
    response = TestClient(app).get("/metrics")

    assert response.status_code == 200
    assert (
        "http_requests_total" in response.text
        or "python_gc_objects_collected_total" in response.text
    )


def test_single_predict_endpoint():
    client = TestClient(app)
    response = client.post("/predict", json=CUSTOMER)

    assert response.status_code == 200
    prediction = response.json()
    assert prediction["default_prediction"] == 1
    assert prediction["default_probability"] == 0.75
    assert prediction["prediction_id"]

    feedback = client.post(
        "/feedback",
        json={"prediction_id": prediction["prediction_id"], "actual_default": 1},
    )
    assert feedback.status_code == 200
    duplicate = client.post(
        "/feedback",
        json={"prediction_id": prediction["prediction_id"], "actual_default": 1},
    )
    assert duplicate.status_code == 409
