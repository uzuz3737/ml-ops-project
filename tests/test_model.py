import numpy as np
import xgboost as xgb
from src.models.evaluate import compute_metrics


def test_compute_metrics():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.85, 0.2, 0.7, 0.6, 0.3])
    metrics = compute_metrics(y_true, y_prob)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] > 0.9


def test_xgboost_fit_predict():
    X = np.random.randn(50, 10)
    y = np.random.randint(0, 2, size=50)

    model = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    probs = model.predict_proba(X)

    assert len(preds) == 50
    assert probs.shape == (50, 2)
