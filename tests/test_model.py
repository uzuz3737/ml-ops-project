"""Model metric tests; inference models come from committed MLflow artifacts."""

import numpy as np

from src.models.evaluate import compute_metrics


def test_compute_metrics():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.85, 0.2, 0.7, 0.6, 0.3])
    metrics = compute_metrics(y_true, y_prob)

    assert {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}.issubset(
        metrics
    )
    assert metrics["roc_auc"] > 0.9


def test_configured_classification_threshold_is_used():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.4, 0.6, 0.45, 0.55])

    metrics = compute_metrics(y_true, y_prob, threshold=0.6)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
