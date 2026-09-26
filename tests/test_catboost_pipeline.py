"""Verify the existing Production CatBoost model without fitting a new model."""

import os

import mlflow.catboost
import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.features.engineering import prepare_features_and_target
from src.utils.config import load_config


def test_registered_catboost_version_one_is_loadable():
    config = load_config()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    model_name = config["catboost_mlflow"]["registered_model_name"]
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        registered = client.get_model_version(model_name, "1")
        model = mlflow.catboost.load_model(f"models:/{model_name}/1")
    except (MlflowException, OSError) as exc:
        pytest.skip(
            f"MLflow registry version 1 is not available at {tracking_uri}: {exc}"
        )

    assert registered.version == "1"
    assert registered.current_stage == "Production"

    raw_inputs = pd.DataFrame(
        [
            {
                name: (
                    24
                    if name == "AGE"
                    else 50000 if name == "LIMIT_BAL" else 1 if name == "SEX" else 2
                )
                for name in config["serving"]["input_features"]
            }
        ]
    )
    features, _ = prepare_features_and_target(raw_inputs, config)
    probabilities = model.predict_proba(features)

    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
    assert np.isclose(probabilities.sum(axis=1), 1.0).all()
