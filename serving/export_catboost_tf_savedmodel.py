"""Export the registered CatBoost classifier as an equivalent TF SavedModel.

TensorFlow Serving loads SavedModel graphs, not CatBoost binaries. This exporter
turns the existing CatBoost oblivious trees into TensorFlow graph operations,
and keeps the project's raw-feature engineering inside the serving signature.
"""
import json
import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.catboost
import numpy as np
import pandas as pd
import tensorflow as tf
from mlflow.tracking import MlflowClient

from src.features.engineering import prepare_features_and_target
from src.utils.config import get_project_root, load_config


RAW_FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2",
    "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2",
    "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1",
    "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
ENGINEERED_FEATURES = [
    "UTILIZATION_RATE", "SUM_BILL_3M", "SUM_PAY_3M", "PAY_TO_BILL_RATIO",
    "MONTHS_DELAYED", "MAX_DELAY_MONTHS",
]


class CatBoostSavedModel(tf.Module):
    """TensorFlow graph that evaluates CatBoost trees and returns class scores."""

    def __init__(self, model_json: dict):
        super().__init__()
        float_features = sorted(
            model_json["features_info"]["float_features"],
            key=lambda feature: feature["flat_feature_index"],
        )
        self.feature_names = [feature["feature_id"] for feature in float_features]
        expected_features = RAW_FEATURES + ENGINEERED_FEATURES
        if self.feature_names != expected_features:
            raise ValueError(
                "CatBoost feature order differs from the shared feature pipeline: "
                f"expected {expected_features}, got {self.feature_names}"
            )

        self.trees = model_json["oblivious_trees"]
        for tree_number, tree in enumerate(self.trees):
            if any(split.get("split_type") != "FloatFeature" for split in tree["splits"]):
                raise ValueError(f"CatBoost tree {tree_number} contains a non-numeric split")

        self.scale = float(model_json["scale_and_bias"][0])
        self.bias = float(model_json["scale_and_bias"][1][0])
        self.serve = tf.function(
            self._serve,
            input_signature=[
                tf.TensorSpec(shape=[None, len(RAW_FEATURES)], dtype=tf.float64, name="raw_features")
            ],
        )

    @staticmethod
    def engineer_features(raw_features: tf.Tensor) -> tf.Tensor:
        """Match src.features.engineering.engineer_features for each row."""
        columns = [raw_features[:, index] for index in range(len(RAW_FEATURES))]

        # Match the existing category cleanup: undocumented values map to "other".
        education = columns[2]
        education_is_other = tf.logical_or(
            tf.equal(education, tf.constant(0.0, dtype=tf.float64)),
            tf.logical_or(
                tf.equal(education, tf.constant(5.0, dtype=tf.float64)),
                tf.equal(education, tf.constant(6.0, dtype=tf.float64)),
            ),
        )
        columns[2] = tf.where(
            education_is_other, tf.constant(4.0, dtype=tf.float64), education
        )
        columns[3] = tf.where(
            tf.equal(columns[3], tf.constant(0.0, dtype=tf.float64)),
            tf.constant(3.0, dtype=tf.float64),
            columns[3],
        )

        utilization = tf.clip_by_value(
            columns[11] / (columns[0] + 1e-5), -1.0, 5.0
        )
        sum_bill_3m = columns[11] + columns[12] + columns[13]
        sum_pay_3m = columns[17] + columns[18] + columns[19]
        pay_to_bill = tf.clip_by_value(
            sum_pay_3m / (tf.abs(sum_bill_3m) + 1.0), 0.0, 10.0
        )
        payment_delays = tf.stack(
            [columns[index] for index in (5, 6, 7, 8, 9, 10)], axis=1
        )
        months_delayed = tf.reduce_sum(
            tf.cast(payment_delays > 0.0, tf.float64), axis=1
        )
        max_delay_months = tf.reduce_max(payment_delays, axis=1)
        engineered = tf.stack(
            [utilization, sum_bill_3m, sum_pay_3m, pay_to_bill, months_delayed, max_delay_months],
            axis=1,
        )
        return tf.concat([tf.stack(columns, axis=1), engineered], axis=1)

    def _serve(self, raw_features: tf.Tensor) -> dict:
        features = self.engineer_features(raw_features)
        row_count = tf.shape(features)[0]
        raw_scores = tf.zeros([row_count], dtype=tf.float64)

        for tree in self.trees:
            leaf_indices = tf.zeros([row_count], dtype=tf.int32)
            # CatBoost uses split order as the low-to-high bit order in leaf index.
            for split_index, split in enumerate(tree["splits"]):
                feature = features[:, split["float_feature_index"]]
                crossed_border = feature > tf.constant(split["border"], dtype=tf.float64)
                leaf_indices += tf.cast(crossed_border, tf.int32) * (1 << split_index)

            leaves = tf.constant(tree["leaf_values"], dtype=tf.float64)
            raw_scores += tf.gather(leaves, leaf_indices)

        raw_scores = raw_scores * self.scale + self.bias
        positive_probability = tf.math.sigmoid(raw_scores)
        class_probabilities = tf.stack(
            [1.0 - positive_probability, positive_probability], axis=1
        )
        return {"probabilities": class_probabilities}


def main() -> None:
    config = load_config()
    model_config = config["catboost_mlflow"]
    model_name = os.getenv("TF_SERVING_MODEL_NAME", model_config["registered_model_name"])
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    output_root = Path(os.getenv("TF_SERVING_MODEL_ROOT", "/models"))
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)
    production_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not production_versions:
        raise RuntimeError(f"No Production model found in MLflow Registry: {model_name}")
    registered_version = max(production_versions, key=lambda version: int(version.version))
    model_uri = f"models:/{model_name}/{registered_version.version}"
    catboost_model = mlflow.catboost.load_model(model_uri)

    with tempfile.TemporaryDirectory() as temporary_directory:
        json_path = Path(temporary_directory) / "catboost_model.json"
        catboost_model.save_model(str(json_path), format="json")
        model_json = json.loads(json_path.read_text(encoding="utf-8"))

    saved_model = CatBoostSavedModel(model_json)

    # Compare against the production CatBoost artifact using the shared pipeline.
    root = get_project_root()
    validation_path = root / config["data"]["val_data_file"]
    if validation_path.is_file():
        validation_data = pd.read_parquet(validation_path).head(256)
        expected_features, _ = prepare_features_and_target(validation_data, config)
        model_inputs = validation_data[RAW_FEATURES].to_numpy(dtype=np.float64)
        graph_probabilities = saved_model.serve(tf.convert_to_tensor(model_inputs))["probabilities"].numpy()[:, 1]
        reference_probabilities = catboost_model.predict_proba(expected_features)[:, 1]
        max_error = float(np.max(np.abs(graph_probabilities - reference_probabilities)))
        if max_error > 1e-10:
            raise RuntimeError(f"SavedModel parity check failed; max probability error={max_error}")
        print(f"CatBoost-to-TensorFlow parity passed for {len(model_inputs)} rows (max error={max_error:.3g})")
    else:
        raise FileNotFoundError(f"Validation split is required to check export parity: {validation_path}")

    version_path = output_root / model_name / str(registered_version.version)
    version_path.parent.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(
        saved_model,
        str(version_path),
        signatures={"serving_default": saved_model.serve.get_concrete_function()},
    )
    metadata = {
        "model_name": model_name,
        "mlflow_model_version": registered_version.version,
        "mlflow_run_id": registered_version.run_id,
        "feature_names": saved_model.feature_names,
        "input_features": RAW_FEATURES,
        "output": "probabilities with class order [no_default, default]",
        "validated_rows": int(len(validation_data)),
        "tensorflow_version": tf.__version__,
    }
    (version_path.parent / "export_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Exported {model_uri} to {version_path}")


if __name__ == "__main__":
    main()
