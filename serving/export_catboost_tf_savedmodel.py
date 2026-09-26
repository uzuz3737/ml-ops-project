"""Export a registered CatBoost model to the TensorFlow Serving format."""

import json
import logging
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

logger = logging.getLogger(__name__)


class CatBoostSavedModel(tf.Module):
    """TensorFlow graph that evaluates CatBoost oblivious trees."""

    def __init__(
        self,
        model_json: dict,
        raw_features: list[str],
        engineered_features: list[str],
        config: dict,
    ) -> None:
        super().__init__()
        self.raw_features = raw_features
        self.engineered_features = engineered_features
        self.raw_feature_index = {
            name: index for index, name in enumerate(raw_features)
        }
        self.feature_config = config["feature_engineering"]
        self.category_mappings = config["features"].get("category_mappings", {})

        float_features = sorted(
            model_json["features_info"]["float_features"],
            key=lambda feature: feature["flat_feature_index"],
        )
        self.feature_names = [feature["feature_id"] for feature in float_features]
        expected_features = raw_features + engineered_features
        if self.feature_names != expected_features:
            raise ValueError(
                "CatBoost feature order differs from the configured serving pipeline: "
                f"expected {expected_features}, got {self.feature_names}"
            )

        self.trees = model_json["oblivious_trees"]
        for tree_number, tree in enumerate(self.trees):
            if any(
                split.get("split_type") != "FloatFeature" for split in tree["splits"]
            ):
                raise ValueError(
                    f"CatBoost tree {tree_number} contains a non-numeric split"
                )

        self.scale = float(model_json["scale_and_bias"][0])
        self.bias = float(model_json["scale_and_bias"][1][0])
        self.serve = tf.function(
            self._serve,
            input_signature=[
                tf.TensorSpec(
                    shape=[None, len(raw_features)],
                    dtype=tf.float64,
                    name="raw_features",
                )
            ],
        )

    def _sum_columns(self, columns: list[tf.Tensor], names: list[str]) -> tf.Tensor:
        values = [columns[self.raw_feature_index[name]] for name in names]
        total = values[0]
        for value in values[1:]:
            total = total + value
        return total

    def engineer_features(self, raw_features: tf.Tensor) -> tf.Tensor:
        """Mirror the configured shared feature engineering in the serving graph."""
        columns = [raw_features[:, index] for index in range(len(self.raw_features))]

        for name, mapping in self.category_mappings.items():
            index = self.raw_feature_index[name]
            for source, target in mapping.items():
                columns[index] = tf.where(
                    tf.equal(
                        columns[index], tf.constant(float(source), dtype=tf.float64)
                    ),
                    tf.constant(float(target), dtype=tf.float64),
                    columns[index],
                )

        utilization_config = self.feature_config["utilization"]
        utilization = tf.clip_by_value(
            columns[self.raw_feature_index[utilization_config["bill_column"]]]
            / (
                columns[self.raw_feature_index[utilization_config["limit_column"]]]
                + utilization_config["denominator_epsilon"]
            ),
            utilization_config["clip_min"],
            utilization_config["clip_max"],
        )

        months = self.feature_config["recent_months"]
        sum_bill = self._sum_columns(columns, [f"BILL_AMT{month}" for month in months])
        sum_pay = self._sum_columns(columns, [f"PAY_AMT{month}" for month in months])
        ratio_config = self.feature_config["pay_to_bill_ratio"]
        pay_to_bill_ratio = tf.clip_by_value(
            sum_pay / (tf.abs(sum_bill) + ratio_config["denominator_epsilon"]),
            ratio_config["clip_min"],
            ratio_config["clip_max"],
        )

        payment_delays = tf.stack(
            [
                columns[self.raw_feature_index[name]]
                for name in self.feature_config["payment_delay_columns"]
            ],
            axis=1,
        )
        months_delayed = tf.reduce_sum(
            tf.cast(payment_delays > 0.0, tf.float64), axis=1
        )
        max_delay_months = tf.reduce_max(payment_delays, axis=1)
        engineered_values = {
            "UTILIZATION_RATE": utilization,
            "SUM_BILL_3M": sum_bill,
            "SUM_PAY_3M": sum_pay,
            "PAY_TO_BILL_RATIO": pay_to_bill_ratio,
            "MONTHS_DELAYED": months_delayed,
            "MAX_DELAY_MONTHS": max_delay_months,
        }
        engineered = tf.stack(
            [engineered_values[name] for name in self.engineered_features], axis=1
        )
        return tf.concat([tf.stack(columns, axis=1), engineered], axis=1)

    def _serve(self, raw_features: tf.Tensor) -> dict:
        features = self.engineer_features(raw_features)
        row_count = tf.shape(features)[0]
        raw_scores = tf.zeros([row_count], dtype=tf.float64)

        for tree in self.trees:
            leaf_indices = tf.zeros([row_count], dtype=tf.int32)
            # CatBoost stores split order as the low-to-high leaf-index bit order.
            for split_index, split in enumerate(tree["splits"]):
                feature = features[:, split["float_feature_index"]]
                crossed_border = feature > tf.constant(
                    split["border"], dtype=tf.float64
                )
                leaf_indices += tf.cast(crossed_border, tf.int32) * (1 << split_index)

            leaves = tf.constant(tree["leaf_values"], dtype=tf.float64)
            raw_scores += tf.gather(leaves, leaf_indices)

        raw_scores = raw_scores * self.scale + self.bias
        positive_probability = tf.math.sigmoid(raw_scores)
        class_probabilities = tf.stack(
            [1.0 - positive_probability, positive_probability], axis=1
        )
        return {"probabilities": class_probabilities}


def _production_version(client: MlflowClient, model_name: str, stage: str):
    """Return the highest registered version currently in the configured stage."""
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        raise RuntimeError(f"No {stage} model found in MLflow Registry: {model_name}")
    return max(versions, key=lambda version: int(version.version))


def _existing_export_matches(version_path: Path, model_name: str, version: str) -> bool:
    """Reuse an already validated committed export for the same registry version."""
    metadata_path = version_path.parent / "export_metadata.json"
    saved_model_path = version_path / "saved_model.pb"
    if not saved_model_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning(
            "Could not read SavedModel metadata at %s; exporting afresh", metadata_path
        )
        return False
    return (
        metadata.get("model_name") == model_name
        and str(metadata.get("mlflow_model_version")) == str(version)
        and int(metadata.get("validated_rows", 0)) > 0
    )


def main() -> None:
    config = load_config()
    serving_config = config["serving"]
    model_name = os.getenv(
        "TF_SERVING_MODEL_NAME", serving_config["tensorflow_model_name"]
    )
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    output_root = Path(
        os.getenv("TF_SERVING_MODEL_ROOT", serving_config["tensorflow_model_root"])
    )
    registry_stage = serving_config["tensorflow_model_stage"]
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)
    registered_version = _production_version(client, model_name, registry_stage)
    version_path = output_root / model_name / str(registered_version.version)
    if _existing_export_matches(version_path, model_name, registered_version.version):
        logger.info(
            "Reusing parity-checked SavedModel %s version %s at %s",
            model_name,
            registered_version.version,
            version_path,
        )
        return

    model_uri = f"models:/{model_name}/{registered_version.version}"
    catboost_model = mlflow.catboost.load_model(model_uri)
    with tempfile.TemporaryDirectory() as temporary_directory:
        json_path = Path(temporary_directory) / "catboost_model.json"
        catboost_model.save_model(str(json_path), format="json")
        model_json = json.loads(json_path.read_text(encoding="utf-8"))

    raw_features = list(serving_config["input_features"])
    engineered_features = list(config["features"]["engineered_cols"])
    saved_model = CatBoostSavedModel(
        model_json, raw_features, engineered_features, config
    )

    root = get_project_root()
    validation_path = root / config["data"]["val_data_file"]
    if not validation_path.is_file():
        raise FileNotFoundError(
            f"Validation split is required to check export parity: {validation_path}"
        )

    validation_data = pd.read_parquet(validation_path).head(256)
    expected_features, _ = prepare_features_and_target(validation_data, config)
    model_inputs = validation_data[raw_features].to_numpy(dtype=np.float64)
    graph_probabilities = saved_model.serve(tf.convert_to_tensor(model_inputs))[
        "probabilities"
    ].numpy()[:, 1]
    reference_probabilities = catboost_model.predict_proba(expected_features)[:, 1]
    max_error = float(np.max(np.abs(graph_probabilities - reference_probabilities)))
    if max_error > 1e-10:
        raise RuntimeError(
            f"SavedModel parity check failed; max probability error={max_error}"
        )
    logger.info(
        "CatBoost-to-TensorFlow parity passed for %d rows (max probability error=%.3g)",
        len(model_inputs),
        max_error,
    )

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
        "input_features": raw_features,
        "output": "probabilities with class order [no_default, default]",
        "validated_rows": len(validation_data),
        "max_probability_error": max_error,
        "tensorflow_version": tf.__version__,
    }
    (version_path.parent / "export_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    logger.info("Exported %s to %s", model_uri, version_path)


if __name__ == "__main__":
    main()
