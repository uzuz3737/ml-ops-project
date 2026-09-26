"""Append-only inference and outcome events used by scheduled ML monitoring."""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.utils.config import get_project_root

_EVENT_LOCK = threading.RLock()


def _event_path(config: dict, key: str) -> Path:
    path = Path(config["serving"][key])
    return path if path.is_absolute() else get_project_root() / path


def _append_jsonl(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _EVENT_LOCK, path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


def record_prediction(features: dict, probability: float, prediction: int, model_version: str, config: dict) -> str:
    """Persist a score event without customer identifiers and return its feedback key."""
    prediction_id = uuid4().hex
    event = {
        "prediction_id": prediction_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "probability": float(probability),
        "prediction": int(prediction),
        "model_version": str(model_version),
    }
    _append_jsonl(_event_path(config, "prediction_log_file"), event)
    return prediction_id


def record_feedback(prediction_id: str, actual_default: int, config: dict) -> None:
    """Persist a delayed outcome for a known prediction; reject unknown/duplicate IDs."""
    with _EVENT_LOCK:
        predictions_path = _event_path(config, "prediction_log_file")
        if not predictions_path.exists():
            raise ValueError("No prediction events have been recorded")
        found = any(
            json.loads(line).get("prediction_id") == prediction_id
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if not found:
            raise ValueError("prediction_id does not exist")

        feedback_path = _event_path(config, "feedback_log_file")
        if feedback_path.exists():
            already_labeled = any(
                json.loads(line).get("prediction_id") == prediction_id
                for line in feedback_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if already_labeled:
                raise ValueError("prediction_id already has a recorded outcome")

        _append_jsonl(feedback_path, {
            "prediction_id": prediction_id,
            "actual_default": int(actual_default),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
