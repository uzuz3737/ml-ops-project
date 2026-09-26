import os
import yaml
from pathlib import Path
from typing import Any, Dict

def get_project_root() -> Path:
    """Return the root directory of the project."""
    return Path(__file__).resolve().parent.parent.parent

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        root = get_project_root()
        config_path = str(root / "config" / "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
