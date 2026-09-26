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

    config_file = Path(config_path)
    try:
        with config_file.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration in {config_file}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {config_file}")
    return config
