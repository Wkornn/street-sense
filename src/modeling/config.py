"""Load a training config and resolve references inside it."""

from pathlib import Path

import yaml

FEATURE_SETS_PATH = Path("configs/features.yaml")


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path):
    """Read a training YAML and replace the feature-set name with its column list."""
    cfg = load_yaml(path)
    if isinstance(cfg["features"], str):
        feature_sets = load_yaml(FEATURE_SETS_PATH)
        name = cfg["features"]
        if name not in feature_sets:
            raise KeyError(f"Feature set '{name}' not found in {FEATURE_SETS_PATH}")
        cfg["feature_set"] = name
        cfg["features"] = feature_sets[name]
    return cfg
