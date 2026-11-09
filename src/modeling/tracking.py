"""Log each training run to MLflow: settings, metrics, split balance and the model folder.

Tracking is optional: training works the same without it. Turn it off with
`enabled: false` in configs/tracking.yaml, or override it with the environment variable
STREET_SENSE_TRACKING=0 / 1 (tests and CI use 0). STREET_SENSE_TRACKING_URI overrides the
database. With a sqlite database, MLflow stores run files in ./mlruns (git-ignored).
"""

import os
from pathlib import Path

import yaml

SETTINGS_PATH = Path("configs/tracking.yaml")


def load_settings():
    settings = {}
    if SETTINGS_PATH.exists():
        settings = yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8")) or {}
    switch = os.environ.get("STREET_SENSE_TRACKING")
    if switch in ("0", "1"):
        settings["enabled"] = switch == "1"
    settings["uri"] = os.environ.get("STREET_SENSE_TRACKING_URI", settings.get("uri"))
    return settings


def flatten(d, prefix=""):
    """{"model": {"params": {"max_depth": 6}}} -> {"model.params.max_depth": 6}"""
    flat = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def log_run(cfg, metrics, split_summary, out_dir):
    """Record one run. Returns the MLflow run ID, or None if tracking is off."""
    settings = load_settings()
    if not settings.get("enabled"):
        return None
    try:
        import mlflow
    except ImportError:
        print("Tracking skipped: mlflow is not installed (uv sync installs it)")
        return None

    mlflow.set_tracking_uri(settings["uri"])
    mlflow.set_experiment(settings.get("experiment", "street-sense"))

    # The feature list is long: log its name and size (the full list is in config.yaml)
    params = {k: v for k, v in cfg.items() if k not in ("features", "output")}
    params["features.count"] = len(cfg["features"])
    params["features.set"] = cfg.get("feature_set", "inline")

    with mlflow.start_run(run_name=cfg["name"]) as run:
        mlflow.log_params(flatten(params))
        mlflow.log_metrics(metrics)
        for part, row in split_summary.items():
            mlflow.log_metrics({f"split.{part}.{k}": v for k, v in row.items()})
        mlflow.set_tags({
            "model_type": cfg["model"]["type"],
            "split_method": cfg["split"].get("method", "random"),
            "target_type": cfg["target"].get("type", "binary"),
        })
        mlflow.log_artifacts(str(out_dir), artifact_path="model")
        return run.info.run_id
