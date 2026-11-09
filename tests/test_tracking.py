"""MLflow tracking: a training run is recorded with its settings, metrics and model files."""

import pytest

from src.modeling.tracking import flatten
from src.modeling.train import run

mlflow = pytest.importorskip("mlflow")


def test_flatten_nested_config():
    assert flatten({"model": {"type": "x", "params": {"max_depth": 6}}, "name": "r"}) == {
        "model.type": "x", "model.params.max_depth": 6, "name": "r",
    }


def test_run_is_recorded(toy_dataset, tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.chdir(tmp_path)              # MLflow writes run files to ./mlruns
    monkeypatch.setenv("STREET_SENSE_TRACKING", "1")
    monkeypatch.setenv("STREET_SENSE_TRACKING_URI", uri)
    data_path = tmp_path / "dataset.parquet"
    toy_dataset.to_parquet(data_path)
    cfg = {
        "name": "tracking_test",
        "data_path": str(data_path),
        "target": {"type": "binary", "columns": ["acc_fatal", "acc_serious"]},
        "features": ["highway_rank", "length_m", "speed_mean"],
        "model": {"type": "xgb_classifier", "params": {"n_estimators": 10, "max_depth": 2},
                  "calibration": "isotonic"},
        "split": {"method": "random", "cal_size": 0.2, "test_size": 0.2, "seed": 0},
        "output": {"model_dir": str(tmp_path / "model")},
    }
    metrics = run(cfg)

    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    runs = client.search_runs([client.get_experiment_by_name("street-sense").experiment_id])
    assert len(runs) == 1
    logged = runs[0]
    assert logged.info.run_name == "tracking_test"
    assert logged.data.params["model.params.max_depth"] == "2"
    assert logged.data.params["features.count"] == "3"
    assert logged.data.metrics["roc_auc"] == pytest.approx(metrics["roc_auc"])
    assert logged.data.metrics["split.test.segments"] > 0
    assert logged.data.tags["model_type"] == "xgb_classifier"
    files = {f.path for f in client.list_artifacts(logged.info.run_id, "model")}
    assert {"model/meta.json", "model/booster.json", "model/risk_scores.parquet"} <= files
