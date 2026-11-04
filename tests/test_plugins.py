"""Every registered model plugin: train through train.py, save, reload, predict, explain.

A new plugin must get an entry in PLUGIN_RUNS, otherwise test_every_plugin_is_tested fails.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modeling import data
from src.modeling.models import MODELS, load_model
from src.modeling.train import run

TARGET = {"columns": ["acc_fatal", "acc_serious"]}

# Small, fast settings per plugin: (target type, model section, features)
PLUGIN_RUNS = {
    "xgb_classifier": ("binary", {"params": {"n_estimators": 30, "max_depth": 3}, "calibration": "isotonic"},
                       ["highway_rank", "length_m", "speed_mean", "congestion_score", "lanes"]),
    "neg_binomial": ("count", {"params": {"alpha": "auto", "offset_col": "length_m", "offset_unit": 100}},
                     ["highway_rank", "length_m", "speed_mean", "lanes"]),
    "fcm_xgb": ("binary", {"params": {"n_grades": 3, "surrogate_cols": ["speed_mean", "congestion_score"],
                                      "invert_cols": ["speed_mean"], "n_estimators": 20, "max_depth": 3,
                                      "random_state": 0}},
                ["highway_rank", "length_m", "speed_mean", "congestion_score", "lanes"]),
}


def test_every_plugin_is_tested():
    assert set(MODELS) == set(PLUGIN_RUNS), "add new plugins to PLUGIN_RUNS"


@pytest.fixture(params=sorted(PLUGIN_RUNS))
def trained(request, toy_dataset, tmp_path):
    """Train one plugin end-to-end with train.run() on the toy data."""
    name = request.param
    target_type, model_section, features = PLUGIN_RUNS[name]
    data_path = tmp_path / "dataset.parquet"
    toy_dataset.to_parquet(data_path)
    cfg = {
        "name": f"test_{name}",
        "data_path": str(data_path),
        "target": {**TARGET, "type": target_type},
        "features": features,
        "model": {"type": name, **model_section},
        "split": {"method": "random", "cal_size": 0.2, "test_size": 0.2, "seed": 0},
        "output": {"model_dir": str(tmp_path / "model")},
    }
    metrics = run(cfg)
    return name, cfg, metrics, toy_dataset


def test_run_writes_model_folder(trained):
    _, cfg, metrics, _ = trained
    out = cfg["output"]["model_dir"]
    meta = json.loads(Path(out, "meta.json").read_text())
    assert meta["model_type"] == cfg["model"]["type"]
    assert meta["features"] == cfg["features"]
    assert set(metrics) >= {"roc_auc", "pr_auc", "brier", "top_decile_lift"}
    if cfg["target"]["type"] == "count":
        assert {"mae", "rmse", "poisson_deviance"} <= set(metrics)


def test_reload_gives_identical_scores(trained):
    _, cfg, _, df = trained
    out = cfg["output"]["model_dir"]
    model = load_model(out)
    X = data.select_features(data.add_exposure(df), cfg["features"])
    saved = pd.read_parquet(f"{out}/risk_scores.parquet")
    np.testing.assert_array_equal(model.predict_risk(X), saved["risk_score"].to_numpy())


def test_scores_are_between_0_and_1(trained):
    _, cfg, _, df = trained
    model = load_model(cfg["output"]["model_dir"])
    risk = model.predict_risk(data.select_features(df, cfg["features"]))
    assert np.all((risk >= 0) & (risk <= 1))


def test_explain_gives_one_value_per_feature(trained):
    _, cfg, _, df = trained
    model = load_model(cfg["output"]["model_dir"])
    X = data.select_features(df, cfg["features"]).iloc[:7]
    contributions = model.explain(X)
    assert contributions.shape == (7, len(cfg["features"]))
    assert np.isfinite(contributions).all()
