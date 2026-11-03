"""Shared data preparation (src/modeling/data.py), used by both training and the app."""

import numpy as np
import pandas as pd
import pytest

from src.modeling import data


def test_select_features_fills_missing_with_zero():
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 2.0]})
    X = data.select_features(df, ["a", "b"])
    assert X.isna().sum().sum() == 0
    assert X.loc[1, "a"] == 0 and X.loc[0, "b"] == 0


def test_select_features_raises_on_missing_column():
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(KeyError, match="speed_mean"):
        data.select_features(df, ["a", "speed_mean"])


def test_binary_and_count_targets(toy_dataset):
    cols = {"columns": ["acc_fatal", "acc_serious"]}
    binary = data.make_target(toy_dataset, {**cols, "type": "binary"})
    count = data.make_target(toy_dataset, {**cols, "type": "count"})
    assert set(binary.unique()) <= {0, 1}
    assert (count == toy_dataset["acc_fatal"] + toy_dataset["acc_serious"]).all()
    assert ((count > 0) == (binary == 1)).all()


def test_split_sizes_and_rare_positives_in_every_part(toy_dataset):
    y = data.make_target(toy_dataset, {"type": "count", "columns": ["acc_fatal", "acc_serious"]})
    parts = data.split(toy_dataset, y, {"method": "random", "cal_size": 0.2, "test_size": 0.2, "seed": 0})
    X_train, X_cal, X_test, y_train, y_cal, y_test = parts
    assert len(X_train) + len(X_cal) + len(X_test) == len(toy_dataset)
    assert len(X_test) == pytest.approx(0.2 * len(toy_dataset), abs=2)
    for part in (y_train, y_cal, y_test):
        assert (part > 0).any()  # stratified on "any event"


def test_exposure_is_probe_volume_times_length():
    df = pd.DataFrame({"probe_count": [10.0, np.nan], "length_m": [50.0, 80.0]})
    out = data.add_exposure(df)
    assert out["exposure"].tolist() == [500.0, 0.0]
    assert out["log_exposure"].iloc[0] == pytest.approx(np.log1p(500))


def test_app_explanation_uses_training_feature_prep(toy_dataset, toy_features):
    """Regression test for the train/serve skew bug: the app's score must match training's."""
    from src.modeling.models import get_plugin
    from src.modeling.xai_explainer import explain_segment

    df = toy_dataset.copy()
    df.loc[5, "speed_mean"] = np.nan                     # a segment with missing traffic data
    X = data.select_features(df, toy_features)
    y = data.make_target(df, {"type": "binary", "columns": ["acc_fatal", "acc_serious"]})
    model = get_plugin("xgb_classifier").fit(
        {"params": {"n_estimators": 20, "max_depth": 3}, "calibration": None}, X, y, X, y
    )
    training_score = model.predict_risk(X.iloc[[5]])[0]
    app_score, _ = explain_segment(5, df, model, toy_features, top_k=3)
    assert app_score == pytest.approx(training_score)
