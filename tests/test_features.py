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


# --- Spatial split -------------------------------------------------------------------

SPATIAL = {"method": "spatial", "cal_size": 0.2, "test_size": 0.2, "seed": 0}


@pytest.fixture
def toy_cells(toy_dataset):
    """50 fake grid cells of 40 segments each."""
    return pd.Series(toy_dataset["segment_id"] // 40, index=toy_dataset.index).astype(str)


def test_spatial_split_never_shares_a_cell(toy_dataset, toy_cells):
    y = data.make_target(toy_dataset, {"type": "binary", "columns": ["acc_fatal", "acc_serious"]})
    X_train, X_cal, X_test, *_ = data.split(toy_dataset, y, SPATIAL, toy_cells)
    cells = [set(toy_cells.loc[part.index]) for part in (X_train, X_cal, X_test)]
    assert not (cells[0] & cells[1]) and not (cells[0] & cells[2]) and not (cells[1] & cells[2])
    assert len(X_train) + len(X_cal) + len(X_test) == len(toy_dataset)


def test_spatial_split_is_about_60_20_20_and_reproducible(toy_dataset, toy_cells):
    y = data.make_target(toy_dataset, {"type": "binary", "columns": ["acc_fatal", "acc_serious"]})
    first = data.split(toy_dataset, y, SPATIAL, toy_cells)
    again = data.split(toy_dataset, y, SPATIAL, toy_cells)
    assert len(first[2]) == pytest.approx(0.2 * len(toy_dataset), rel=0.25)
    assert all(a.index.equals(b.index) for a, b in zip(first[:3], again[:3], strict=True))


def test_spatial_split_needs_cells_and_equal_cal_test(toy_dataset, toy_cells):
    y = data.make_target(toy_dataset, {"type": "binary", "columns": ["acc_fatal", "acc_serious"]})
    with pytest.raises(ValueError, match="groups"):
        data.split(toy_dataset, y, SPATIAL)
    with pytest.raises(ValueError, match="cal_size == test_size"):
        data.split(toy_dataset, y, {**SPATIAL, "cal_size": 0.1}, toy_cells)


def test_spatial_groups_puts_nearby_segments_in_the_same_cell(tmp_path):
    import geopandas as gpd
    from shapely.geometry import LineString

    segments = gpd.GeoDataFrame({
        "segment_id": [1, 2, 3],
        "geometry": [LineString([(100, 100), (150, 100)]),     # cell 0_0
                     LineString([(900, 900), (950, 900)]),     # cell 0_0 (same 2 km square)
                     LineString([(2500, 100), (2550, 100)])],  # cell 1_0
    }, crs="EPSG:32647")
    path = tmp_path / "segments.gpkg"
    segments.to_file(path)
    cells = data.spatial_groups([3, 1, 2], path, cell_size_m=2000)
    assert cells.tolist() == ["1_0", "0_0", "0_0"]
    with pytest.raises(KeyError, match="not found"):
        data.spatial_groups([1, 99], path, cell_size_m=2000)


def test_describe_split_counts_events_and_cells(toy_dataset, toy_cells):
    y = data.make_target(toy_dataset, {"type": "binary", "columns": ["acc_fatal", "acc_serious"]})
    X_train, X_cal, X_test, y_train, y_cal, y_test = data.split(toy_dataset, y, SPATIAL, toy_cells)
    summary = data.describe_split(
        {"train": (X_train, y_train), "cal": (X_cal, y_cal), "test": (X_test, y_test)}, toy_cells
    )
    assert sum(row["segments"] for row in summary.values()) == len(toy_dataset)
    assert sum(row["events"] for row in summary.values()) == int(y.sum())
    assert sum(row["cells"] for row in summary.values()) == toy_cells.nunique()
