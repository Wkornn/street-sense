"""Checks on the REAL pipeline outputs. Skipped when the files don't exist (e.g. on CI).

These catch the inconsistencies found in October 2026: road segments, snapped accidents and
the model dataset built from different segmentations.

Run only these: pytest -m data
"""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

pytestmark = pytest.mark.data

CFG = yaml.safe_load(open("configs/data_sources.yaml", encoding="utf-8"))
SEGMENTS = Path(CFG["road_segments"]["output"])
SNAPPED = Path(CFG["road_segments"]["snapped_accidents_output"])
DATASET = Path(CFG["features"]["output_dir"]) / "model_dataset.parquet"


def need(*paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        pytest.skip(f"pipeline output not found: {missing}")


@pytest.fixture(scope="module")
def segments():
    need(SEGMENTS)
    import geopandas as gpd
    return gpd.read_file(SEGMENTS)


def test_segment_ids_are_unique(segments):
    assert segments["segment_id"].is_unique


def test_segments_are_at_most_segment_length(segments):
    limit = CFG["road_segments"]["segment_length_m"]
    assert (segments.length <= limit + 0.01).all()


def test_snapped_accidents_within_radius_and_on_known_segments(segments):
    need(SNAPPED)
    import geopandas as gpd
    acc = gpd.read_parquet(SNAPPED)
    matched = acc[acc["segment_id"] != -1]
    assert (matched["snap_dist_m"] <= CFG["road_segments"]["snap_max_distance_m"]).all()
    assert matched["segment_id"].isin(segments["segment_id"]).all()


def test_dataset_matches_segments(segments):
    """Same IDs AND same roads: lengths in the dataset must match the geometry."""
    need(DATASET)
    ds = pd.read_parquet(DATASET, columns=["segment_id", "length_m"])
    assert set(ds["segment_id"]) == set(segments["segment_id"])
    merged = ds.merge(segments.assign(geom_len=segments.length)[["segment_id", "geom_len"]], on="segment_id")
    assert (merged["length_m"] - merged["geom_len"]).abs().max() < 1.0


@pytest.mark.parametrize("model_dir", sorted(p for p in Path("models").glob("*") if (p / "meta.json").exists()))
def test_model_scores_cover_dataset(model_dir):
    need(DATASET)
    meta = json.load(open(model_dir / "meta.json"))
    scores = pd.read_parquet(model_dir / "risk_scores.parquet")
    ds = pd.read_parquet(DATASET, columns=["segment_id"])
    assert scores["segment_id"].is_unique
    assert set(scores["segment_id"]) == set(ds["segment_id"]), f"{meta['name']} scores are stale"
