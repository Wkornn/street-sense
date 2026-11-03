"""Accident snapping: each accident goes to the nearest segment within max distance, or -1."""

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point

from src.geospatial.spatial.snap_accidents import snap_accidents

CRS = "EPSG:32647"


def make_segments():
    return gpd.GeoDataFrame({
        "segment_id": [10, 11],
        "geometry": [LineString([(0, 0), (100, 0)]), LineString([(0, 200), (100, 200)])],
    }, crs=CRS)


def make_accidents(points):
    return gpd.GeoDataFrame(geometry=[Point(p) for p in points], crs=CRS)


def test_snaps_to_nearest_segment_within_radius():
    acc = make_accidents([(50, 5), (50, 190)])
    out = snap_accidents(acc, make_segments(), max_dist_m=50)
    assert out["segment_id"].tolist() == [10, 11]
    assert out["snap_dist_m"].tolist() == [5.0, 10.0]


def test_far_accident_is_unmatched():
    acc = make_accidents([(50, 100)])  # 100 m from both roads
    out = snap_accidents(acc, make_segments(), max_dist_m=50)
    assert out["segment_id"].tolist() == [-1]
    assert np.isnan(out["snap_dist_m"].iloc[0])


def test_snap_distance_never_exceeds_radius():
    rng = np.random.default_rng(1)
    acc = make_accidents(rng.uniform(-50, 250, size=(300, 2)))
    out = snap_accidents(acc, make_segments(), max_dist_m=50)
    matched = out[out["segment_id"] != -1]
    assert (matched["snap_dist_m"] <= 50).all()
