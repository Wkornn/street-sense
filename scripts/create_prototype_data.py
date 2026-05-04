"""
Create small local prototype artifacts for the Streamlit dashboard.

This does not replace the real pipeline. It only creates enough synthetic
Bangkok-shaped data for local UI/model prototyping when data/ and models/ are
not available after cloning the repository.

Run from project root:
    python scripts/create_prototype_data.py
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString


FEATURES_V2 = [
    "highway_rank",
    "lanes",
    "length_m",
    "lanes_known",
    "probe_count",
    "log_probe_count",
    "speed_mean",
    "has_probe_data",
    "congestion_score",
    "speed_drop_morning",
    "log_dist_intersection_m",
    "log_poi_count_200m",
    "log_building_density_200m",
    "log_dist_school_m",
    "log_dist_hospital_m",
    "log_dist_fuel_m",
    "log_dist_mall_m",
]

PROTOTYPE_SEGMENTS = [103470, 88421, 45112, 77005, 25018]


def make_model_dataset(n_rows=240, seed=42):
    rng = np.random.default_rng(seed)
    generated_ids = list(range(200000, 200000 + n_rows - len(PROTOTYPE_SEGMENTS)))
    segment_ids = np.array(PROTOTYPE_SEGMENTS + generated_ids)

    highway_rank = rng.integers(1, 7, size=n_rows)
    lanes_known = rng.binomial(1, 0.78, size=n_rows)
    lanes = np.where(lanes_known == 1, rng.integers(1, 5, size=n_rows), 2)
    length_m = rng.uniform(45, 140, size=n_rows).round(2)
    probe_count = rng.poisson(lam=320, size=n_rows) + rng.integers(0, 90, size=n_rows)
    has_probe_data = (probe_count > 0).astype(int)
    speed_mean = np.clip(rng.normal(34, 9, size=n_rows), 8, 75).round(2)
    congestion_score = np.clip(rng.beta(2.2, 4.0, size=n_rows) * 100, 0, 100).round(2)
    speed_drop_morning = np.clip(rng.normal(5, 5, size=n_rows), -10, 24).round(2)

    dist_intersection_m = rng.gamma(shape=2.0, scale=75, size=n_rows)
    poi_count_200m = rng.poisson(lam=8, size=n_rows)
    building_density_200m = rng.poisson(lam=35, size=n_rows)
    dist_school_m = rng.gamma(shape=2.3, scale=320, size=n_rows)
    dist_hospital_m = rng.gamma(shape=2.5, scale=430, size=n_rows)
    dist_fuel_m = rng.gamma(shape=2.0, scale=380, size=n_rows)
    dist_mall_m = rng.gamma(shape=2.0, scale=500, size=n_rows)

    risk_signal = (
        highway_rank * 0.18
        + lanes * 0.06
        + congestion_score * 0.035
        + speed_drop_morning * 0.055
        + np.log1p(poi_count_200m) * 0.35
        + np.log1p(building_density_200m) * 0.16
        - np.log1p(dist_intersection_m) * 0.17
        - speed_mean * 0.018
        + rng.normal(0, 0.35, size=n_rows)
    )
    is_risky = (risk_signal > np.quantile(risk_signal, 0.62)).astype(int)

    df = pd.DataFrame(
        {
            "segment_id": segment_ids,
            "highway_rank": highway_rank,
            "lanes": lanes,
            "length_m": length_m,
            "lanes_known": lanes_known,
            "probe_count": probe_count,
            "log_probe_count": np.log1p(probe_count),
            "speed_mean": speed_mean,
            "has_probe_data": has_probe_data,
            "congestion_score": congestion_score,
            "speed_drop_morning": speed_drop_morning,
            "log_dist_intersection_m": np.log1p(dist_intersection_m),
            "log_poi_count_200m": np.log1p(poi_count_200m),
            "log_building_density_200m": np.log1p(building_density_200m),
            "log_dist_school_m": np.log1p(dist_school_m),
            "log_dist_hospital_m": np.log1p(dist_hospital_m),
            "log_dist_fuel_m": np.log1p(dist_fuel_m),
            "log_dist_mall_m": np.log1p(dist_mall_m),
            "is_risky": is_risky,
            "risk_level": np.where(is_risky == 1, 1, 0),
        }
    )

    # Make the default demo segment visibly risky and stable for XAI testing.
    df.loc[df["segment_id"] == 103470, FEATURES_V2] = [
        5,
        3,
        98.2,
        1,
        780,
        np.log1p(780),
        22.4,
        1,
        72.5,
        14.7,
        np.log1p(42),
        np.log1p(23),
        np.log1p(88),
        np.log1p(260),
        np.log1p(410),
        np.log1p(190),
        np.log1p(550),
    ]
    df.loc[df["segment_id"] == 103470, ["is_risky", "risk_level"]] = [1, 2]
    return df


def make_road_segments(segment_ids, seed=42):
    rng = np.random.default_rng(seed)
    center_lon, center_lat = 100.5018, 13.7563
    records = []

    fixed_paths = {
        103470: [(100.4982, 13.7528), (100.5058, 13.7562), (100.5129, 13.7589)],
        88421: [(100.5238, 13.7444), (100.5298, 13.7481), (100.5368, 13.7517)],
        45112: [(100.4867, 13.7655), (100.4937, 13.7678), (100.5011, 13.7702)],
        77005: [(100.5451, 13.7231), (100.5508, 13.7295), (100.5562, 13.7348)],
        25018: [(100.4699, 13.7356), (100.4787, 13.7395), (100.4874, 13.7427)],
    }

    for sid in segment_ids:
        if sid in fixed_paths:
            coords = fixed_paths[sid]
        else:
            lon = center_lon + rng.normal(0, 0.035)
            lat = center_lat + rng.normal(0, 0.025)
            dx = rng.normal(0.004, 0.002)
            dy = rng.normal(0.002, 0.0015)
            coords = [(lon - dx, lat - dy), (lon, lat), (lon + dx, lat + dy)]
        records.append({"segment_id": int(sid), "geometry": LineString(coords)})

    return gpd.GeoDataFrame(records, crs="EPSG:4326")


def main():
    features_dir = Path("data/processed/features")
    processed_dir = Path("data/processed")
    features_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = make_model_dataset()
    dataset_path = features_dir / "model_dataset.parquet"
    df.to_parquet(dataset_path, index=False)

    segments = make_road_segments(df["segment_id"].tolist())
    segments_path = processed_dir / "road_segments.gpkg"
    segments.to_file(segments_path, driver="GPKG")

    print(f"Prototype dataset saved -> {dataset_path} ({len(df)} rows)")
    print(f"Prototype road segments saved -> {segments_path} ({len(segments)} rows)")


if __name__ == "__main__":
    main()
