"""
Snap accident points to the nearest road segment.

For each accident, finds the nearest segment within max_distance_m
and assigns its segment_id. Accidents beyond the threshold are flagged
with segment_id = -1 and snap_dist_m = NaN.

Config: configs/data_sources.yaml
"""

import os
import yaml
import numpy as np
import geopandas as gpd
from shapely.ops import nearest_points

CONFIG_PATH = "configs/data_sources.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seg = cfg["road_segments"]
    acc = cfg["accidents"]
    return {
        "segments_path":  seg["output"],
        "accidents_path": acc["clean_parquet"],
        "output_path":    "data/processed/accidents_snapped.parquet",
        "boundary":       cfg["boundary"]["bangkok"],
        "max_dist_m":     seg["snap_max_distance_m"],
        "crs":            seg["projected_crs"],
    }


def snap_accidents(accidents, segments, max_dist_m):
    """
    Vectorised nearest-segment lookup using STRtree.
    Returns accidents GeoDataFrame with added columns:
      segment_id, snap_dist_m, snapped_geometry
    """
    tree = segments.sindex

    seg_ids   = np.full(len(accidents), -1, dtype=int)
    distances = np.full(len(accidents), np.nan)

    for i, acc_geom in enumerate(accidents.geometry):
        # candidates within max_dist_m
        candidates = list(tree.query(acc_geom.buffer(max_dist_m)))
        if not candidates:
            continue

        cand_segs = segments.iloc[candidates]
        dists     = cand_segs.geometry.distance(acc_geom)
        nearest   = dists.idxmin()
        dist      = dists[nearest]

        if dist <= max_dist_m:
            seg_ids[i]   = segments.loc[nearest, "segment_id"]
            distances[i] = round(dist, 2)

    accidents = accidents.copy()
    accidents["segment_id"]  = seg_ids
    accidents["snap_dist_m"] = distances
    return accidents


def main():
    cfg = load_config()

    print("Loading segments...")
    segments = gpd.read_file(cfg["segments_path"])
    print(f"  {len(segments):,} segments")

    print("Loading Bangkok boundary...")
    boundary = gpd.read_file(cfg["boundary"])
    bkk_geom = boundary.to_crs(cfg["crs"]).union_all()

    print("Loading accidents...")
    accidents = gpd.read_parquet(cfg["accidents_path"])
    accidents = accidents.to_crs(cfg["crs"])
    # Filter to accidents actually inside Bangkok polygon
    accidents = accidents[accidents.geometry.within(bkk_geom)].copy()
    print(f"  {len(accidents):,} accidents inside Bangkok boundary")

    print(f"Snapping (max {cfg['max_dist_m']}m)...")
    snapped = snap_accidents(accidents, segments, cfg["max_dist_m"])

    matched   = (snapped["segment_id"] != -1).sum()
    unmatched = (snapped["segment_id"] == -1).sum()
    print(f"  Matched  : {matched:,} ({matched/len(snapped)*100:.1f}%)")
    print(f"  Unmatched: {unmatched:,} ({unmatched/len(snapped)*100:.1f}%)")
    print(f"  Avg snap distance: {snapped['snap_dist_m'].mean():.1f}m")

    os.makedirs(os.path.dirname(cfg["output_path"]) or ".", exist_ok=True)
    snapped.to_parquet(cfg["output_path"], index=False)
    print(f"  Saved → {cfg['output_path']}")


if __name__ == "__main__":
    main()
