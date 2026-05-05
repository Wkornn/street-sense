"""
Feature Engineering — Part 2: Spatial Context Features
src/features/feat_spatial.py

For each road segment centroid, computes:
  - dist_intersection_m  : distance to nearest intersection (m)
  - poi_count_200m       : POI count within 200m buffer
  - dist_school_m        : distance to nearest school/university
  - dist_hospital_m      : distance to nearest hospital
  - dist_fuel_m          : distance to nearest fuel station
  - dist_mall_m          : distance to nearest mall/supermarket
  - building_density_200m: building count within 200m buffer

Output: data/processed/features/feat_spatial.parquet
"""

import os
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

CONFIG_PATH = "configs/data_sources.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seg = cfg["road_segments"]
    feat = cfg["features"]
    return {
        "segments":      seg["output"],
        "intersections": seg["intersections_output"],
        "poi":           "data/raw/osm/osm_poi.gpkg",
        "buildings":     "data/raw/osm/osm_buildings.gpkg",
        "output_dir":    feat["output_dir"],
        "buffer_m":      feat["buffer_m"],
        "crs":           seg["projected_crs"],
        "poi_types":     feat["poi_types"],
    }


def nearest_distance(origins_gdf, targets_gdf):
    """Return array of distances from each origin to its nearest target."""
    tree = targets_gdf.sindex
    dists = np.full(len(origins_gdf), np.nan)
    for i, geom in enumerate(origins_gdf.geometry):
        candidates = list(tree.query(geom.buffer(5000)))  # 5km search radius
        if not candidates:
            candidates = list(range(len(targets_gdf)))    # fallback: all
        sub = targets_gdf.iloc[candidates]
        dists[i] = sub.geometry.distance(geom).min()
    return dists


def count_within_buffer(origins_gdf, targets_gdf, buffer_m):
    """Return array of target counts within buffer_m of each origin."""
    tree = targets_gdf.sindex
    counts = np.zeros(len(origins_gdf), dtype=int)
    for i, geom in enumerate(origins_gdf.geometry):
        buf = geom.buffer(buffer_m)
        candidates = list(tree.query(buf))
        if candidates:
            sub = targets_gdf.iloc[candidates]
            counts[i] = sub.geometry.intersects(buf).sum()
    return counts


def main():
    cfg = load_config()
    os.makedirs(cfg["output_dir"], exist_ok=True)
    crs = cfg["crs"]
    buf = cfg["buffer_m"]

    print("Loading segments...")
    segs = gpd.read_file(cfg["segments"])
    # Use segment midpoint as representative point
    centroids = segs.copy()
    centroids["geometry"] = segs.geometry.interpolate(0.5, normalized=True)
    centroids = centroids[["segment_id", "geometry"]].set_crs(crs)

    # --- Intersection distance ---
    print("Computing distance to nearest intersection...")
    ints = gpd.read_file(cfg["intersections"])
    centroids["dist_intersection_m"] = nearest_distance(centroids, ints)

    # --- POI ---
    print("Loading POI...")
    poi_raw = gpd.read_file(cfg["poi"]).to_crs(crs)
    poi_raw["geometry"] = poi_raw.geometry.centroid

    # All POI count within buffer
    print(f"  POI count within {buf}m...")
    centroids[f"poi_count_{buf}m"] = count_within_buffer(centroids, poi_raw, buf)

    # Distance to specific POI types
    poi_targets = {
        "school":   poi_raw[poi_raw["amenity"].isin(["school", "university"])],
        "hospital": poi_raw[poi_raw["amenity"] == "hospital"],
        "fuel":     poi_raw[poi_raw["amenity"] == "fuel"],
        "mall":     poi_raw[poi_raw["shop"].isin(["mall", "supermarket"])],
    }
    for name, subset in poi_targets.items():
        col = f"dist_{name}_m"
        if len(subset) == 0:
            centroids[col] = np.nan
        else:
            print(f"  Distance to {name} ({len(subset)} POIs)...")
            centroids[col] = nearest_distance(centroids, subset)

    # --- Building density ---
    print(f"Computing building density within {buf}m...")
    bldg = gpd.read_file(cfg["buildings"]).to_crs(crs)
    bldg["geometry"] = bldg.geometry.centroid
    centroids[f"building_density_{buf}m"] = count_within_buffer(centroids, bldg, buf)

    # --- Save ---
    feat_cols = ["segment_id", "dist_intersection_m", f"poi_count_{buf}m",
                 "dist_school_m", "dist_hospital_m", "dist_fuel_m",
                 "dist_mall_m", f"building_density_{buf}m"]
    feat = centroids[feat_cols].copy()
    # Round distances to 1 decimal
    for col in feat_cols[1:]:
        if "dist" in col:
            feat[col] = feat[col].round(1)

    out = os.path.join(cfg["output_dir"], "feat_spatial.parquet")
    feat.to_parquet(out, index=False)
    print(f"  Saved {len(feat):,} rows → {out}")
    print(feat.describe().T[["mean", "max"]].to_string())


if __name__ == "__main__":
    main()
