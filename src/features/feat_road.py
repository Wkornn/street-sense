"""
Feature Engineering — Part 3: Road Network Attributes
src/features/feat_road.py

Extracts static road attributes from the segment layer:
  - highway_type   : OSM highway tag (encoded as category)
  - highway_rank   : numeric rank (motorway=6 … residential=1)
  - lanes          : number of lanes (NaN → 0)
  - length_m       : segment length in metres

Output: data/processed/features/feat_road.parquet
"""

import os
import yaml
import pandas as pd
import geopandas as gpd

CONFIG_PATH = "configs/data_sources.yaml"

HIGHWAY_RANK = {
    "motorway":       6,
    "motorway_link":  5,
    "trunk":          5,
    "trunk_link":     4,
    "primary":        4,
    "primary_link":   3,
    "secondary":      3,
    "secondary_link": 2,
    "tertiary":       2,
    "tertiary_link":  1,
    "unclassified":   1,
    "residential":    1,
}


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {
        "segments":   cfg["road_segments"]["output"],
        "output_dir": cfg["features"]["output_dir"],
    }


def main():
    cfg = load_config()
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print("Loading segments...")
    segs = gpd.read_file(cfg["segments"])[
        ["segment_id", "highway", "lanes", "length_m"]
    ]

    segs["highway_rank"] = segs["highway"].map(HIGHWAY_RANK).fillna(1).astype(int)
    segs["lanes"] = pd.to_numeric(segs["lanes"], errors="coerce").fillna(0).astype(int)
    segs["length_m"] = segs["length_m"].round(2)

    feat = segs[["segment_id", "highway", "highway_rank", "lanes", "length_m"]]

    out = os.path.join(cfg["output_dir"], "feat_road.parquet")
    feat.to_parquet(out, index=False)
    print(f"  Saved {len(feat):,} rows → {out}")
    print(feat["highway"].value_counts().to_string())


if __name__ == "__main__":
    main()
