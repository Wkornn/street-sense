"""
Feature Engineering — Part 1: Accident Features
src/features/feat_accidents.py

For each road segment, computes:
  Spatial:
    - acc_total          : total accident count
    - acc_fatal          : severity == 2
    - acc_serious        : severity == 1
    - acc_minor          : severity == 0
  Temporal:
    - acc_morning_peak   : time_bin == 'morning_peak'
    - acc_evening_peak   : time_bin == 'evening_peak'
    - acc_daytime        : time_bin == 'daytime'
    - acc_late_night     : time_bin == 'late_night'
    - acc_monsoon        : month in monsoon_months
    - acc_dry            : month not in monsoon_months

Output: data/processed/features/feat_accidents.parquet
"""

import os
import yaml
import pandas as pd
import geopandas as gpd

CONFIG_PATH = "configs/data_sources.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {
        "accidents":      cfg["road_segments"]["snapped_accidents_output"],
        "segments":       cfg["road_segments"]["output"],
        "output_dir":     cfg["features"]["output_dir"],
        "monsoon_months": set(cfg["features"]["monsoon_months"]),
    }


def main():
    cfg = load_config()
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print("Loading snapped accidents...")
    acc = gpd.read_file(cfg["accidents"])
    acc = acc[acc["segment_id"] != -1].copy()
    print(f"  {len(acc):,} matched accidents")

    print("Loading segments (for full index)...")
    segs = gpd.read_file(cfg["segments"])[["segment_id"]]

    # --- Severity counts ---
    sev = acc.groupby("segment_id")["severity"].value_counts().unstack(fill_value=0)
    sev.columns = [f"acc_sev_{int(c)}" for c in sev.columns]
    sev["acc_total"] = sev.sum(axis=1)
    sev = sev.rename(columns={"acc_sev_0": "acc_minor",
                               "acc_sev_1": "acc_serious",
                               "acc_sev_2": "acc_fatal"})
    for col in ["acc_minor", "acc_serious", "acc_fatal"]:
        if col not in sev.columns:
            sev[col] = 0

    # --- Time-bin counts ---
    time_bins = ["morning_peak", "evening_peak", "daytime", "late_night"]
    tbin = acc.groupby("segment_id")["time_bin"].value_counts().unstack(fill_value=0)
    for tb in time_bins:
        col = f"acc_{tb}"
        if tb in tbin.columns:
            tbin = tbin.rename(columns={tb: col})
        else:
            tbin[col] = 0
    tbin = tbin[[f"acc_{tb}" for tb in time_bins]]

    # --- Monsoon vs dry ---
    acc["is_monsoon"] = acc["month"].isin(cfg["monsoon_months"])
    season = acc.groupby("segment_id")["is_monsoon"].agg(
        acc_monsoon="sum",
        acc_dry=lambda x: (~x).sum()
    )

    # --- Merge all onto full segment index ---
    feat = segs.set_index("segment_id")
    feat = feat.join(sev, how="left")
    feat = feat.join(tbin, how="left")
    feat = feat.join(season, how="left")
    feat = feat.fillna(0).astype(int)
    feat = feat.reset_index()

    out = os.path.join(cfg["output_dir"], "feat_accidents.parquet")
    feat.to_parquet(out, index=False)
    print(f"  Saved {len(feat):,} rows → {out}")
    print(feat.describe().T[["mean", "max"]].to_string())


if __name__ == "__main__":
    main()
