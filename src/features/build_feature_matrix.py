"""
Feature Engineering — Part 5: Merge All Features
src/features/build_feature_matrix.py

Joins all feature tables into one master matrix:
  feat_road.parquet      (static road attributes)
  feat_accidents.parquet (accident counts)
  feat_spatial.parquet   (intersection dist, POI, buildings)
  feat_probe.parquet     (speed statistics)

Output: data/processed/features/feature_matrix.parquet
"""

import os
import yaml
import pandas as pd

CONFIG_PATH = "configs/data_sources.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["features"]["output_dir"]


def main():
    output_dir = load_config()

    tables = {
        "road":     "feat_road.parquet",
        "accident": "feat_accidents.parquet",
        "spatial":  "feat_spatial.parquet",
        "probe":    "feat_probe.parquet",
    }

    print("Loading feature tables...")
    dfs = {}
    for name, fname in tables.items():
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping")
            continue
        dfs[name] = pd.read_parquet(path)
        print(f"  {name:10s}: {len(dfs[name]):,} rows, {len(dfs[name].columns)} cols")

    if "road" not in dfs:
        raise FileNotFoundError("feat_road.parquet is required as the base table.")

    matrix = dfs["road"]
    for name, df in dfs.items():
        if name == "road":
            continue
        matrix = matrix.merge(df, on="segment_id", how="left")

    out = os.path.join(output_dir, "feature_matrix.parquet")
    matrix.to_parquet(out, index=False)

    print(f"\nFeature matrix: {len(matrix):,} rows × {len(matrix.columns)} columns")
    print("Columns:", matrix.columns.tolist())
    print(f"\nSaved → {out}")

    # Quick null report
    null_pct = (matrix.isnull().sum() / len(matrix) * 100).round(1)
    null_pct = null_pct[null_pct > 0]
    if len(null_pct):
        print("\nNull % per column:")
        print(null_pct.to_string())
    else:
        print("\nNo nulls in feature matrix.")


if __name__ == "__main__":
    main()
