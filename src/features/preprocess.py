"""
Preprocessing — Build Model-Ready Dataset
src/features/preprocess.py

Loads the merged feature_matrix.parquet and produces a clean
model-ready dataset with:
  - Speed/congestion imputation by road type (highway_rank)
  - Spatial distance NaN → 5000m (no POI found)
  - Derived features (acc_rate, congestion_score, log transforms)
  - Target variable: is_risky (binary) and risk_level (3-class)

Output: data/processed/features/model_dataset.parquet
"""

import os
import yaml
import numpy as np
import pandas as pd

CONFIG_PATH = "configs/data_sources.yaml"
MODEL_CONFIG_PATH = "configs/model_params.yaml"

TIME_BINS = ["morning_peak", "daytime", "evening_peak", "night", "late_night"]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def impute_by_road_type(df, cols):
    """Fill NaN in cols with the median value for the same highway_rank group."""
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = df.groupby("highway_rank")[col].transform(
            lambda x: x.fillna(x.median())
        )
        # Fallback: if the whole road-type group is NaN, fill with 0
        df[col] = df[col].fillna(0)
    return df


def main():
    data_cfg = load_yaml(CONFIG_PATH)
    model_cfg = load_yaml(MODEL_CONFIG_PATH)
    
    output_dir = data_cfg["features"]["output_dir"]
    prep_cfg = model_cfg.get("preprocessing", {})

    # Parameters from config
    sparse_bins = prep_cfg.get("sparse_bins_to_drop", ["night", "late_night"])
    default_dist = prep_cfg.get("default_poi_dist_m", 5000)
    w = prep_cfg.get("congestion_weights", {
        "morning_peak": 0.35, "daytime": 0.20, "evening_peak": 0.35, "night": 0.10
    })

    matrix_path = os.path.join(output_dir, "feature_matrix.parquet")
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"feature_matrix.parquet not found at {matrix_path}. "
                                f"Run build_feature_matrix.py first.")

    print("Loading feature matrix...")
    df = pd.read_parquet(matrix_path)
    print(f"  {len(df):,} segments, {len(df.columns)} columns")

    # ----------------------------------------------------------------
    # 1. Impute probe speed & congestion by road type (highway_rank)
    #    Speed NaN = no probe data for that bin → use road-type median
    # ----------------------------------------------------------------
    print("Imputing probe speed/congestion by highway_rank...")
    speed_cols = [f"speed_mean_{t}" for t in TIME_BINS if t not in sparse_bins]
    cong_cols  = [f"pct_below_20kmh_{t}" for t in TIME_BINS if t not in sparse_bins]
    all_probe_impute = speed_cols + cong_cols

    df = impute_by_road_type(df, all_probe_impute)

    # Drop sparse time-bin columns (too unreliable for model input)
    sparse_speed = [f"speed_mean_{t}" for t in sparse_bins]
    sparse_cong  = [f"pct_below_20kmh_{t}" for t in sparse_bins]
    df = df.drop(columns=sparse_speed + sparse_cong, errors="ignore")

    # ----------------------------------------------------------------
    # 2. Spatial: fill missing distances with default (no POI found)
    # ----------------------------------------------------------------
    print("Filling missing spatial distances...")
    dist_cols = ["dist_intersection_m", "dist_school_m",
                 "dist_hospital_m", "dist_fuel_m", "dist_mall_m"]
    df[dist_cols] = df[[c for c in dist_cols if c in df.columns]].fillna(default_dist)

    # ----------------------------------------------------------------
    # 3. Road: handle unknown lanes (0 = not recorded)
    # ----------------------------------------------------------------
    if "lanes" in df.columns:
        df["lanes_known"] = (df["lanes"] > 0).astype(int)
        lane_median = df.loc[df["lanes"] > 0, "lanes"].median()
        df["lanes"] = df["lanes"].replace(0, lane_median)

    # Drop raw highway string — highway_rank already encodes it numerically
    df = df.drop(columns=["highway"], errors="ignore")

    # ----------------------------------------------------------------
    # 4. Derived features
    # ----------------------------------------------------------------
    print("Adding derived features...")

    # Accident rate normalized by road length
    if "acc_total" in df.columns and "length_m" in df.columns:
        df["acc_rate_per_100m"] = (
            df["acc_total"] / (df["length_m"] / 100)
        ).fillna(0)
        # Cap at 99th percentile to reduce extreme outlier influence
        cap = df["acc_rate_per_100m"].quantile(0.99)
        df["acc_rate_per_100m"] = df["acc_rate_per_100m"].clip(upper=cap)

    # Congestion severity score (weighted avg of peak-hour congestion)
    has_morning = "pct_below_20kmh_morning_peak" in df.columns
    has_evening = "pct_below_20kmh_evening_peak" in df.columns
    has_daytime = "pct_below_20kmh_daytime" in df.columns
    if has_morning and has_evening and has_daytime:
        if "pct_below_20kmh_night" in df.columns:
            df["congestion_score"] = (
                df["pct_below_20kmh_morning_peak"] * w["morning_peak"] +
                df["pct_below_20kmh_daytime"]      * w["daytime"] +
                df["pct_below_20kmh_evening_peak"] * w["evening_peak"] +
                df["pct_below_20kmh_night"]        * w["night"]
            ).fillna(0)
        else:
            # Fallback if night bin was dropped
            df["congestion_score"] = (
                df["pct_below_20kmh_morning_peak"] * w["morning_peak"] +
                df["pct_below_20kmh_daytime"]      * w["daytime"] +
                df["pct_below_20kmh_evening_peak"] * w["evening_peak"]
            ).fillna(0) / (w["morning_peak"] + w["daytime"] + w["evening_peak"])

    # Speed drop between peak and base: higher = more congestion variance
    if "speed_mean_morning_peak" in df.columns and "speed_mean_daytime" in df.columns:
        df["speed_drop_morning"] = (
            df["speed_mean_daytime"] - df["speed_mean_morning_peak"]
        ).fillna(0)

    # Log transforms for heavily skewed count features and large distances
    for col in ["probe_count", "poi_count_200m", "building_density_200m", 
                "dist_intersection_m", "dist_school_m", "dist_hospital_m", 
                "dist_fuel_m", "dist_mall_m",
                "acc_total", "acc_rate_per_100m", "acc_morning_peak", 
                "acc_evening_peak", "acc_monsoon"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    # ----------------------------------------------------------------
    # 5. Define target variables
    # ----------------------------------------------------------------
    print("Defining target variables...")
    if "acc_total" in df.columns:
        # Binary: any accident vs none
        df["is_risky"] = (df["acc_total"] > 0).astype(int)

        # 3-class risk level
        df["risk_level"] = pd.cut(
            df["acc_total"],
            bins=[-1, 0, 2, float("inf")],
            labels=[0, 1, 2]   # 0=none, 1=low (1-2), 2=high (3+)
        ).astype(int)

    # ----------------------------------------------------------------
    # 6. Save
    # ----------------------------------------------------------------
    out_path = os.path.join(output_dir, "model_dataset.parquet")
    df.to_parquet(out_path, index=False)

    print(f"\nModel dataset: {len(df):,} rows × {len(df.columns)} columns")
    print(f"Saved → {out_path}")

    # Null report
    null_pct = (df.isnull().sum() / len(df) * 100).round(1)
    null_pct = null_pct[null_pct > 0]
    if len(null_pct):
        print("\nRemaining nulls (%):")
        print(null_pct.to_string())
    else:
        print("\nNo nulls — dataset is clean.")

    # Target distribution
    if "is_risky" in df.columns:
        print(f"\nTarget (is_risky): {df['is_risky'].value_counts().to_dict()}")
        pct_risky = df["is_risky"].mean() * 100
        print(f"  → {pct_risky:.1f}% of segments are risky (have ≥1 accident)")


if __name__ == "__main__":
    main()
