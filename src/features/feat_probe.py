import os
import glob
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
import pickle
from tqdm import tqdm

CONFIG_PATH = "configs/data_sources.yaml"
SNAP_RADIUS_M = 30      # metres
BATCH_SIZE    = 500_000  # rows per batch

# Match the bins in preprocess_accidents.py
def _get_time_bin(hour):
    if   6  <= hour < 9:  return "morning_peak"
    elif 9  <= hour < 16: return "daytime"
    elif 16 <= hour < 20: return "evening_peak"
    elif 20 <= hour < 24: return "night"
    else:                 return "late_night"

def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seg  = cfg["road_segments"]
    feat = cfg["features"]
    return {
        "segments":    seg["output"],
        "probe_dir":   "data/processed/probe_bangkok",
        "output_dir":  feat["output_dir"],
        "crs":         seg["projected_crs"],
        "speed_low":   feat["probe_speed_low_kmh"],
        "snap_radius": SNAP_RADIUS_M,
    }

def process_file(path, segments, cfg, accum):
    """
    Read one probe Parquet, snap to segments, update accumulators per (sid, bin).
    """
    # Load required columns
    df = pd.read_parquet(path, columns=["lat", "lon", "speed", "gps_valid", "timestamp"])
    
    # Pre-filter and parse time
    df = df[(df["gps_valid"] == 1) & (df["speed"] > 0) & (df["speed"] < 200)].copy()
    if df.empty: return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["time_bin"] = df["timestamp"].dt.hour.apply(_get_time_bin)

    for start in range(0, len(df), BATCH_SIZE):
        chunk = df.iloc[start : start + BATCH_SIZE].copy()
        
        gdf = gpd.GeoDataFrame(
            chunk, 
            geometry=gpd.points_from_xy(chunk["lon"], chunk["lat"]), 
            crs="EPSG:4326"
        ).to_crs(cfg["crs"])

        snapped = gpd.sjoin_nearest(gdf, segments, max_distance=cfg["snap_radius"], how="inner")
        snapped = snapped[~snapped.index.duplicated(keep="first")]

        # Update Stats per (segment, bin) — vectorized Welford batch update
        for (sid, tbin), grp in snapped.groupby(["segment_id", "time_bin"]):
            speeds = grp["speed"].values.astype(float)
            key = (sid, tbin)
            
            if key not in accum:
                accum[key] = {"n": 0, "mean": 0.0, "M2": 0.0, "below": 0}
            
            a = accum[key]
            # Vectorized Welford: merge existing stats with new batch in one step
            n_b = len(speeds)
            mean_b = speeds.mean()
            M2_b   = speeds.var() * n_b  # sum of squared deviations
            n_new  = a["n"] + n_b
            delta  = mean_b - a["mean"]
            a["mean"] = (a["n"] * a["mean"] + n_b * mean_b) / n_new
            a["M2"]  += M2_b + delta ** 2 * a["n"] * n_b / n_new
            a["n"]    = n_new
            a["below"] += int(np.sum(speeds < cfg["speed_low"]))

def main():
    cfg = load_config()
    os.makedirs(cfg["output_dir"], exist_ok=True)
    
    output_path = os.path.join(cfg["output_dir"], "feat_probe.parquet")
    state_file = os.path.join(cfg["output_dir"], "feat_probe_state.pkl")
    accum = {} # (sid, bin) -> stats
    processed_files = set()

    print(f"--- Feature Engineering: Probe Data (Time-Binned) ---")
    
    if os.path.exists(output_path) and not os.path.exists(state_file):
        print(f"\n[!] WARNING: Final output already exists at: {output_path}")
        if input("    Overwrite and restart? (y/n): ").lower() != 'y':
            return

    if os.path.exists(state_file):
        print(f"Loading checkpoint from {state_file}...")
        with open(state_file, "rb") as f:
            state = pickle.load(f)
            accum = state["accum"]
            processed_files = state["processed_files"]
        print(f"  Resuming: {len(processed_files)} months already completed.")

    print("Loading road segments...")
    # FIX: Must include geometry column for spatial join to work!
    segments = gpd.read_file(cfg["segments"])[["segment_id", "geometry"]]
    
    probe_files = sorted(glob.glob(os.path.join(cfg["probe_dir"], "*.parquet")))
    to_process = [f for f in probe_files if os.path.basename(f) not in processed_files]
    
    if to_process:
        for path in tqdm(to_process, desc="Months"):
            process_file(path, segments, cfg, accum)
            processed_files.add(os.path.basename(path))
            with open(state_file, "wb") as f:
                pickle.dump({"accum": accum, "processed_files": processed_files}, f)

    # --- Aggregate and Format (Wide Format) ---
    print("\nFinalizing binned features...")
    if not accum:
        print("No data collected.")
        return

    # Convert accumulator dict to a DataFrame
    rows = []
    for (sid, tbin), a in accum.items():
        rows.append({
            "segment_id": sid,
            "tbin": tbin,
            "count": a["n"],
            "speed": round(a["mean"], 2),
            "pct_below": round((a["below"] / a["n"]) * 100, 2) if a["n"] > 0 else 0.0
        })
    df_long = pd.DataFrame(rows)

    # Create Wide Format for counts, speeds, and congestion
    df_counts  = df_long.pivot(index="segment_id", columns="tbin", values="count").add_prefix("probe_count_")
    df_speeds  = df_long.pivot(index="segment_id", columns="tbin", values="speed").add_prefix("speed_mean_")
    df_congestion = df_long.pivot(index="segment_id", columns="tbin", values="pct_below").add_prefix("pct_below_20kmh_")
    df_wide = pd.concat([df_counts, df_speeds, df_congestion], axis=1).reset_index()
    
    # Calculate GLOBAL totals (weighted average of means)
    print("Calculating global aggregates...")
    df_global = df_long.groupby("segment_id").apply(lambda x: pd.Series({
        "probe_count": x["count"].sum(),
        "speed_mean": round((x["count"] * x["speed"]).sum() / x["count"].sum(), 2)
    }), include_groups=False).reset_index()

    final_df = df_wide.merge(df_global, on="segment_id", how="left")

    # Merge with ALL segments to ensure completeness (filling 0 for count, NaN for speed)
    all_sids = pd.DataFrame({"segment_id": segments["segment_id"].unique()})
    final_df = all_sids.merge(final_df, on="segment_id", how="left")
    
    # Fill 0s for count columns
    count_cols = [c for c in final_df.columns if "count" in c]
    final_df[count_cols] = final_df[count_cols].fillna(0).astype(int)

    # Add probe data flag: 1 if segment has any probe pings, 0 if fully unobserved
    final_df["has_probe_data"] = (final_df["probe_count"] > 0).astype(int)

    final_df.to_parquet(output_path, index=False)
    if os.path.exists(state_file): os.remove(state_file)
    print(f"Success → {output_path}")

if __name__ == "__main__":
    main()
