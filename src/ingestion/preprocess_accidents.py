"""
preprocess_accidents.py
-----------------------
Reads raw full-Thailand CSVs from data/raw/MOT_accident_data/,
filters to Bangkok, cleans, and produces:
  - accidents_clean.gpkg     (EPSG:4326  — open in QGIS)
  - accidents_clean.parquet  (EPSG:32647 — ready for feature engineering)

All paths and settings are read from configs/data_sources.yaml.

Run from project root:
    python src/ingestion/preprocess_accidents.py
"""

import glob
import os
import yaml
import geopandas as gpd
import pandas as pd

SEVERITY_LABEL = {0: "none", 1: "minor", 2: "serious", 3: "fatal"}


def load_config(config_path="configs/data_sources.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _severity(row) -> int:
    if row["ผู้เสียชีวิต"] >= 1:
        return 3
    if row["ผู้บาดเจ็บสาหัส"] >= 1:
        return 2
    if row["ผู้บาดเจ็บเล็กน้อย"] >= 1:
        return 1
    return 0


def _time_bin(h: int) -> str:
    if   6  <= h < 9:  return "morning_peak"
    elif 9  <= h < 16: return "daytime"
    elif 16 <= h < 20: return "evening_peak"
    elif 20 <= h < 24: return "night"
    else:              return "late_night"


def main():
    config        = load_config()
    bkk_path      = config["boundary"]["bangkok"]
    acc_cfg       = config["accidents"]
    raw_dir       = acc_cfg["raw_dir"]
    out_gpkg      = acc_cfg["clean_gpkg"]
    out_parquet   = acc_cfg["clean_parquet"]
    projected_crs = acc_cfg["projected_crs"]
    valid_weather = set(acc_cfg["valid_weather"])

    os.makedirs(os.path.dirname(out_gpkg), exist_ok=True)

    # ── 1. Load all raw CSVs ────────────────────────────────────────
    raw_files = sorted(glob.glob(f"{raw_dir}/accidents_*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"No raw CSVs found in {raw_dir}/. Run load_accident_data.py first.")

    print(f"Loading {len(raw_files)} raw CSV(s) from {raw_dir} …")
    frames = [pd.read_csv(f, encoding="utf-8-sig", low_memory=False) for f in raw_files]

    # ── 2. Parse & merge datetime ─────────────────────────────────────────────
    for i, frame in enumerate(frames):
        if i < 4:
            frame['วันที่เกิดเหตุ'] = pd.to_datetime(frame['วันที่เกิดเหตุ'])
            frame['เวลา'] = pd.to_timedelta(frame['เวลา'] + ':00')
        elif i == 4:
            frame['วันที่เกิดเหตุ'] = pd.to_datetime(frame['วันที่เกิดเหตุ'], format='%m/%d/%Y')
            frame['เวลา'] = pd.to_timedelta(frame['เวลา'] + ':00')           
        elif i == 5:
            frame['วันที่เกิดเหตุ'] = pd.to_datetime(frame['วันที่เกิดเหตุ'], origin='1899-12-30', unit='D')
            frame['เวลา'] = pd.to_timedelta(frame['เวลา'], unit='D')
        elif i == 6:
            frame['วันที่เกิดเหตุ'] = pd.to_datetime(frame['วันที่เกิดเหตุ'], origin='1899-12-30', unit='D')
            frame['เวลา'] = pd.to_timedelta(frame['เวลา'] + ':00')
        frame['วันที่และเวลาที่เกิดเหตุ'] = (frame['วันที่เกิดเหตุ'] + frame['เวลา']).dt.round('s')
        frames[i] = frame.drop(columns=["วันที่เกิดเหตุ", "เวลา", "วันที่รายงาน", "เวลาที่รายงาน"], errors="ignore")

    # ── 3. merge all CSVs ────────────────────────────────────────
    df = pd.concat(frames, ignore_index=True)
    print(f"  Total rows (all Thailand): {len(df):,}")
    
    # ── 4. Build geometry ─────────────────────────────────────────────────────
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            pd.to_numeric(df["LONGITUDE"], errors="coerce"),
            pd.to_numeric(df["LATITUDE"],  errors="coerce"),
        ),
        crs="EPSG:4326",
    )
    gdf = gdf.drop(columns=["LATITUDE", "LONGITUDE"])

    # ── 5. Filter to Bangkok boundary ─────────────────────────────────────────────
    bkk = gpd.read_file(bkk_path)
    bkk = bkk.to_crs(gdf.crs)
    gdf = gpd.sjoin(gdf, bkk, how="inner", predicate="within")

    # ── 6. Drop column-shift rows ─────────────────────────────────────────────
    # ~6 000 rows have a lat value in สภาพอากาศ — unrecoverable, drop them
    col_shift = ~gdf["สภาพอากาศ"].isin(valid_weather | {None}) & gdf["สภาพอากาศ"].notna()
    n_shift = col_shift.sum()
    gdf = gdf[~col_shift].copy()
    print(f"  Dropped column-shift rows: {n_shift:,}")

    # ── 7. Normalise categorical columns ─────────────────────────────────────
    gdf["สภาพอากาศ"] = gdf["สภาพอากาศ"].apply(
        lambda v: v if v in valid_weather else ("ไม่ระบุ" if pd.isna(v) else "อื่นๆ")
    )
    for col in ["มูลเหตุสันนิษฐาน", "ลักษณะการเกิดเหตุ"]:
        gdf[col] = gdf[col].fillna("ไม่ระบุ")

    # ── 8. Datetime features ──────────────────────────────────────────────────
    dt = gdf["วันที่และเวลาที่เกิดเหตุ"]
    gdf["hour"]        = dt.dt.hour
    gdf["day_of_week"] = dt.dt.dayofweek
    gdf["month"]       = dt.dt.month
    gdf["year"]        = dt.dt.year
    gdf["time_bin"]    = gdf["hour"].apply(_time_bin)

    # ── 9. Severity ───────────────────────────────────────────────────────────
    gdf["severity"]       = gdf.apply(_severity, axis=1)
    gdf["severity_label"] = gdf["severity"].map(SEVERITY_LABEL)

    # ── 10. UTM coords ─────────────────────────────────────────────────────────
    gdf_utm = gdf.to_crs(projected_crs)
    gdf["x_utm"] = gdf_utm.geometry.x
    gdf["y_utm"] = gdf_utm.geometry.y

    print(f"  Final clean rows: {len(gdf):,}")

    # ── 11. Save ──────────────────────────────────────────────────────────────
    gdf.to_file(out_gpkg, driver="GPKG")
    print(f"  Saved → {out_gpkg}")

    gdf_utm["x_utm"] = gdf_utm.geometry.x
    gdf_utm["y_utm"] = gdf_utm.geometry.y
    gdf_utm.to_parquet(out_parquet, index=False)
    print(f"  Saved → {out_parquet}")

    # ── 12. Summary ───────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────────")
    print(gdf["severity_label"].value_counts().to_string())
    print()
    print(gdf["year"].value_counts().sort_index().to_string())
    print()
    print(gdf["time_bin"].value_counts().to_string())
    print("─────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
