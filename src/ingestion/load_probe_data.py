import os
import tarfile
import requests
import pandas as pd
import geopandas as gpd
from geopandas import points_from_xy
import yaml

BASE_URL = "https://itic.longdo.com/opendata/probe-data"
RAW_DIR = "data/raw/iTIC_probe_data"
OUT_DIR = "data/processed/probe_bangkok"

PROBE_COLS = ["vehicle_id", "gps_valid", "lat", "lon", "timestamp",
              "speed", "heading", "for_hire_light", "engine_acc"]


def load_boundary(path):
    bnd = gpd.read_file(path).to_crs("EPSG:4326")
    union = bnd.union_all()
    b = union.bounds  # (minx, miny, maxx, maxy)
    return union, {"lon_min": b[0], "lat_min": b[1], "lon_max": b[2], "lat_max": b[3]}


def extract_bangkok(tar_path, out_path, boundary, bbox):
    chunks = []
    with tarfile.open(tar_path, "r:bz2") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith((".csv", ".out")):
                continue
            f = tar.extractfile(member)
            for chunk in pd.read_csv(f, header=None, names=PROBE_COLS, chunksize=100_000,
                                     dtype={"gps_valid": float, "speed": float, "lat": float, "lon": float}):
                pre = chunk[
                    (chunk["gps_valid"] == 1) &
                    (chunk["speed"] > 0) &
                    (chunk["lat"].between(bbox["lat_min"], bbox["lat_max"])) &
                    (chunk["lon"].between(bbox["lon_min"], bbox["lon_max"]))
                ]
                if pre.empty:
                    continue
                gdf = gpd.GeoDataFrame(pre, geometry=points_from_xy(pre["lon"], pre["lat"]), crs="EPSG:4326")
                bkk = pre[gdf.within(boundary)]
                if not bkk.empty:
                    chunks.append(bkk)

    if chunks:
        df = pd.concat(chunks, ignore_index=True)
        df.to_parquet(out_path, index=False)
        print(f"  Saved {len(df)} Bangkok rows → {out_path}")
    else:
        print(f"  No Bangkok rows found in {tar_path}")


def download_file(url, file_path, retries=5):
    for attempt in range(1, retries + 1):
        existing_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}

        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                mode = "ab" if existing_size else "wb"
                with open(file_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
            return
        except Exception as e:
            print(f"  Attempt {attempt}/{retries} failed: {e}")
            if attempt == retries:
                raise


def load_config(config_path="configs/data_sources.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    config = load_config("configs/data_inventory.yaml")
    years = [d["year"] for d in config["datasets"]]

    config = load_config("configs/data_sources.yaml")
    boundary, bbox = load_boundary(config["boundary"]["bangkok"])

    files = [
        f"PROBE-{year}{month:02d}.tar.bz2"
        for year in years
        for month in range(1, 13)
    ]

    print(f"Processing {len(files)} files...")

    for filename in files:
        stem = filename.replace(".tar.bz2", "")
        tar_path = os.path.join(RAW_DIR, filename)
        out_path = os.path.join(OUT_DIR, f"{stem}.parquet")

        # Already extracted — skip entirely
        if os.path.exists(out_path):
            print(f"Skip (done): {filename}")
            continue

        # Download if raw tar doesn't exist
        if not os.path.exists(tar_path):
            url = f"{BASE_URL}/{filename}"
            print(f"Downloading: {filename}...")
            try:
                download_file(url, tar_path)
            except requests.HTTPError as e:
                print(f"  Not available: {e}")
                if os.path.exists(tar_path):
                    os.remove(tar_path)
                continue

        # Extract Bangkok rows then delete raw tar
        print(f"Extracting Bangkok rows: {filename}...")
        try:
            extract_bangkok(tar_path, out_path, boundary, bbox)
            os.remove(tar_path)
            print(f"  Deleted raw: {filename}")
        except Exception as e:
            print(f"  Extraction failed: {e}")

    print(f"\nDone → {OUT_DIR}")


if __name__ == "__main__":
    main()
