"""
Road segmentation + intersection detection for Bangkok.

1. Filter OSM roads to Bangkok bbox, keep drivable highway types
2. Split LineStrings into fixed-length segments (~100m)
3. Detect intersections (nodes where ≥3 road endpoints meet)
4. Save road_segments.gpkg and intersections.gpkg

Config: configs/data_sources.yaml
"""

import os
import yaml
import numpy as np
import geopandas as gpd
from shapely.ops import substring
from shapely.geometry import Point
from collections import Counter

CONFIG_PATH = "configs/data_sources.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seg_cfg  = cfg["road_segments"]
    return {
        "boundary":             cfg["boundary"]["bangkok"],
        "input":                seg_cfg["input"],
        "output":               seg_cfg["output"],
        "intersections_output": seg_cfg["intersections_output"],
        "seg_len":              seg_cfg["segment_length_m"],
        "crs":                  seg_cfg["projected_crs"],
        "drivable":             set(seg_cfg["drivable_highway_types"]),
    }


def split_line(geom, seg_len):
    total = geom.length
    if total <= seg_len:
        return [geom]
    cuts = np.arange(0, total, seg_len)
    return [substring(geom, s, min(s + seg_len, total)) for s in cuts]


def detect_intersections(roads, crs, precision=1):
    """
    Return a GeoDataFrame of intersection points.
    A node is an intersection if ≥3 road endpoints share the same
    rounded coordinate (precision = decimal places in metres).
    """
    endpoint_counts = Counter()
    for geom in roads.geometry:
        coords = list(geom.coords)
        for coord in [coords[0], coords[-1]]:
            key = (round(coord[0], precision), round(coord[1], precision))
            endpoint_counts[key] += 1

    intersection_coords = [k for k, v in endpoint_counts.items() if v >= 3]
    print(f"  Intersections found: {len(intersection_coords):,}")

    gdf = gpd.GeoDataFrame(
        {"node_id": range(len(intersection_coords)),
         "geometry": [Point(x, y) for x, y in intersection_coords]},
        crs=crs,
    )
    return gdf


def main():
    cfg = load_config()

    print("Loading Bangkok boundary...")
    boundary = gpd.read_file(cfg["boundary"]).to_crs("EPSG:4326")
    bkk_geom = boundary.union_all()
    bkk_bbox = bkk_geom.bounds

    print("Loading OSM roads...")
    roads = gpd.read_file(cfg["input"], bbox=bkk_bbox)
    roads = roads[roads["highway"].isin(cfg["drivable"])].copy()
    # Clip to Bangkok polygon
    roads = roads[roads.geometry.intersects(bkk_geom)].copy()
    print(f"  Drivable roads in Bangkok: {len(roads):,}")

    roads = roads.to_crs(cfg["crs"])
    roads = roads.explode(index_parts=False).reset_index(drop=True)

    # --- Intersection detection (on full road lines, before splitting) ---
    print("Detecting intersections...")
    intersections = detect_intersections(roads, cfg["crs"])

    # --- Split into segments ---
    print(f"Splitting into ~{cfg['seg_len']}m segments...")
    records = []
    for _, row in roads.iterrows():
        for seg in split_line(row.geometry, cfg["seg_len"]):
            if seg.length > 1:
                records.append({
                    "osm_id":   row["id"],
                    "highway":  row["highway"],
                    "name":     row.get("name"),
                    "oneway":   row.get("oneway"),
                    "lanes":    row.get("lanes"),
                    "length_m": round(seg.length, 2),
                    "geometry": seg,
                })

    segments = gpd.GeoDataFrame(records, crs=cfg["crs"])
    segments.insert(0, "segment_id", range(len(segments)))
    print(f"  Total segments: {len(segments):,}")

    # --- Save ---
    os.makedirs(os.path.dirname(cfg["output"]) or ".", exist_ok=True)

    segments.to_file(cfg["output"], driver="GPKG")
    print(f"  Saved segments  → {cfg['output']}")

    intersections.to_file(cfg["intersections_output"], driver="GPKG")
    print(f"  Saved intersections → {cfg['intersections_output']}")


if __name__ == "__main__":
    main()
