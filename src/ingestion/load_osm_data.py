import osmnx as ox
import yaml
import os


def fetch_and_save(place, tags, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f"Layer {os.path.basename(output_path)} already exists. Skipping...")
        return

    print(f"Fetching {os.path.basename(output_path)}...")
    gdf = ox.features_from_place(place, tags=tags)

    whitelist = ['amenity', 'building', 'highway', 'landuse', 'shop', 'name', 'oneway', 'lanes']
    
    existing_cols = [c for c in whitelist if c in gdf.columns]
    gdf = gdf[existing_cols + ['geometry']]

    for col in gdf.columns:
        if col != 'geometry':
            gdf[col] = gdf[col].astype(str)

    gdf.columns = [c.replace(":", "_").replace(" ", "_") for c in gdf.columns]

    gdf.to_file(output_path, driver="GPKG")
    print(f"Saved → {output_path} ({len(gdf)} features)")


def fetch_and_save_boundary(place, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f"Boundary already exists at {output_path}. Skipping...")
        return

    print(f"Fetching boundary for {place}...")
    try:
        gdf = ox.geocode_to_gdf(place)
        gdf.to_file(output_path, driver="GPKG")
        print(f"Saved boundary → {output_path}")
    except Exception as e:
        print(f"Error fetching boundary: {e}")


def load_config(config_path="configs/data_sources.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    place = config["osm"]["place"]

    print(f"=== Downloading OSM data for: {place} ===")

    # 1. Fetch Boundary
    if "boundary" in config and "bangkok" in config["boundary"]:
        fetch_and_save_boundary(place, config["boundary"]["bangkok"])

    # 2. Fetch OSM Layers (Roads, Buildings, POIs, etc.)
    for layer in config["osm"]["layers"]:
        print(f"\n--- Layer: {layer['name']} ---")
        fetch_and_save(place, layer["tags"], layer["output_path"])


if __name__ == "__main__":
    main()
