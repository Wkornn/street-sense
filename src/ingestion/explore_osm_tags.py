"""
Exploration script — run this to see what VALUES each OSM key has in Bangkok.
Use the output to decide what values to put in configs/data_sources.yaml.

Output: data/raw/osm_schema.csv
"""
import os
import osmnx as ox
import pandas as pd

SAMPLE_PLACE = "Bangkok, Thailand"

# Uncomment keys you want to explore
keys_to_explore = [
    # "aerialway",
    # "aeroway",
    "amenity",
    # "barrier",
    # "boundary",
    # "building",
    # "craft",
    # "emergency",
    # "healthcare",
    # "highway",
    # "historic",
    # "landuse",
    # "leisure",
    # "man_made",
    # "natural",
    # "office",
    # "place",
    # "power",
    # "public_transport",
    # "railway",
    "shop",
    # "tourism",
    # "waterway",
]


def main():
    all_pairs = []

    for key in keys_to_explore:
        print(f"\n{'='*40}")
        print(f"KEY: {key}")

        try:
            gdf = ox.features_from_place(SAMPLE_PLACE, tags={key: True})
        except Exception as e:
            print(f"  No data: {e}")
            continue

        print(f"  {len(gdf)} features found")
        print(f"  Values:")
        for val, count in gdf[key].value_counts().items():
            print(f"    {val:<35} {count}")
            all_pairs.append({"key": key, "value": val, "count": count})

    output_path = "data/raw/osm/osm_schema.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pd.DataFrame(all_pairs).to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
