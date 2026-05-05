import urllib.request
import ssl
import json
import csv
import time
import yaml
import os

BASE_URL = "https://datagov.mot.go.th/api/3/action/datastore_search"
context = ssl._create_unverified_context()


def fetch_and_save(resource_id, output_path, limit=1000, sleep_time=0.3):
    offset = 0
    total = 0
    first_batch = True

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Start downloading → {resource_id}")

    while True:
        url = f"{BASE_URL}?resource_id={resource_id}&limit={limit}&offset={offset}"
        try:
            with urllib.request.urlopen(url, context=context) as response:
                data = json.loads(response.read().decode("utf-8"))
                records = data["result"]["records"]

                if not records:
                    print("Finished downloading.")
                    break

                mode = "w" if first_batch else "a"
                with open(output_path, mode, newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=records[0].keys())
                    if first_batch:
                        writer.writeheader()
                        first_batch = False
                    writer.writerows(records)

                batch_size = len(records)
                total += batch_size
                print(f"Batch: {batch_size} | Total: {total}")

                offset += batch_size
                time.sleep(sleep_time)

        except Exception as e:
            print(f"Error at offset {offset}: {e}")
            break

    print(f"Saved → {output_path} ({total} records)")


def load_config(config_path="configs/data_sources.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config    = load_config("configs/data_sources.yaml")
    inventory = load_config("configs/data_inventory.yaml")
    raw_dir   = config["accidents"]["raw_dir"]

    for dataset in inventory["datasets"]:
        year        = dataset["year"]
        resource_id = dataset["resource_id"]
        raw_path    = f"{raw_dir}/accidents_{year}.csv"

        print(f"\n=== Year {year} ===")

        if os.path.exists(raw_path):
            print(f"  Already exists, skipping.")
            continue

        fetch_and_save(resource_id, raw_path)

    print(f"\nDone → {raw_dir}")


if __name__ == "__main__":
    main()
