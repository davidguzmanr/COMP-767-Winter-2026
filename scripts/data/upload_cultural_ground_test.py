"""
Upload CulturalGround VQA test data to Hugging Face for multiple countries.

Filters out training IDs already in davidguzmanr/CulturalGround, then uploads
up to 20 samples per country to davidguzmanr/CulturalGround-test.

Optimizations:
  - Single JSON pass to collect rows for all countries at once.
  - Parallel download/extract + dataset building via ThreadPoolExecutor.
  - HF uploads are serialized (one at a time) to avoid rate-limit issues.

Usage:
    python scripts/data/upload_cultural_ground_test.py
"""

import os
import subprocess
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import ijson
import pandas as pd
from datasets import Dataset, Features, Value, Image as HFImage, load_dataset
from PIL import Image
from tqdm import tqdm

HF_TRAIN_REPO = "davidguzmanr/CulturalGround"
HF_TEST_REPO = "davidguzmanr/CulturalGround-test"
BASE_DIR = "CultureGroundImages"
JSON_FILE = os.path.join(BASE_DIR, "CulturalGround-OE-Filtered-14M.json")
MAX_TEST_SAMPLES = 20
NUM_WORKERS = 4  # parallel workers for download/extract/build

COUNTRIES = [
    "bangladesh",
    "brazil",
    "bulgaria",
    "china",
    "czechia",
    "egypt",
    "ethiopia",
    "france",
    "germany",
    "greece",
    "india",
    "indonesia",
    "iran",
    "ireland",
    "israel",
    "italy",
    "japan",
    "kenya",
    "malaysia",
    "mexico",
    "mongolia",
    "netherlands",
    "nigeria",
    "norway",
    "pakistan",
    "poland",
    "portugal",
    "romania",
    "russia",
    "rwanda",
    "saudi_arabia",
    "singapore",
    "south_korea",
    "spain",
    "sri_lanka",
    "taiwan",
    "tanzania",
    "thailand",
    "turkey",
    "ukraine",
    "united_kingdom",
    "vietnam",
]

# Lock so only one thread uploads to HF at a time
_upload_lock = threading.Lock()


def download_json():
    """Download the shared JSON annotation file if not already present."""
    if os.path.exists(JSON_FILE):
        print(f"JSON already exists at {JSON_FILE}, skipping download.")
        return
    print("Downloading CulturalGround-OE-Filtered-14M.json ...")
    subprocess.run(
        [
            "hf", "download", "neulab/CulturalGround",
            "CulturalGround-OE-Filtered-14M.json",
            "--repo-type", "dataset",
            "--local-dir", BASE_DIR,
        ],
        check=True,
    )


def download_and_extract(country: str):
    """Download and extract the tar.gz for a country if not already done."""
    country_dir = os.path.join(BASE_DIR, country)
    if os.path.isdir(country_dir) and os.listdir(country_dir):
        print(f"Images for {country} already extracted, skipping.")
        return

    tar_path = os.path.join(BASE_DIR, f"{country}.tar.gz")

    if not os.path.exists(tar_path):
        print(f"Downloading {country}.tar.gz ...")
        subprocess.run(
            [
                "hf", "download", "neulab/CulturalGround",
                f"CultureGroundImages/{country}.tar.gz",
                "--repo-type", "dataset",
                "--local-dir", "./",
            ],
            check=True,
        )

    print(f"Extracting {country}.tar.gz ...")
    subprocess.run(
        ["tar", "-xzf", tar_path, "-C", BASE_DIR],
        check=True,
    )


def load_image(path: str):
    if os.path.exists(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None
    return None


def scan_json_all_countries() -> dict[str, list[dict]]:
    """Single pass over the JSON — collect English rows keyed by country."""
    country_set = set(COUNTRIES)
    rows_by_country: dict[str, list[dict]] = defaultdict(list)

    print("Scanning JSON (single pass for all countries) ...")
    with open(JSON_FILE, "rb") as f:
        for item in tqdm(ijson.items(f, "item"), desc="JSON rows"):
            if item.get("language", "").lower() != "en":
                continue
            image_path = item.get("image", "").lower()
            for country in country_set:
                if f"/{country}/" in image_path:
                    rows_by_country[country].append(item)
                    break

    return rows_by_country


def get_train_ids(country: str) -> set:
    """Load the training split from HF and return the set of sample IDs."""
    try:
        ds = load_dataset(HF_TRAIN_REPO, country)["train"]
    except Exception as e:
        print(f"  Could not load training data for {country}: {e}")
        return set()
    return {item["id"] for item in ds}


def build_hf_test_dataset(df: pd.DataFrame) -> Dataset:
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "image": load_image(row["image_path"]),
                "language": row["language"],
            }
        )

    features = Features(
        {
            "id": Value("string"),
            "question": Value("string"),
            "answer": Value("string"),
            "image": HFImage(),
            "language": Value("string"),
        }
    )
    return Dataset.from_list(records, features=features)


def process_country(country: str, raw_rows: list[dict]) -> str | None:
    """Download images, filter train IDs, build and upload the test dataset."""
    print(f"\n=== {country} ===")

    download_and_extract(country)

    print(f"  Loading train IDs for {country} ...")
    ids_train = get_train_ids(country)
    print(f"  {len(ids_train)} training IDs to exclude.")

    df = pd.DataFrame(raw_rows)
    df = df[~df["id"].isin(ids_train)]
    df = df.drop_duplicates(subset="id").reset_index(drop=True)

    if df.empty:
        print(f"  No eligible test samples for {country}, skipping.")
        return None

    df["image_path"] = df["image"].apply(
        lambda x: os.path.join(BASE_DIR, country, os.path.basename(x))
    )
    df["question"] = df["conversations"].apply(
        lambda x: x[0].get("value", "").replace("<image>\n", "")
    )
    df["answer"] = df["conversations"].apply(lambda x: x[1].get("value", ""))

    df = df.sample(min(MAX_TEST_SAMPLES, len(df)), random_state=42).reset_index(drop=True)

    print(f"  {len(df)} samples — building HF test dataset ...")
    ds = build_hf_test_dataset(df)

    # Serialize uploads to avoid HF rate-limit issues
    with _upload_lock:
        print(f"  Uploading {country} to {HF_TEST_REPO} ...")
        ds.push_to_hub(
            HF_TEST_REPO,
            config_name=country,
            split="test",
            commit_message=f"Uploaded test split for {country}",
        )

    return f"{country}: {len(ds)} rows uploaded"


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    download_json()

    rows_by_country = scan_json_all_countries()

    results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_country, country, rows_by_country.get(country, [])): country
            for country in COUNTRIES
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Countries"):
            country = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"ERROR processing {country}: {e}")

    print("\n=== Summary ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
