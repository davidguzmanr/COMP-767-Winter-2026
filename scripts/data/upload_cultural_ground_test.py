"""
Upload CulturalGround VQA test data to Hugging Face for multiple countries.

Filters out training IDs already in davidguzmanr/CulturalGround, then uploads
up to 20 samples per country to davidguzmanr/CulturalGround-test.

Usage:
    python scripts/data/upload_cultural_ground_test.py
"""

import os
import subprocess
import ijson
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, Features, Value, Image as HFImage, load_dataset
from PIL import Image

HF_TRAIN_REPO = "davidguzmanr/CulturalGround"
HF_TEST_REPO = "davidguzmanr/CulturalGround-test"
BASE_DIR = "CultureGroundImages"
JSON_FILE = os.path.join(BASE_DIR, "CulturalGround-OE-Filtered-14M.json")
MAX_TEST_SAMPLES = 20

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


def get_train_ids(country: str) -> set:
    """Load the training split from HF and return the set of sample IDs."""
    try:
        ds = load_dataset(HF_TRAIN_REPO, country)["train"]
    except Exception as e:
        print(f"  Could not load training data for {country}: {e}")
        return set()
    return {item["id"] for item in ds}


def build_test_dataframe(country: str, ids_train: set) -> pd.DataFrame:
    """Stream the JSON, collect English rows for the country, exclude train IDs."""
    keyword = f"/{country}/"
    rows = []
    with open(JSON_FILE, "rb") as f:
        for item in ijson.items(f, "item"):
            if (
                keyword in item.get("image", "").lower()
                and item.get("language", "").lower() == "en"
            ):
                rows.append(item)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df[~df["id"].isin(ids_train)].reset_index(drop=True)
    if df.empty:
        return df

    df["image_path"] = df["image"].apply(
        lambda x: os.path.join(BASE_DIR, country, os.path.basename(x))
    )
    df["question"] = df["conversations"].apply(
        lambda x: x[0].get("value", "").replace("<image>\n", "")
    )
    df["answer"] = df["conversations"].apply(lambda x: x[1].get("value", ""))

    return df


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


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    download_json()

    for country in tqdm(COUNTRIES, desc="Countries"):
        try:
            print(f"\n=== {country} ===")

            download_and_extract(country)

            print(f"Loading train IDs for {country} ...")
            ids_train = get_train_ids(country)
            print(f"  {len(ids_train)} training IDs to exclude.")

            print(f"Building test dataframe for {country} ...")
            df = build_test_dataframe(country, ids_train)

            if df.empty:
                print(f"No eligible test samples found for {country}, skipping.")
                continue

            df = df.sample(
                min(MAX_TEST_SAMPLES, len(df)), random_state=42
            ).reset_index(drop=True)

            print(f"{len(df)} samples - building HF test dataset ...")
            ds = build_hf_test_dataset(df)

            print(f"Uploading to {HF_TEST_REPO} (config_name={country}) ...")
            ds.push_to_hub(
                HF_TEST_REPO,
                config_name=country,
                split="test",
                commit_message=f"Uploaded test split for {country}",
            )

            print(f"Done: {country} ({len(ds)} rows uploaded)")

        except Exception as e:
            print(f"ERROR processing {country}: {e}")
            continue


if __name__ == "__main__":
    main()
