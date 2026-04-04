"""
Upload CulturalGround VQA data to Hugging Face for multiple countries.

Usage:
    python scripts/upload_cultural_ground.py
"""

import os
import shutil
import subprocess
import ijson
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, Features, Value, Image as HFImage
from PIL import Image

HF_REPO = "davidguzmanr/CulturalGround"
BASE_DIR = "CultureGroundImages"
JSON_FILE = os.path.join(BASE_DIR, "CulturalGround-OE-Filtered-14M.json")
MAX_SAMPLES = 10_000

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


def build_dataframe(country: str) -> pd.DataFrame:
    """Stream the JSON and collect rows for the given country, English only."""
    keyword = f"/{country}/"
    rows = []
    with open(JSON_FILE, "rb") as f:
        for item in ijson.items(f, "item"):
            if keyword in item.get("image", "").lower():
                rows.append(item)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["image"] = df["image"].apply(os.path.basename)
    df["image_path"] = df["image"].apply(
        lambda x: os.path.join(BASE_DIR, country, x)
    )
    df = df[df["language"] == "en"].drop(columns=["image"]).reset_index(drop=True)
    return df


def load_image(path: str):
    if os.path.exists(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None
    return None


def build_hf_dataset(df: pd.DataFrame) -> Dataset:
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "id": row["id"],
                "conversations": row["conversations"],
                "image": load_image(row["image_path"]),
                "language": row["language"],
            }
        )

    features = Features(
        {
            "id": Value("string"),
            "conversations": [{"from": Value("string"), "value": Value("string")}],
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

            print(f"Building dataframe for {country} ...")
            df = build_dataframe(country)

            if df.empty:
                print(f"No English samples found for {country}, skipping.")
                continue

            if len(df) > MAX_SAMPLES:
                df = df.sample(MAX_SAMPLES, random_state=42).reset_index(drop=True)

            print(f"{len(df)} samples - building HF dataset ...")
            ds = build_hf_dataset(df)

            print(f"Uploading to {HF_REPO} (config_name={country}) ...")
            ds.push_to_hub(HF_REPO, config_name=country, split="train", commit_message=f"Uploaded {country}")

            print(f"Done: {country} ({len(ds)} rows uploaded)")

            # Clean up to free disk space
            tar_path = os.path.join(BASE_DIR, f"{country}.tar.gz")
            country_dir = os.path.join(BASE_DIR, country)
            if os.path.exists(tar_path):
                os.remove(tar_path)
                print(f"Removed {tar_path}")
            if os.path.isdir(country_dir):
                shutil.rmtree(country_dir)
                print(f"Removed {country_dir}/")

        except Exception as e:
            print(f"ERROR processing {country}: {e}")
            continue


if __name__ == "__main__":
    main()