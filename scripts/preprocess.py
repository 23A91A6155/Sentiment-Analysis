"""
preprocess.py — IMDB Dataset Preprocessing Pipeline

Loads the IMDB dataset from Hugging Face, applies text-cleaning
transformations, and persists clean train/test CSV files for
downstream model training.

Usage:
    python scripts/preprocess.py
"""

import os
import re
import sys

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

# All paths are relative to the project root (one level above scripts/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TRAIN_OUTPUT = os.path.join(PROCESSED_DIR, "train.csv")
TEST_OUTPUT = os.path.join(PROCESSED_DIR, "test.csv")


# ---------------------------------------------------------------------------
# Text-cleaning helpers
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Apply successive cleaning steps to a single text string.

    Steps
    -----
    1. Remove URLs (http/https/www patterns).
    2. Remove HTML tags (e.g. ``<br />``, ``<p>``).
    3. Remove special characters — keep only alphanumeric chars and
       basic punctuation (.,!?;:'-).
    4. Collapse multiple whitespace characters into a single space and
       strip leading/trailing whitespace.
    """
    # 1. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 2. Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # 3. Remove special characters (keep alphanumeric + basic punctuation)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?;:'\-]", "", text)

    # 4. Collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def process_split(split_data, split_name: str, output_path: str) -> None:
    """Clean every example in *split_data* and save to *output_path*.

    Parameters
    ----------
    split_data : datasets.Dataset
        A Hugging Face ``Dataset`` object with ``text`` and ``label`` columns.
    split_name : str
        Human-readable name for progress messages (e.g. ``'train'``).
    output_path : str
        Absolute path where the resulting CSV will be written.
    """
    print(f"  [INFO] Processing '{split_name}' split ({len(split_data):,} samples)...")

    texts = split_data["text"]
    labels = split_data["label"]

    cleaned_texts = []
    total = len(texts)
    for idx, raw_text in enumerate(texts, start=1):
        cleaned_texts.append(clean_text(raw_text))
        if idx % 5000 == 0 or idx == total:
            print(f"    -> Cleaned {idx:,}/{total:,} samples")

    df = pd.DataFrame({"text": cleaned_texts, "label": labels})
    df.to_csv(output_path, index=False)
    print(f"  [INFO] Saved {split_name} data to {output_path} ({len(df):,} rows)\n")


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------
def main() -> None:
    """Load the IMDB dataset, clean both splits, and write CSVs."""
    print("=" * 60)
    print("  IMDB Dataset Preprocessing Pipeline")
    print("=" * 60)

    # Ensure output directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print(f"\n[INFO] Output directory: {PROCESSED_DIR}")

    # Load dataset from Hugging Face Hub
    print("[INFO] Loading IMDB dataset from Hugging Face...")
    try:
        dataset = load_dataset("imdb")
    except Exception as exc:
        print(f"[ERROR] Failed to load IMDB dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Dataset loaded successfully.")
    print(f"       Train samples : {len(dataset['train']):,}")
    print(f"       Test  samples : {len(dataset['test']):,}\n")

    # Process each split
    process_split(dataset["train"], "train", TRAIN_OUTPUT)
    process_split(dataset["test"], "test", TEST_OUTPUT)

    print("=" * 60)
    print("  Preprocessing complete!")
    print(f"  Train -> {TRAIN_OUTPUT}")
    print(f"  Test  -> {TEST_OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
