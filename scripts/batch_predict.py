"""
batch_predict.py — Batch Sentiment Prediction Pipeline

Loads a fine-tuned transformer model and tokenizer, reads unseen text
from a CSV file, and writes per-row predictions (sentiment label +
confidence score) to an output CSV.

Usage:
    python scripts/batch_predict.py

Environment variables (all optional):
    MODEL_PATH   Directory containing saved model & tokenizer
                 (default: model_output)
"""

import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.getenv("MODEL_PATH", "model_output")
MODEL_DIR = os.path.join(PROJECT_ROOT, MODEL_PATH)

INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "unseen", "predict_data.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "predictions.csv")

# Sentiment label mapping (IMDB convention: 0 = negative, 1 = positive)
LABEL_MAP = {0: "negative", 1: "positive"}

# Batch size for tokenisation / inference (tune for your GPU memory)
INFERENCE_BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Main prediction pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("  Sentiment Analysis — Batch Prediction Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load model & tokenizer (once)
    # ------------------------------------------------------------------
    print(f"\n[INFO] Loading model from '{MODEL_DIR}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    except Exception as exc:
        print(
            f"[ERROR] Failed to load model/tokenizer from '{MODEL_DIR}': {exc}\n"
            "       Make sure you have run `python scripts/train.py` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Use GPU when available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"[INFO] Model loaded successfully (device: {device})")

    # ------------------------------------------------------------------
    # 2. Load input data
    # ------------------------------------------------------------------
    print(f"\n[INFO] Reading input data from '{INPUT_CSV}'...")
    try:
        input_df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(
            f"[ERROR] Input file not found: {INPUT_CSV}\n"
            "       Please provide a CSV with a 'text' column.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] Failed to read input CSV: {exc}", file=sys.stderr)
        sys.exit(1)

    if "text" not in input_df.columns:
        print(
            "[ERROR] Input CSV must contain a 'text' column.",
            file=sys.stderr,
        )
        sys.exit(1)

    total_rows = len(input_df)
    print(f"[INFO] Found {total_rows:,} rows to predict.")

    # ------------------------------------------------------------------
    # 3. Run inference in batches (model loaded once)
    # ------------------------------------------------------------------
    all_sentiments: list[str] = []
    all_confidences: list[float] = []

    texts = input_df["text"].astype(str).tolist()

    print("[INFO] Running inference...")
    with torch.no_grad():
        for batch_start in range(0, total_rows, INFERENCE_BATCH_SIZE):
            batch_end = min(batch_start + INFERENCE_BATCH_SIZE, total_rows)
            batch_texts = texts[batch_start:batch_end]

            # Tokenise the batch
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )
            encodings = {k: v.to(device) for k, v in encodings.items()}

            # Forward pass
            outputs = model(**encodings)
            logits = outputs.logits

            # Softmax probabilities → predicted class + confidence
            probs = F.softmax(logits, dim=-1)
            confidences, predictions = torch.max(probs, dim=-1)

            for pred_id, conf in zip(predictions.cpu().tolist(), confidences.cpu().tolist()):
                all_sentiments.append(LABEL_MAP[pred_id])
                all_confidences.append(round(conf, 4))

            print(
                f"    -> Processed {batch_end:,}/{total_rows:,} samples"
            )

    # ------------------------------------------------------------------
    # 4. Build & save output dataframe
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_df = pd.DataFrame(
        {
            "text": texts,
            "predicted_sentiment": all_sentiments,
            "confidence": all_confidences,
        }
    )
    output_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n[INFO] Predictions saved to {OUTPUT_CSV}")
    print(f"       Total predictions : {len(output_df):,}")

    # Quick summary
    pos_count = all_sentiments.count("positive")
    neg_count = all_sentiments.count("negative")
    avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    print(f"       Positive          : {pos_count:,}")
    print(f"       Negative          : {neg_count:,}")
    print(f"       Avg confidence    : {avg_conf:.4f}")

    print("\n" + "=" * 60)
    print("  Batch prediction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
