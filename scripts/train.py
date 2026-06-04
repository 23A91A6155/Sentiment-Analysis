"""
train.py — Sentiment Analysis Model Training Pipeline

Fine-tunes a pre-trained transformer (default: DistilBERT) on the IMDB
sentiment dataset using the Hugging Face Trainer API.  Evaluates on a
held-out test set and persists metrics, a run summary, and the trained
model artefacts.

Usage:
    python scripts/train.py

Environment variables (all optional — sensible defaults provided):
    MODEL_NAME      Pre-trained model identifier  (default: distilbert-base-uncased)
    LEARNING_RATE   Optimiser learning rate        (default: 2e-5)
    BATCH_SIZE      Per-device batch size           (default: 16)
    NUM_EPOCHS      Number of training epochs       (default: 3)
    MODEL_PATH      Directory to save model to      (default: model_output)
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased")
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))
MODEL_PATH = os.getenv("MODEL_PATH", "model_output")

TRAIN_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
TEST_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, MODEL_PATH)

# Subset sizes for practical training speed
TRAIN_SUBSET_SIZE = 200
TEST_SUBSET_SIZE = 100


# ---------------------------------------------------------------------------
# Custom PyTorch Dataset
# ---------------------------------------------------------------------------
class SentimentDataset(Dataset):
    """Wraps tokenised text and integer labels for PyTorch data-loading."""

    def __init__(self, encodings: dict, labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Metric computation callback for Trainer
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred) -> dict:
    """Return accuracy, precision, recall, and F1 from Trainer predictions."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="binary"),
        "recall": recall_score(labels, predictions, average="binary"),
        "f1_score": f1_score(labels, predictions, average="binary"),
    }


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("  Sentiment Analysis — Training Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load & subsample data
    # ------------------------------------------------------------------
    print("\n[INFO] Loading processed datasets...")
    try:
        train_df = pd.read_csv(TRAIN_CSV)
        test_df = pd.read_csv(TEST_CSV)
    except FileNotFoundError as exc:
        print(
            f"[ERROR] Processed data not found: {exc}\n"
            "       Run `python scripts/preprocess.py` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"       Full train size : {len(train_df):,}")
    print(f"       Full test  size : {len(test_df):,}")

    # Reproducible subsets for practical training time
    train_df = train_df.sample(
        n=min(TRAIN_SUBSET_SIZE, len(train_df)), random_state=42
    ).reset_index(drop=True)
    test_df = test_df.sample(
        n=min(TEST_SUBSET_SIZE, len(test_df)), random_state=42
    ).reset_index(drop=True)

    print(f"       Train subset    : {len(train_df):,}")
    print(f"       Test  subset    : {len(test_df):,}")

    # ------------------------------------------------------------------
    # 2. Tokenise
    # ------------------------------------------------------------------
    print(f"\n[INFO] Loading tokenizer for '{MODEL_NAME}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as exc:
        print(f"[ERROR] Failed to load tokenizer: {exc}", file=sys.stderr)
        sys.exit(1)

    print("[INFO] Tokenizing train split...")
    train_encodings = tokenizer(
        train_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=512,
    )

    print("[INFO] Tokenizing test split...")
    test_encodings = tokenizer(
        test_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=512,
    )

    train_dataset = SentimentDataset(train_encodings, train_df["label"].tolist())
    test_dataset = SentimentDataset(test_encodings, test_df["label"].tolist())

    # ------------------------------------------------------------------
    # 3. Load pre-trained model
    # ------------------------------------------------------------------
    print(f"\n[INFO] Loading pre-trained model '{MODEL_NAME}'...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        )
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Configure Trainer
    # ------------------------------------------------------------------
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_score",
        seed=42,
        report_to="none",  # disable W&B / MLflow unless configured
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print("\n[INFO] Starting training...")
    print(f"       Model          : {MODEL_NAME}")
    print(f"       Learning rate  : {LEARNING_RATE}")
    print(f"       Batch size     : {BATCH_SIZE}")
    print(f"       Epochs         : {NUM_EPOCHS}")
    print(f"       Train samples  : {len(train_dataset):,}")
    print(f"       Test  samples  : {len(test_dataset):,}")
    print()

    try:
        trainer.train()
    except Exception as exc:
        print(f"[ERROR] Training failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    print("\n[INFO] Evaluating on test set...")
    eval_results = trainer.evaluate()

    metrics = {
        "accuracy": round(eval_results["eval_accuracy"], 4),
        "precision": round(eval_results["eval_precision"], 4),
        "recall": round(eval_results["eval_recall"], 4),
        "f1_score": round(eval_results["eval_f1_score"], 4),
    }

    print("\n  --- Evaluation Metrics ---")
    for key, value in metrics.items():
        print(f"  {key:>12s} : {value:.4f}")

    # ------------------------------------------------------------------
    # 7. Save metrics
    # ------------------------------------------------------------------
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\n[INFO] Metrics saved to {metrics_path}")

    # ------------------------------------------------------------------
    # 8. Save run summary
    # ------------------------------------------------------------------
    run_summary = {
        "hyperparameters": {
            "model_name": MODEL_NAME,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
        },
        "final_metrics": {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
        },
    }

    summary_path = os.path.join(RESULTS_DIR, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(run_summary, fh, indent=2)
    print(f"[INFO] Run summary saved to {summary_path}")

    # ------------------------------------------------------------------
    # 9. Save model and tokenizer
    # ------------------------------------------------------------------
    print(f"\n[INFO] Saving model to {MODEL_OUTPUT_DIR}...")
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    # Verify expected artefacts exist
    expected_files = ["config.json", "tokenizer_config.json"]
    for fname in expected_files:
        fpath = os.path.join(MODEL_OUTPUT_DIR, fname)
        if os.path.isfile(fpath):
            print(f"       [OK] {fname}")
        else:
            print(f"       [MISSING] {fname} (missing)")

    # vocab.txt is produced by BERT-family tokenizers; others may use
    # different filenames (e.g. sentencepiece.bpe.model)
    vocab_path = os.path.join(MODEL_OUTPUT_DIR, "vocab.txt")
    if os.path.isfile(vocab_path):
        print(f"       [OK] vocab.txt")
    else:
        print(f"       [WARN] vocab.txt not found (tokenizer may use a different format)")

    # Model weights
    weight_files = [f for f in os.listdir(MODEL_OUTPUT_DIR) if "model" in f.lower()]
    for wf in weight_files:
        print(f"       [OK] {wf}")

    print("\n" + "=" * 60)
    print("  Training pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
