"""
FlowScore — Fine-tune DistilBERT for Transaction Categorization
================================================================
Extracts (merchant_name, category) pairs from the synthetic dataset,
fine-tunes distilbert-base-uncased as a 25-class classifier, and saves
the model + tokenizer to models/distilbert_categorizer/.

The trained model replaces the Claude API fallback in categorizer.py —
making the full pipeline self-contained and production-architectured.

Why DistilBERT over rules alone?
  - Handles noisy/truncated merchant strings rules miss
  - Generalizes to merchant names outside the training vocabulary
  - Produces calibrated confidence scores for each category
  - 97x faster and 40% smaller than BERT with ~97% of its accuracy

Usage:
    python src/train_distilbert.py \\
        --dataset data/flowscore_dataset.json \\
        --output models/distilbert_categorizer/ \\
        --max_per_class 4000 \\
        --epochs 3

Requirements:
    pip install transformers torch datasets accelerate
"""

import argparse
import json
import os
import random
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    from torch.optim import AdamW
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: transformers and torch are required.")
    print("       pip install transformers torch accelerate")
    exit(1)


# ============================================================================
# DATA EXTRACTION
# ============================================================================

# Categories whose label depends on transaction AMOUNT, not on merchant text.
# DistilBERT only sees the merchant name — it cannot distinguish these from text alone.
# We collapse them to a single canonical label here; _apply_amount_logic() in
# DistilBERTCategorizer re-splits them at inference time using the actual amount.
AMOUNT_DIRECTION_COLLAPSE = {
    "gambling_win":          "gambling",
    "payday_loan_deposit":   "payday_loan",
    "payday_loan_repayment": "payday_loan",
}


def extract_training_pairs(dataset_path: str) -> List[Tuple[str, str]]:
    """
    Extract (merchant_name, category) pairs from the synthetic dataset.
    Each transaction has a 'merchant' field and a 'category' field (ground truth).

    Amount-direction-dependent categories (gambling_win, payday_loan_deposit/repayment)
    are collapsed to their text-distinguishable base class so DistilBERT is not
    asked to make an impossible prediction from merchant name alone.
    """
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path) as f:
        consumers = json.load(f)
    print(f"Loaded {len(consumers)} consumers.")

    pairs = []
    for consumer in consumers:
        for txn in consumer["transactions"]:
            merchant = txn.get("merchant", "").strip()
            category = txn.get("category", "other").strip()
            if not merchant or not category:
                continue
            # Collapse amount-direction classes to their base form
            category = AMOUNT_DIRECTION_COLLAPSE.get(category, category)
            pairs.append((merchant, category))

    print(f"Extracted {len(pairs):,} (merchant, category) pairs.")
    cat_counts = Counter(cat for _, cat in pairs)
    print("\nCategory distribution:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<30s} {count:>8,}")
    print(f"\nNote: gambling_win → gambling, payday variants → payday_loan")
    print(f"      Amount-direction re-splitting happens at inference via _apply_amount_logic()")

    return pairs


def balance_and_sample(
    pairs: List[Tuple[str, str]],
    max_per_class: int = 4000,
    min_per_class: int = 200,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """
    Cap each category at max_per_class examples and ensure at least
    min_per_class (upsample if needed). This prevents dominant classes
    from overwhelming rare but important ones (fees, gambling, payday loans).
    """
    rng = random.Random(seed)
    by_category: Dict[str, List[str]] = {}
    for merchant, category in pairs:
        by_category.setdefault(category, []).append(merchant)

    balanced = []
    for cat, merchants in by_category.items():
        if len(merchants) >= max_per_class:
            sampled = rng.sample(merchants, max_per_class)
        elif len(merchants) < min_per_class:
            # Upsample with replacement to reach min_per_class
            sampled = [rng.choice(merchants) for _ in range(min_per_class)]
        else:
            sampled = merchants
        balanced.extend((m, cat) for m in sampled)

    rng.shuffle(balanced)
    print(f"\nAfter balancing: {len(balanced):,} examples "
          f"({len(by_category)} categories, max {max_per_class}/class)")
    return balanced


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class TransactionDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 64):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ============================================================================
# TRAINING
# ============================================================================

def train_distilbert(
    pairs: List[Tuple[str, str]],
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 3e-5,
    max_length: int = 64,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device selection: MPS (Apple Silicon) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS acceleration.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU (training may be slow).")

    # Label encoding
    le = LabelEncoder()
    texts = [p[0] for p in pairs]
    raw_labels = [p[1] for p in pairs]
    labels = le.fit_transform(raw_labels).tolist()
    n_classes = len(le.classes_)
    print(f"\n{n_classes} label classes: {list(le.classes_)}")

    # Train/val/test split (80/10/10)
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )
    print(f"Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    # Tokenizer and model
    print("\nLoading distilbert-base-uncased...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=n_classes
    )
    model = model.to(device)

    # Datasets and loaders
    train_ds = TransactionDataset(X_train, y_train, tokenizer, max_length)
    val_ds = TransactionDataset(X_val, y_val, tokenizer, max_length)
    test_ds = TransactionDataset(X_test, y_test, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    print(f"\nTraining for {epochs} epochs ({total_steps} steps)...")
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=labels_batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)

        # ---- Validate ----
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(batch["labels"].numpy())

        val_acc = accuracy_score(val_true, val_preds)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch}/{epochs} | loss={avg_loss:.4f} | "
              f"val_acc={val_acc:.4f} | {elapsed:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            # Save best model
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    print(f"\nBest validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")

    # ---- Test evaluation on best model ----
    print("\nLoading best checkpoint for test evaluation...")
    model = DistilBertForSequenceClassification.from_pretrained(output_dir)
    model = model.to(device)
    model.eval()

    test_preds, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(batch["labels"].numpy())

    test_acc = accuracy_score(test_true, test_preds)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print("\nPer-category report:")
    print(classification_report(
        test_true, test_preds,
        target_names=le.classes_,
        digits=3,
    ))

    # Save label encoder classes
    label_map = {str(i): cls for i, cls in enumerate(le.classes_)}
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    # Save training summary
    summary = {
        "model": "distilbert-base-uncased",
        "n_classes": n_classes,
        "classes": list(le.classes_),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "best_val_accuracy": round(best_val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
    }
    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nModel saved to {output_dir}/")
    print(f"  model weights:     pytorch_model.bin")
    print(f"  tokenizer:         tokenizer files")
    print(f"  label map:         label_map.json")
    print(f"  training summary:  training_summary.json")

    return test_acc, le.classes_


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for transaction categorization"
    )
    parser.add_argument("--dataset", required=True,
                        help="Path to flowscore_dataset.json")
    parser.add_argument("--output", default="models/distilbert_categorizer/",
                        help="Output directory for model + tokenizer")
    parser.add_argument("--max_per_class", type=int, default=4000,
                        help="Max examples per category (default: 4000)")
    parser.add_argument("--min_per_class", type=int, default=300,
                        help="Min examples per category, upsample if needed (default: 300)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Learning rate (default: 3e-5)")
    parser.add_argument("--max_length", type=int, default=64,
                        help="Max token length for merchant strings (default: 64)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Extract and balance training data
    pairs = extract_training_pairs(args.dataset)
    pairs = balance_and_sample(pairs, args.max_per_class, args.min_per_class, args.seed)

    # Train
    test_acc, classes = train_distilbert(
        pairs,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        seed=args.seed,
    )

    print(f"\n{'='*60}")
    print(f"DISTILBERT TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Test accuracy:  {test_acc:.4f}")
    print(f"  Model saved to: {args.output}")
    print(f"\nNext step — use DistilBERT in the categorizer:")
    print(f"  python src/categorizer.py --input data/flowscore_dataset.json \\")
    print(f"      --output data/results_distilbert.json \\")
    print(f"      --distilbert {args.output} \\")
    print(f"      --noise medium")


if __name__ == "__main__":
    main()
