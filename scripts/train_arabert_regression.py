"""
Track A: Fine-tune AraBERT v2 as a regression model on Jarḥ wa Taʿdīl
biographical text to predict continuous narrator reliability scores (0.0–1.0).

This implements the "Track A" approach described in §4.7.1 of the thesis.

Data sources:
  1. Encyclopedia entries with extracted grade terms → mapped to GRADE_SCORE_MAP
  2. Narrator grades (matched_name text) → mapped via GRADE_SCORE_MAP

Output: trained model, evaluation metrics, comparison vs Track B (keyword lookup).
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.stdout.reconfigure(encoding="utf-8")

# ── Project paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "track_a"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_ROOT / "output"
ENCYCLOPEDIA_PATH = DATA_DIR / "shamela" / "narrator_encyclopedia.json"
NARRATOR_GRADES_PATH = DATA_DIR / "stage2" / "narrator_grades.json"
NODE_NAMES_PATH = DATA_DIR / "stage2" / "node_names.json"
STAGE2_GRAPH_PATH = DATA_DIR / "stage2" / "pyg_graph_large_graded.npz"

# ── AraBERT config ──
MODEL_NAME = "aubmindlab/bert-base-arabertv2"
MAX_LENGTH = 256             # Full context for GPU
BATCH_SIZE = 16              # Fits comfortably in 8GB VRAM
GRADIENT_ACCUMULATION = 1    # No accumulation needed on GPU
NUM_EPOCHS = 10              # Full training
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15  # from training portion

# ── GRADE_SCORE_MAP (from src/stage2/config.py) ──
GRADE_SCORE_MAP: dict[str, float] = {
    "صحابي": 1.0,
    "ثقة ثبت": 0.95, "ثقة حافظ": 0.95, "ثقة متقن": 0.95, "إمام": 0.95,
    "حافظ": 0.90, "ثبت": 0.90, "متقن": 0.90,
    "ثقة": 0.85, "ثقات": 0.85, "مأمون": 0.85,
    "صدوق": 0.70, "لا بأس به": 0.65, "ليس به بأس": 0.65, "محله الصدق": 0.65,
    "صدوق يهم": 0.55, "صدوق سيء الحفظ": 0.50, "سيء الحفظ": 0.40,
    "مقبول": 0.40,
    "مستور": 0.25, "مجهول الحال": 0.20,
    "ضعيف": 0.15, "ضعيف الحديث": 0.15, "لين": 0.15, "لين الحديث": 0.15,
    "ليس بالقوي": 0.15, "فيه ضعف": 0.15,
    "منكر الحديث": 0.10, "مجهول": 0.10, "مجهول العين": 0.05,
    "متروك": 0.05, "متروك الحديث": 0.05, "ساقط": 0.05, "مختلط": 0.05,
    "متهم": 0.0, "كذاب": 0.0, "وضاع": 0.0, "دجال": 0.0,
}

DEFAULT_GRADE_SCORE = 0.30


def normalize_arabic(text: str) -> str:
    """Light normalization: strip diacritics, collapse alif variants."""
    text = re.sub(r"[ً-ٌٍ-َّْٰ]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه").replace("ى", "ي")
    return text.strip()


def map_grade_to_score(grade_text: str) -> Optional[float]:
    """Map a grade term to a continuous score using GRADE_SCORE_MAP.

    Tries longest-match-first to handle compound terms (e.g. 'ثقة ثبت').
    Returns None if no term matches.
    """
    if not grade_text:
        return None
    # Try longest match first
    for term in sorted(GRADE_SCORE_MAP, key=len, reverse=True):
        if term in grade_text:
            return GRADE_SCORE_MAP[term]
    return None


# ────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────

@dataclass
class TrainingExample:
    text: str
    score: float
    source: str = ""  # "encyclopedia" or "grades_json"
    narrator_name: str = ""


class GradeRegressionDataset(Dataset):
    """PyTorch Dataset for grade → score regression."""

    def __init__(self, examples: list[TrainingExample], tokenizer, max_length: int = MAX_LENGTH):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        encoded = self.tokenizer(
            ex.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex.score, dtype=torch.float),
        }


# ────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────

def load_encyclopedia_data() -> list[TrainingExample]:
    """Load training examples from the NEW narrator encyclopedia (15K entries).

    Two sources:
      - Taqrib entries: have explicit grade term embedded, source='taqrib'
      - Tahdhib entries: have extracted grades list from biographical text
    Both map to GRADE_SCORE_MAP scores.
    """
    examples: list[TrainingExample] = []
    with open(ENCYCLOPEDIA_PATH, encoding="utf-8") as f:
        enc: list[dict] = json.load(f)

    for entry in enc:
        text = entry.get("full_text", "")
        grades_list = entry.get("grades", [])
        source = entry.get("source", "tahdhib")

        if not text or len(text) < 30:
            continue

        # Taqrib: grades list has 1 entry (the grade term)
        # Tahdhib: grades list has 0+ extracted terms
        if not grades_list:
            continue

        # Map each grade term to score
        scores = [map_grade_to_score(g) for g in grades_list]
        scores = [s for s in scores if s is not None]
        if not scores:
            continue

        avg_score = float(np.mean(scores))
        examples.append(TrainingExample(
            text=text[:2000],
            score=avg_score,
            source=f"encyclopedia:{source}",
            narrator_name=entry.get("name", ""),
        ))

    return examples


def _score_to_bin(score: float) -> str:
    """Bin a continuous score into a grade category."""
    if score >= 0.85:
        return "thiqah"
    elif score >= 0.60:
        return "saduq"
    elif score >= 0.35:
        return "maqbul"
    elif score >= 0.15:
        return "daif"
    else:
        return "matruk"


_BIN_ORDER = ["matruk", "daif", "maqbul", "saduq", "thiqah"]


def prepare_datasets(
    tokenizer,
    test_size: float = TEST_SPLIT,
    val_size: float = VAL_SPLIT,
    stratify_by_grade: bool = True,
) -> tuple[GradeRegressionDataset, GradeRegressionDataset, GradeRegressionDataset, dict]:
    """Load all data, split, and return Datasets.

    When stratify_by_grade=True, samples balanced across grade categories
    so the model cannot simply predict the majority class.
    """
    # Load from encyclopedia (primary source, 10.9K with grades)
    combined = load_encyclopedia_data()

    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(combined)

    if stratify_by_grade:
        # Bin each example by grade category
        bins: dict[str, list[TrainingExample]] = {b: [] for b in _BIN_ORDER}
        for ex in combined:
            bin_name = _score_to_bin(ex.score)
            if bin_name in bins:
                bins[bin_name].append(ex)

        # Log pre-balance distribution
        pre_counts = {b: len(bins[b]) for b in _BIN_ORDER}
        print(f"\n  Pre-balance grade distribution:")
        for b in _BIN_ORDER:
            bar = "#" * min(pre_counts[b], 60)
            print(f"    {b:<8s}: {pre_counts[b]:>4d} {bar}")

        # Subsample all bins to the size of the smallest non-empty bin
        min_count = min(v for v in pre_counts.values() if v > 0)
        for b in _BIN_ORDER:
            if len(bins[b]) > min_count:
                rng.shuffle(bins[b])
                bins[b] = bins[b][:min_count]

        # Flatten back
        combined = []
        for b in _BIN_ORDER:
            combined.extend(bins[b])
        rng.shuffle(combined)

        balanced_counts = {b: len(bins[b]) for b in _BIN_ORDER}
        print(f"\n  Post-balance grade distribution (all → {min_count}):")
        for b in _BIN_ORDER:
            bar = "#" * min(balanced_counts[b], 60)
            print(f"    {b:<8s}: {balanced_counts[b]:>4d} {bar}")

    # Split into train / val / test, preserving stratification
    n = len(combined)
    n_test = max(1, int(n * test_size))
    n_val = max(1, int(n * val_size))

    test_examples = combined[:n_test]
    val_examples = combined[n_test:n_test + n_val]
    train_examples = combined[n_test + n_val:]

    # Score distribution stats
    train_scores = [ex.score for ex in train_examples]

    stats = {
        "total": n,
        "train": len(train_examples),
        "val": len(val_examples),
        "test": len(test_examples),
        "stratified": stratify_by_grade,
        "train_score_mean": float(np.mean(train_scores)),
        "train_score_std": float(np.std(train_scores)),
    }

    train_ds = GradeRegressionDataset(train_examples, tokenizer)
    val_ds = GradeRegressionDataset(val_examples, tokenizer)
    test_ds = GradeRegressionDataset(test_examples, tokenizer)

    return train_ds, val_ds, test_ds, stats


# ────────────────────────────────────────────────────────────
# Manual training loop (no accelerate / Trainer dependency)
# ────────────────────────────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict:
    """Collate batch of tokenized examples for manual training."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.float)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train_epoch(model, dataloader, optimizer, scheduler, device, scaler=None):
    """Single training epoch with optional mixed precision. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_steps = 0

    for step, batch in enumerate(dataloader, 1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                loss = nn.MSELoss()(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            loss = nn.MSELoss()(logits, labels)
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        n_steps += 1

    return total_loss / max(n_steps, 1)


@torch.no_grad()
def evaluate_epoch(model, dataloader, device) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate model on dataloader. Returns (y_true, y_pred) arrays."""
    model.eval()
    all_preds, all_true = [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.sigmoid(outputs.logits).squeeze(-1)
        all_preds.append(preds.cpu().numpy())
        all_true.append(labels.cpu().numpy())
    return np.concatenate(all_true), np.concatenate(all_preds)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Compute all regression and classification metrics."""
    # Clip predictions to [0, 1]
    y_pred = np.clip(y_pred, 0.0, 1.0)

    # Regression metrics
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    spear_r, spear_p = spearmanr(y_true, y_pred)
    pear_r, pear_p = pearsonr(y_true, y_pred)

    # Classification: accepted (>= 0.65) vs rejected
    y_true_bin = (y_true >= 0.65).astype(int)
    y_pred_bin = (y_pred >= 0.65).astype(int)
    bin_acc = float(accuracy_score(y_true_bin, y_pred_bin))
    bin_f1 = float(f1_score(y_true_bin, y_pred_bin))

    # Ordinal categories → Cohen's Kappa
    def to_cat(s):
        if s >= 0.85: return 4  # thiqah
        elif s >= 0.60: return 3  # saduq
        elif s >= 0.35: return 2  # maqbul
        elif s >= 0.15: return 1  # daif
        else: return 0  # matruk

    y_true_cat = np.array([to_cat(s) for s in y_true])
    y_pred_cat = np.array([to_cat(s) for s in y_pred])
    kappa = float(cohen_kappa_score(y_true_cat, y_pred_cat, weights="quadratic"))

    # Per-category MAE
    cat_labels = {0: "matruk", 1: "daif", 2: "maqbul", 3: "saduq", 4: "thiqah"}
    cat_mae = {}
    for cat_id, label in cat_labels.items():
        mask = y_true_cat == cat_id
        if mask.sum() >= 3:
            cat_mae[label] = float(np.abs(y_pred[mask] - y_true[mask]).mean())
        else:
            cat_mae[label] = None

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman_r": float(spear_r),
        "spearman_p": float(spear_p),
        "pearson_r": float(pear_r),
        "pearson_p": float(pear_p),
        "binary_accuracy": bin_acc,
        "binary_f1": bin_f1,
        "cohen_kappa": kappa,
        "cat_mae": cat_mae,
        "y_true_mean": float(y_true.mean()),
        "y_pred_mean": float(y_pred.mean()),
        "y_true_std": float(y_true.std()),
        "y_pred_std": float(y_pred.std()),
        "n": len(y_true),
    }


# ────────────────────────────────────────────────────────────
# Score all narrators in the graph with trained model
# ────────────────────────────────────────────────────────────

def score_all_narrators(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, float]:
    """Use trained model to score every narrator in the Knowledge Graph.

    Looks up each node name in the encyclopedia for biographical text.
    Falls back to the node name itself if no encyclopedia entry found.
    """
    # Load node names
    with open(NODE_NAMES_PATH, encoding="utf-8") as f:
        id_to_name: dict[str, str] = json.load(f)

    # Load alias index for name → entry mapping
    alias_path = Path("output/shamela/alias_index.json")
    if alias_path.exists():
        with open(alias_path, encoding="utf-8") as f:
            alias_index: dict[str, list[int]] = json.load(f)
    else:
        alias_index = {}

    # Load encyclopedia for text lookups (by entry_id)
    with open(ENCYCLOPEDIA_PATH, encoding="utf-8") as f:
        enc: list[dict] = json.load(f)

    # Build entry_id → full_text lookup
    entry_text: dict[int, str] = {}
    for entry in enc:
        eid = entry.get("entry_id")
        text = entry.get("full_text", "")
        if eid and text and len(text) > 30:
            entry_text[eid] = text[:2000]

    # Build O(1) name→text lookup from alias index
    name_to_text: dict[str, str] = {}
    for alias_key, entry_ids in alias_index.items():
        eid = entry_ids[0]
        if eid in entry_text:
            name_to_text[alias_key] = entry_text[eid]

    print(f"  Name→text lookup: {len(name_to_text):,} entries")

    # Score each node
    node_ids = sorted(id_to_name.keys(), key=int)
    texts: list[str] = []
    node_list: list[tuple[str, str]] = []

    matched = 0
    for nid in node_ids:
        name = id_to_name[nid]
        nkey = normalize_arabic(name)
        bio_text = name_to_text.get(nkey, name)
        if bio_text != name:
            matched += 1
        texts.append(bio_text)
        node_list.append((nid, name))

    print(f"  KG narrators with exact alias bio text: {matched:,}/{len(node_ids):,} ({matched/len(node_ids)*100:.1f}%)")

    # Batch inference
    scores: dict[str, float] = {}
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Scoring narrators"):
            batch_texts = texts[i:i + batch_size]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)

            outputs = model(**encoded)
            batch_scores = torch.sigmoid(outputs.logits).squeeze(-1).cpu().numpy()

            for j, (nid, name) in enumerate(node_list[i:i + batch_size]):
                scores[nid] = float(batch_scores[j])

    return scores


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  TRACK A: AraBERT v2 Regression for Narrator Reliability Scoring")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        scaler = torch.amp.GradScaler("cuda")
    else:
        torch.set_num_threads(min(16, (torch.get_num_threads() or 4)))
        print(f"  CPU threads: {torch.get_num_threads()}")
        scaler = None

    # ── Load tokenizer and model ──
    print(f"\n[1/5] Loading AraBERT v2: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        problem_type="regression",
    ).to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,} (all trainable)")

    # ── Prepare datasets ──
    print(f"\n[2/5] Loading and preparing training data...")
    train_ds, val_ds, test_ds, stats = prepare_datasets(tokenizer)
    print(f"  Total examples: {stats['total']}")
    print(f"  Train: {stats['train']}, Val: {stats['val']}, Test: {stats['test']}")
    print(f"  Train score distribution: mean={stats['train_score_mean']:.3f}, "
          f"std={stats['train_score_std']:.3f}")

    # Score distribution from stats (already computed in prepare_datasets)
    print(f"\n  Training score histogram:")
    bins = [0.0, 0.15, 0.35, 0.60, 0.85, 1.01]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        print(f"    [{lo:.2f}, {hi:.2f}): (see pre-balance above)")

    # ── Training (manual PyTorch loop) ──
    print(f"\n[3/5] Starting fine-tuning...")
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = NUM_EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6,
    )

    best_val_mae = float("inf")
    best_state = None
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        y_val_true, y_val_pred = evaluate_epoch(model, val_loader, device)
        val_mae = float(mean_absolute_error(y_val_true, y_val_pred))
        val_r2 = float(r2_score(y_val_true, y_val_pred))

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch:>2}/{NUM_EPOCHS}: loss={train_loss:.4f}, "
              f"val_mae={val_mae:.4f}, val_r2={val_r2:.4f}")

    train_elapsed = time.time() - start_time
    print(f"  Training completed in {train_elapsed:.0f}s (best val MAE: {best_val_mae:.4f})")

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    # ── Evaluate on test set ──
    print(f"\n[4/5] Evaluating on held-out test set...")
    y_true, y_pred = evaluate_epoch(model, test_loader, device)

    metrics = compute_regression_metrics(y_true, y_pred)

    print(f"\n  Test Set Results (n={metrics['n']}):")
    print(f"  {'Metric':<25s} {'Value':>10s}")
    print(f"  {'-' * 37}")
    for key in ["mae", "rmse", "r2", "spearman_r", "pearson_r", "cohen_kappa",
                 "binary_accuracy", "binary_f1"]:
        print(f"  {key:<25s} {metrics[key]:>10.4f}")
    print(f"\n  Per-category MAE:")
    for cat, val in metrics["cat_mae"].items():
        print(f"    {cat}: {val:.4f}" if val is not None else f"    {cat}: N/A")

    # ── Save model ──
    print(f"\n[5/5] Saving model and results...")
    model_path = OUTPUT_DIR / "arabert_regression_model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"  Model saved: {model_path}")

    # Save test predictions
    test_df = []
    for i in range(len(y_true)):
        entry = {
            "true_score": float(y_true[i]),
            "predicted_score": float(y_pred[i]),
            "error": float(abs(y_true[i] - y_pred[i])),
        }
        test_df.append(entry)
    with open(OUTPUT_DIR / "test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(test_df, f, indent=2, ensure_ascii=False)

    # Save metrics
    with open(OUTPUT_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  Evaluation: {OUTPUT_DIR / 'evaluation_metrics.json'}")

    # ── Score all narrators ──
    print(f"\n  Scoring all graph narrators...")
    all_scores = score_all_narrators(model, tokenizer, device)
    with open(OUTPUT_DIR / "narrator_scores.json", "w", encoding="utf-8") as f:
        json.dump(all_scores, f, indent=2, ensure_ascii=False)
    print(f"  Scored {len(all_scores)} narrators")
    print(f"  Score range: {min(all_scores.values()):.3f} – {max(all_scores.values()):.3f}")

    # ── Save training stats ──
    stats["train_elapsed_seconds"] = train_elapsed
    stats["model_name"] = MODEL_NAME
    stats["num_epochs"] = NUM_EPOCHS
    with open(OUTPUT_DIR / "training_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'=' * 65}")
    print(f"  TRACK A COMPLETE")
    print(f"{'=' * 65}")
    print(f"\nNext: Run compare_tracks.py to compare Track A vs Track B.")


if __name__ == "__main__":
    main()
