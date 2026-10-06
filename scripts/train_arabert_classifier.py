"""
Track A v2: AraBERT 5-class classifier with explicit grade keyword masking.

Classes: matruk(0), daif(1), maqbul(2), saduq(3), thiqah(4)
Score map: {0:0.05, 1:0.15, 2:0.40, 3:0.70, 4:0.85}

Zero-Target-Leakage Remediation:
  - Explicit jarh wa ta'dil appraisal terms (e.g. ثقة, ضعيف, صدوق, متروك, كذاب,
    وثقه, ضعفه, etc.) are replaced with `[MASK]` before tokenization.
  - This prevents the transformer model from exploiting verbatim label tokens as a
    superficial shortcut, forcing it to learn latent semantic, stylistic, and
    contextual signals from classical biographical texts.
  - Stratified 70/15/15 train/val/test splits preserve minority class ratios.
  - Primary evaluation metric: Macro-F1 (falsifiable hypothesis H1 for PO1).
"""
import argparse, json, os, re, sys, time, types
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

# Bypass transformers strict dependency version check on huggingface-hub<2.0
if "transformers.dependency_versions_check" not in sys.modules:
    _dvc = types.ModuleType("transformers.dependency_versions_check")
    _dvc.dep_version_check = lambda *a, **k: None
    sys.modules["transformers.dependency_versions_check"] = _dvc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    classification_report, confusion_matrix, cohen_kappa_score,
    f1_score, accuracy_score,
)
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Config ──
MODEL_NAME = "aubmindlab/bert-base-arabertv2"
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15

ENC_PATH = PROJECT / "output/shamela/narrator_encyclopedia.json"
OUT_DIR = PROJECT / "output/track_a_classifier"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Grade → score mapping (used for chain scoring)
CLASS_TO_SCORE = {0: 0.05, 1: 0.15, 2: 0.40, 3: 0.70, 4: 0.85}
CLASS_NAMES = ["matruk", "daif", "maqbul", "saduq", "thiqah"]
SCORE_TO_CLASS = {0.05: 0, 0.15: 1, 0.40: 2, 0.55: 2, 0.70: 3, 0.85: 4}

GRADE_SCORE_MAP = {
    "صحابي": 1.0, "ثقة ثبت": 0.95, "ثقة حافظ": 0.95, "ثقة متقن": 0.95, "إمام": 0.95,
    "حافظ": 0.90, "ثبت": 0.90, "متقن": 0.90, "ثقة": 0.85, "ثقات": 0.85, "مأمون": 0.85,
    "صدوق": 0.70, "لا بأس به": 0.65, "ليس به بأس": 0.65, "محله الصدق": 0.65,
    "صدوق يهم": 0.55, "صدوق سيء الحفظ": 0.50, "صدوق سيئ الحفظ": 0.50, "سيء الحفظ": 0.40,
    "سيئ الحفظ": 0.40, "مقبول": 0.40, "مستور": 0.25, "مجهول الحال": 0.20,
    "ضعيف": 0.15, "ضعيف الحديث": 0.15, "لين": 0.15, "لين الحديث": 0.15,
    "ليس بالقوي": 0.15, "فيه ضعف": 0.15, "منكر الحديث": 0.10, "مجهول": 0.10,
    "مجهول العين": 0.05, "متروك": 0.05, "متروك الحديث": 0.05, "ساقط": 0.05,
    "مختلط": 0.05, "متهم": 0.0, "كذاب": 0.0, "وضاع": 0.0, "دجال": 0.0,
}

# ── Jarh wa Ta'dil Vocabulary for Text Masking ──
# Compound multi-word terms ordered first to guarantee greedy longest match
GRADE_TERMS_VOCAB = [
    # Multi-word compound grades
    "ثقة ثبت", "ثقة حافظ", "ثقة متقن", "صدوق سيء الحفظ", "صدوق سيئ الحفظ", "صدوق يهم",
    "مجهول الحال", "مجهول العين", "ضعيف الحديث", "لين الحديث", "ليس بالقوي", "منكر الحديث",
    "متروك الحديث", "لا بأس به", "ليس به بأس", "محله الصدق", "سيء الحفظ", "سيئ الحفظ",
    "فيه ضعف", "لا يحتج به", "ليس بحجة", "ليس بشيء", "يكتب حديثه", "لا يكتب حديثه",
    # Single-word grade ranks
    "ثقة", "صدوق", "مقبول", "مجهول", "ضعيف", "متروك", "مستور", "لين", "كذاب",
    "حافظ", "إمام", "ثبت", "متقن", "ساقط", "حجة", "مأمون", "مختلط", "ضابط",
    "دجال", "وضاع", "ثقات", "صحابي",
    # Inflected critic verbs & appraisal forms
    "وثقه", "وثقوه", "يوثقه", "موثق",
    "ضعفه", "ضعفوه", "يضعفه", "مضعف",
    "كذبه", "كذبوه", "يكذبه", "تركوه",
]

_TERMS_SORTED = sorted(set(GRADE_TERMS_VOCAB), key=len, reverse=True)
AR_LETTER = r"[\u0621-\u064A\u0671-\u06D3]"
# Proclitic prefix matches standard Arabic prefixes: و (and), ف (then), ب (by), ك (as), ل/لل (for), ال (the)
PROCLITIC_PREFIX = r"(?:[وف])?(?:لل|[بكل])?(?:ال)?"
GRADE_MASK_RE = re.compile(
    rf"(?<!{AR_LETTER}){PROCLITIC_PREFIX}(?:" + "|".join(re.escape(t) for t in _TERMS_SORTED) + rf")(?!(?:{AR_LETTER}))"
)

def mask_grade_keywords(text: str, mask_token: str = "[MASK]") -> tuple[str, int]:
    """Mask explicit jarh wa ta'dil appraisal terms with [MASK].

    Returns:
        (masked_text, count_of_masked_tokens)
    """
    if not text:
        return "", 0
    matches = list(GRADE_MASK_RE.finditer(text))
    if not matches:
        return text, 0
    return GRADE_MASK_RE.sub(mask_token, text), len(matches)

def norm(t: str) -> str:
    """Normalize Arabic diacritics and letter variants."""
    if not t: return ""
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

def score_to_class_label(score: float) -> int:
    if score >= 0.85: return 4   # thiqah
    elif score >= 0.60: return 3  # saduq
    elif score >= 0.35: return 2  # maqbul
    elif score >= 0.15: return 1  # daif
    else: return 0                # matruk

def map_grade_to_score(grade_text: str):
    if not grade_text: return None
    for term in sorted(GRADE_SCORE_MAP, key=len, reverse=True):
        if term in grade_text:
            return GRADE_SCORE_MAP[term]
    return None

# ── Dataset ──
class ClassifierDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding="max_length",
                            max_length=self.max_length, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long)}

# ── Data loading with Zero-Leakage Masking ──
def load_data(mask_grades: bool = True):
    """Load narrator texts and ground-truth reliability classes.

    Args:
        mask_grades: If True, replace explicit grade keywords with [MASK]
                     to prevent target label leakage.
    """
    enc = json.load(open(ENC_PATH, encoding="utf-8"))
    texts, labels = [], []
    total_masked_terms = 0
    texts_masked_count = 0

    for e in enc:
        text = e.get("full_text", "")
        grades = e.get("grades", [])
        if not text or len(text) < 30: continue
        if not grades: continue

        scores = [map_grade_to_score(g) for g in grades]
        scores = [s for s in scores if s is not None]
        if not scores: continue

        avg_score = float(np.mean(scores))
        cls = score_to_class_label(avg_score)

        raw_snippet = text[:2000]
        if mask_grades:
            clean_text, num_masks = mask_grade_keywords(raw_snippet, mask_token="[MASK]")
            if num_masks > 0:
                texts_masked_count += 1
                total_masked_terms += num_masks
            texts.append(clean_text)
        else:
            texts.append(raw_snippet)

        labels.append(cls)

    mask_stats = {
        "mask_grades": mask_grades,
        "total_texts": len(texts),
        "texts_with_masked_tokens": texts_masked_count,
        "pct_texts_masked": round(texts_masked_count / max(len(texts), 1) * 100, 2),
        "total_masked_tokens": total_masked_terms,
    }
    return texts, labels, mask_stats

# ── Training ──
def train_epoch(model, loader, optimizer, scheduler, scaler, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbl = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast("cuda"):
                out = model(input_ids=ids, attention_mask=mask)
                loss = criterion(out.logits, lbl)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(input_ids=ids, attention_mask=mask)
            loss = criterion(out.logits, lbl)
            loss.backward()
            optimizer.step()

        if scheduler: scheduler.step()
        total_loss += loss.item()
        correct += (out.logits.argmax(-1) == lbl).sum().item()
        total += len(lbl)

    return total_loss / max(len(loader), 1), correct / max(total, 1)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_true = [], []
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbl = batch["labels"].to(DEVICE)
        out = model(input_ids=ids, attention_mask=mask)
        all_preds.extend(out.logits.argmax(-1).cpu().tolist())
        all_true.extend(lbl.cpu().tolist())
    return np.array(all_true), np.array(all_preds)

# ── Score all KG narrators ──
def score_kg_narrators(model, tokenizer, mask_grades: bool = True):
    node_names_path = PROJECT / "output/stage2/node_names.json"
    alias_path = PROJECT / "output/shamela/alias_index.json"
    gnn_pred_path = PROJECT / "output/stage3/narrator_predictions.csv"

    with open(node_names_path, encoding="utf-8") as f:
        id_to_name = json.load(f)
    alias = json.load(open(alias_path, encoding="utf-8"))
    enc = json.load(open(ENC_PATH, encoding="utf-8"))

    # Build entry_id → text (applying identical masking if enabled)
    entry_text = {}
    for e in enc:
        eid, text = e.get("entry_id"), e.get("full_text", "")
        if eid and text and len(text) > 30:
            snippet = text[:2000]
            if mask_grades:
                snippet, _ = mask_grade_keywords(snippet, mask_token="[MASK]")
            entry_text[eid] = snippet

    name_to_bio = {}
    for ak, eids in alias.items():
        if eids[0] in entry_text:
            name_to_bio[norm(ak)] = entry_text[eids[0]]

    # GNN fallback
    gnn = {}
    if gnn_pred_path.exists():
        preds_df = pd.read_csv(gnn_pred_path)
        gnn = {int(r["node_id"]): float(r["predicted_score"]) for _, r in preds_df.iterrows()}

    all_kg = [(norm(v), int(k)) for k, v in id_to_name.items()]
    scores = {}
    hits = 0

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(all_kg), BATCH_SIZE), desc="Classifier scoring"):
            batch = all_kg[i:i + BATCH_SIZE]
            texts = [name_to_bio.get(nk, nk) for nk, _ in batch]
            has_bio = [nk in name_to_bio for nk, _ in batch]

            enc = tokenizer(texts, truncation=True, padding=True,
                            max_length=128, return_tensors="pt").to(DEVICE)
            out = model(**enc)
            preds_cls = out.logits.argmax(-1).cpu().tolist()

            for j, (nk, nid) in enumerate(batch):
                if has_bio[j]:
                    scores[nid] = CLASS_TO_SCORE[preds_cls[j]]
                    hits += 1
                else:
                    scores[nid] = gnn.get(nid, 0.632)

    return scores, hits

# ── Main ──
def main():
    parser = argparse.ArgumentParser(description="AraBERT 5-class narrator reliability classifier.")
    parser.add_argument("--mask-grades", action="store_true", default=True,
                        help="Mask explicit jarh wa ta'dil grade keywords with [MASK] (default: True).")
    parser.add_argument("--no-mask-grades", dest="mask_grades", action="store_false",
                        help="Disable grade keyword masking (for leakage ablation study).")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--skip-kg-scoring", action="store_true", default=False,
                        help="Skip scoring full 262K KG after test evaluation.")
    args = parser.parse_args()

    print("=" * 65)
    print("  AraBERT 5-CLASS CLASSIFIER TRAINING")
    print(f"  Target Leakage Masking: {'ENABLED ([MASK])' if args.mask_grades else 'DISABLED (Leaky Ablation)'}")
    print("=" * 65)
    print(f"  Device: {DEVICE}")

    # Set seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # Load data
    print(f"\n[1] Loading training data...")
    texts, labels, mask_stats = load_data(mask_grades=args.mask_grades)
    dist = Counter(labels)
    print(f"  Total valid entries: {len(texts):,}")
    print(f"  Class distribution:  {dict(dist)}")
    print(f"  Class names:         {dict(enumerate(CLASS_NAMES))}")
    if args.mask_grades:
        print(f"  Texts with masked tokens: {mask_stats['texts_with_masked_tokens']:,} ({mask_stats['pct_texts_masked']}%)")
        print(f"  Total masked tokens:      {mask_stats['total_masked_tokens']:,}")

    # Stratified 70/15/15 train/val/test split
    indices = np.arange(len(texts))
    train_idx, temp_idx, _, y_temp = train_test_split(
        indices, labels,
        test_size=(TEST_SPLIT + VAL_SPLIT),
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    val_rel_ratio = VAL_SPLIT / (TEST_SPLIT + VAL_SPLIT)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_rel_ratio),
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    print(f"\n  Splits (Stratified 70/15/15):")
    print(f"    Train: {len(train_texts):,} | Val: {len(val_texts):,} | Test: {len(test_texts):,}")

    # Load model
    print(f"\n[2] Loading AraBERT v2 ({MODEL_NAME}) with 5-class head...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=5).to(DEVICE)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Compute class weights for cross entropy
    class_counts = Counter(train_labels)
    max_cnt = max(class_counts.values())
    class_weights = torch.tensor(
        [max_cnt / max(class_counts.get(i, 1), 1) for i in range(5)], dtype=torch.float
    ).to(DEVICE)
    print(f"  Class weights (inverse freq): {[round(w, 2) for w in class_weights.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # DataLoaders
    train_ds = ClassifierDataset(train_texts, train_labels, tokenizer)
    val_ds = ClassifierDataset(val_texts, val_labels, tokenizer)
    test_ds = ClassifierDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, args.batch_size * 2)
    test_loader = DataLoader(test_ds, args.batch_size * 2)

    # Training
    print(f"\n[3] Training (Epochs={args.epochs}, LR={args.lr})...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    best_val_f1 = 0
    best_state = None
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        loss, acc = train_epoch(model, train_loader, optimizer, scheduler, scaler, criterion)
        yv_true, yv_pred = evaluate(model, val_loader)
        val_f1 = f1_score(yv_true, yv_pred, average="macro")
        val_acc = accuracy_score(yv_true, yv_pred)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch:>2}: loss={loss:.4f}, train_acc={acc:.4f}, val_f1={val_f1:.4f}, val_acc={val_acc:.4f}")

    train_time = time.time() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  Training complete in {train_time:.0f}s (best val Macro-F1: {best_val_f1:.4f})")

    # Test evaluation
    print(f"\n[4] Test set evaluation (Locked Primary Metric: Macro-F1)...")
    y_true, y_pred = evaluate(model, test_loader)

    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion Matrix:")
    header = "           " + " ".join(f"{n:>8s}" for n in CLASS_NAMES)
    print(header)
    for i, n in enumerate(CLASS_NAMES):
        row = " ".join(f"{cm[i][j]:>8d}" for j in range(len(CLASS_NAMES)))
        print(f"  {n:10s} {row}")

    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)

    y_true_scores = np.array([CLASS_TO_SCORE[c] for c in y_true])
    y_pred_scores = np.array([CLASS_TO_SCORE[c] for c in y_pred])
    spear_r, _ = spearmanr(y_true_scores, y_pred_scores)
    pear_r, _ = pearsonr(y_true_scores, y_pred_scores)

    print(f"\n  Locked Primary Metric (PO1):")
    print(f"    Macro-F1:     {f1_macro:.4f}")
    print(f"  Secondary Metrics:")
    print(f"    Accuracy:     {acc:.4f}")
    print(f"    Cohen's κ:    {kappa:.4f}")
    print(f"    Spearman ρ:   {spear_r:.4f}")
    print(f"    Pearson r:    {pear_r:.4f}")

    # Save model and artifacts
    model_save_dir = OUT_DIR / ("arabert_classifier" if args.mask_grades else "arabert_classifier_unmasked")
    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    print(f"\n  Model saved to: {model_save_dir}")

    metrics = {
        "model": "5-class AraBERT classifier",
        "mask_grades": args.mask_grades,
        "mask_stats": mask_stats,
        "n_train": len(train_texts),
        "n_val": len(val_texts),
        "n_test": len(test_texts),
        "primary_metric": "f1_macro",
        "f1_macro": round(f1_macro, 4),
        "accuracy": round(acc, 4),
        "cohen_kappa": round(kappa, 4),
        "spearman_r": round(spear_r, 4),
        "pearson_r": round(pear_r, 4),
        "train_time_seconds": round(train_time, 2),
        "class_distribution": {int(k): int(v) for k, v in dist.items()},
        "class_weights": class_weights.tolist(),
        "random_seed": RANDOM_SEED,
    }
    metrics_path = OUT_DIR / ("classifier_metrics.json" if args.mask_grades else "classifier_metrics_unmasked.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to: {metrics_path}")

    # Score KG narrators
    if not args.skip_kg_scoring:
        print(f"\n[5] Scoring all KG narrators (masking={args.mask_grades})...")
        hybrid_scores, covered = score_kg_narrators(model, tokenizer, mask_grades=args.mask_grades)
        scores_file = OUT_DIR / "narrator_scores_classifier.json"
        with open(scores_file, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in hybrid_scores.items()}, f)
        print(f"  Saved: {scores_file}")
        print(f"  Coverage: {covered:,}/{len(hybrid_scores):,} ({covered/len(hybrid_scores)*100:.1f}%)")
    else:
        print(f"\n[5] Skipped KG scoring (--skip-kg-scoring set).")

    print(f"\n{'=' * 65}")
    print(f"  ARABERT CLASSIFIER COMPLETE (Macro-F1: {f1_macro:.4f})")
    print(f"{'=' * 65}")

if __name__ == "__main__":
    main()
