"""
Track A v2: AraBERT 5-class classifier instead of regression.

Classes: matruk(0), daif(1), maqbul(2), saduq(3), thiqah(4)
Score map: {0:0.05, 1:0.15, 2:0.40, 3:0.70, 4:0.85}

This forces the model to place narrators in discrete reliability buckets,
preventing the mean-collapse problem of regression-based scoring.
"""
import json, re, sys, time
from pathlib import Path
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, f1_score, accuracy_score
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

PROJECT = Path(__file__).resolve().parent.parent
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
    "صدوق يهم": 0.55, "صدوق سيء الحفظ": 0.50, "سيء الحفظ": 0.40, "مقبول": 0.40,
    "مستور": 0.25, "مجهول الحال": 0.20, "ضعيف": 0.15, "ضعيف الحديث": 0.15,
    "لين": 0.15, "لين الحديث": 0.15, "ليس بالقوي": 0.15, "فيه ضعف": 0.15,
    "منكر الحديث": 0.10, "مجهول": 0.10, "مجهول العين": 0.05,
    "متروك": 0.05, "متروك الحديث": 0.05, "ساقط": 0.05, "مختلط": 0.05,
    "متهم": 0.0, "كذاب": 0.0, "وضاع": 0.0, "دجال": 0.0,
}

def norm(t):
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

# ── Data loading ──
def load_data():
    enc = json.load(open(ENC_PATH, encoding="utf-8"))
    texts, labels = [], []

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
        texts.append(text[:2000])
        labels.append(cls)

    return texts, labels

# ── Training ──
def train_epoch(model, loader, optimizer, scheduler, scaler):
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
                loss = nn.CrossEntropyLoss()(out.logits, lbl)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(input_ids=ids, attention_mask=mask)
            loss = nn.CrossEntropyLoss()(out.logits, lbl)
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
def score_kg_narrators(model, tokenizer):
    node_names_path = PROJECT / "output/stage2/node_names.json"
    alias_path = PROJECT / "output/shamela/alias_index.json"
    gnn_pred_path = PROJECT / "output/stage3/narrator_predictions.csv"

    with open(node_names_path, encoding="utf-8") as f:
        id_to_name = json.load(f)
    alias = json.load(open(alias_path, encoding="utf-8"))
    enc = json.load(open(ENC_PATH, encoding="utf-8"))

    # Build entry_id → text
    entry_text = {}
    for e in enc:
        eid, text = e.get("entry_id"), e.get("full_text", "")
        if eid and text and len(text) > 30: entry_text[eid] = text[:2000]

    name_to_bio = {}
    for ak, eids in alias.items():
        if eids[0] in entry_text: name_to_bio[norm(ak)] = entry_text[eids[0]]

    # GNN fallback
    preds = pd.read_csv(gnn_pred_path)
    gnn = {int(r["node_id"]): float(r["predicted_score"]) for _, r in pd.read_csv(gnn_pred_path).iterrows()}

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
    print("=" * 65)
    print("  AraBERT 5-CLASS CLASSIFIER TRAINING")
    print("=" * 65)
    print(f"  Device: {DEVICE}")

    # Load data
    print(f"\n[1] Loading training data...")
    texts, labels = load_data()
    dist = Counter(labels)
    print(f"  Total: {len(texts):,} | Per class: {dict(dist)}")
    print(f"  Class names: {dict(enumerate(CLASS_NAMES))}")

    # Shuffle and split
    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.permutation(len(texts))
    n_test = max(1, int(len(texts) * TEST_SPLIT))
    n_val = max(1, int(len(texts) * VAL_SPLIT))
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test+n_val]
    train_idx = idx[n_test+n_val:]

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    print(f"\n  Train: {len(train_texts):,} | Val: {len(val_texts):,} | Test: {len(test_texts):,}")

    # Load model
    print(f"\n[2] Loading AraBERT v2 with 5-class head...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=5).to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Compute class weights (inverse frequency)
    class_counts = Counter(train_labels)
    max_cnt = max(class_counts.values())
    class_weights = torch.tensor(
        [max_cnt / class_counts.get(i, 1) for i in range(5)], dtype=torch.float
    ).to(DEVICE)
    print(f"  Class weights: {class_weights.tolist()}")

    # DataLoaders
    train_ds = ClassifierDataset(train_texts, train_labels, tokenizer)
    val_ds = ClassifierDataset(val_texts, val_labels, tokenizer)
    test_ds = ClassifierDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE * 2)
    test_loader = DataLoader(test_ds, BATCH_SIZE * 2)

    # Training
    print(f"\n[3] Training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = NUM_EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    best_val_f1 = 0
    best_state = None
    start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        loss, acc = train_epoch(model, train_loader, optimizer, scheduler, scaler)
        yv_true, yv_pred = evaluate(model, val_loader)
        val_f1 = f1_score(yv_true, yv_pred, average="macro")
        val_acc = accuracy_score(yv_true, yv_pred)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch:>2}: loss={loss:.4f}, train_acc={acc:.4f}, val_f1={val_f1:.4f}, val_acc={val_acc:.4f}")

    train_time = time.time() - start
    model.load_state_dict(best_state)
    print(f"  Complete in {train_time:.0f}s (best val F1: {best_val_f1:.4f})")

    # Test evaluation
    print(f"\n[4] Test set evaluation...")
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

    # For comparison: also compute Spearman if we map classes to scores
    y_true_scores = np.array([CLASS_TO_SCORE[c] for c in y_true])
    y_pred_scores = np.array([CLASS_TO_SCORE[c] for c in y_pred])
    spear_r, _ = spearmanr(y_true_scores, y_pred_scores)
    pear_r, _ = pearsonr(y_true_scores, y_pred_scores)

    print(f"\n  Accuracy:     {acc:.4f}")
    print(f"  F1 Macro:     {f1_macro:.4f}")
    print(f"  Cohen's κ:    {kappa:.4f}")
    print(f"  Spearman ρ:   {spear_r:.4f}")
    print(f"  Pearson r:    {pear_r:.4f}")

    # Compare with regression version
    print(f"\n  {'=' * 55}")
    print(f"  REGRESSION vs CLASSIFICATION")
    print(f"  {'=' * 55}")
    print(f"  {'Metric':<25s} {'Regression':>12s} {'Classification':>12s}")
    print(f"  {'-' * 50}")
    # From saved regression eval
    try:
        reg_eval = json.load(open(PROJECT / "output/track_a/evaluation_metrics.json", encoding="utf-8"))
        for m, reg_val in [("Spearman ρ", reg_eval["spearman_r"]),
                          ("Cohen's κ", reg_eval["cohen_kappa"]),
                          ("F1 (binary)", reg_eval["binary_f1"])]:
            cls_val = {"Spearman ρ": spear_r, "Cohen's κ": kappa, "F1 (binary)": acc}[m]
            delta = cls_val - reg_val
            arrow = "↑" if delta > 0 else "↓"
            print(f"  {m:<25s} {reg_val:>12.4f} {cls_val:>12.4f} {arrow}")
    except:
        pass

    # ── Save model ──
    model.save_pretrained(OUT_DIR / "arabert_classifier")
    tokenizer.save_pretrained(OUT_DIR / "arabert_classifier")
    print(f"\n  Saved: {OUT_DIR / 'arabert_classifier'}")

    metrics = {
        "model": "5-class classifier",
        "n_train": len(train_texts),
        "n_val": len(val_texts),
        "n_test": len(test_texts),
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "cohen_kappa": round(kappa, 4),
        "spearman_r": round(spear_r, 4),
        "pearson_r": round(pear_r, 4),
        "train_time_seconds": train_time,
        "class_distribution": {int(k): int(v) for k, v in dist.items()},
        "class_weights": class_weights.tolist(),
    }
    with open(OUT_DIR / "classifier_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {OUT_DIR / 'classifier_metrics.json'}")

    # ── Score ALL KG narrators ──
    print(f"\n[5] Scoring all 262K KG narrators...")
    hybrid_scores, covered = score_kg_narrators(model, tokenizer)

    scores_file = OUT_DIR / "narrator_scores_classifier.json"
    with open(scores_file, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in hybrid_scores.items()}, f)
    print(f"  Saved: {scores_file}")
    print(f"  Coverage: {covered:,}/{len(hybrid_scores):,} ({covered/len(hybrid_scores)*100:.1f}%)")

    vals = list(hybrid_scores.values())
    print(f"  Score range: [{min(vals):.4f}, {max(vals):.4f}]")
    print(f"  Score mean: {np.mean(vals):.4f}, std: {np.std(vals):.4f}")

    # Histogram
    bins = {0.85: "thiqah", 0.70: "saduq", 0.40: "maqbul", 0.15: "daif", 0.05: "matruk"}
    for threshold, name in sorted(bins.items(), reverse=True):
        cnt = sum(1 for v in vals if abs(v - threshold) < 0.01)
        print(f"    {name:10s} ({threshold:.2f}): {cnt:>8,} ({cnt/len(vals)*100:.1f}%)")

    print(f"\n{'=' * 65}")
    print(f"  CLASSIFIER TRAINING COMPLETE")
    print(f"  F1 Macro: {f1_macro:.4f} | Spearman ρ: {spear_r:.4f}")
    print(f"{'=' * 65}")

import pandas as pd
if __name__ == "__main__":
    main()
