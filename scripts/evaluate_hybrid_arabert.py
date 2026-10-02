"""
Hybrid chain evaluation: AraBERT + GNN combined scoring.

For narrators with bio text → AraBERT inference (wider score range)
For narrators without → GNN fallback
Then: weakest-link + risk modifiers → compare vs Kutub as-Sitta grades

Usage:
  python scripts/evaluate_hybrid_arabert.py
"""
import json, ast, sys, re, time, math
from collections import defaultdict
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
from tqdm import tqdm

SANADSET = Path("Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv")
NODE_NAMES = Path("output/stage2/node_names.json")
GNN_PRED = Path("output/stage3/narrator_predictions.csv")
RISK_MOD = Path("output/risk_modifiers/risk_modifiers.json")
ALIAS_IDX = Path("output/shamela/alias_index.json")
ENCYCLOPEDIA = Path("output/shamela/narrator_encyclopedia.json")
EVAL_CSV = Path("output/stage1/kutub_sitta_evaluation.csv")
ALIGNMENT = Path("output/stage1/alignment_results.json")
ARABERT_DIR = Path("output/track_a/arabert_regression_model")
OUT_DIR = Path("output/final_evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
MAX_LENGTH = 128

def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

def to_grade(s):
    if s >= 0.70: return "Sahih"
    elif s >= 0.55: return "Hasan"
    elif s >= 0.35: return "Daif"
    else: return "Matruk"

print("=" * 65)
print("  HYBRID AraBERT + GNN CHAIN EVALUATION")
print("=" * 65)

# ── 1. Load AraBERT model ──
print(f"\n[1/5] Loading AraBERT model...")
tokenizer = AutoTokenizer.from_pretrained(str(ARABERT_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(ARABERT_DIR)).to(DEVICE)
model.eval()
print(f"  Device: {DEVICE} | Model: {sum(p.numel() for p in model.parameters()):,} params")

# ── 2. Build bio-text lookup for AraBERT ──
print(f"\n[2/5] Building name → bio text lookup...")
enc = json.load(open(ENCYCLOPEDIA, encoding="utf-8"))
alias = json.load(open(ALIAS_IDX, encoding="utf-8"))

entry_text = {}
for e in enc:
    eid, text = e.get("entry_id"), e.get("full_text", "")
    if eid and text and len(text) > 30:
        entry_text[eid] = text[:2000]

name_to_bio = {}
for ak, eids in alias.items():
    eid = eids[0]
    if eid in entry_text:
        name_to_bio[norm(ak)] = entry_text[eid]

bio_narrators = len(name_to_bio)
print(f"  Narrators with bio text: {bio_narrators:,}")

# ── 3. Score ALL KG narrators with hybrid method ──
print(f"\n[3/5] Scoring 183K KG narrators (AraBERT if bio text, else GNN)...")

# Load GNN for fallback
preds = pd.read_csv(GNN_PRED)
gnn = {int(r["node_id"]): float(r["predicted_score"]) for _, r in preds.iterrows()}

with open(NODE_NAMES, encoding="utf-8") as f:
    id_to_name = json.load(f)

# Build list of all unique KG narrators
all_kg = []
for nid, name in id_to_name.items():
    nkey = norm(name)
    all_kg.append((nkey, int(nid)))

# Score: AraBERT for those with bio, GNN for rest
hybrid_scores = {}
arabert_hits = 0

with torch.no_grad():
    for i in tqdm(range(0, len(all_kg), BATCH_SIZE), desc="AraBERT+GNN scoring"):
        batch = all_kg[i:i + BATCH_SIZE]
        texts = [name_to_bio.get(nk, nk) for nk, nid in batch]
        has_bio = [nk in name_to_bio for nk, nid in batch]

        encoded = tokenizer(texts, truncation=True, padding=True,
                           max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
        outputs = model(**encoded)
        arabert_scores = torch.sigmoid(outputs.logits).squeeze(-1).cpu().numpy()

        for j, (nk, nid) in enumerate(batch):
            if has_bio[j]:
                hybrid_scores[nid] = float(arabert_scores[j])
                arabert_hits += 1
            else:
                hybrid_scores[nid] = gnn.get(nid, 0.632)

vals = list(hybrid_scores.values())
print(f"\n  AraBERT-covered: {arabert_hits:,}/{len(all_kg):,} ({arabert_hits/len(all_kg)*100:.1f}%)")
print(f"  Score range: [{min(vals):.4f}, {max(vals):.4f}]")
print(f"  Score mean: {np.mean(vals):.4f}, std: {np.std(vals):.4f}")

# Compare: AraBERT-only vs GNN-only score distribution
arabert_vals = [hybrid_scores[nid] for nk, nid in all_kg if nk in name_to_bio]
gnn_vals = [hybrid_scores[nid] for nk, nid in all_kg if nk not in name_to_bio]
if arabert_vals:
    print(f"\n  AraBERT scores: mean={np.mean(arabert_vals):.4f}, std={np.std(arabert_vals):.4f}")
if gnn_vals:
    print(f"  GNN scores:     mean={np.mean(gnn_vals):.4f}, std={np.std(gnn_vals):.4f}")

# ── Build chain-level name lookup ──
name_to_score = {norm(v): hybrid_scores.get(int(k), 0.632) for k, v in id_to_name.items()}

# Risk modifiers
risk_mods = json.load(open(RISK_MOD, encoding="utf-8"))
entry_risk = {int(k): min(m["modifier"] for m in v) for k, v in risk_mods.items()}
narrator_risk = {}
for ak, eids in alias.items():
    if eids[0] in entry_risk:
        narrator_risk[norm(ak)] = entry_risk[eids[0]]

# ── 4. Evaluate against Kutub as-Sitta ──
print(f"\n[4/5] Evaluating against Kutub as-Sitta...")

eval_df = pd.read_csv(EVAL_CSV)
grade_labels = {"Ṣaḥīḥ": "Sahih", "Ḥasan": "Hasan", "Ḍaʿīf": "Daif", "Mawḍūʿ": "Matruk"}
eval_df["label"] = eval_df["grade_normalised"].map(grade_labels).fillna("Daif")

sanad_col = []
for chunk in pd.read_csv(SANADSET, usecols=["Sanad"], chunksize=100_000, dtype_backend="pyarrow"):
    sanad_col.extend(chunk["Sanad"].tolist())

align = json.load(open(ALIGNMENT, encoding="utf-8"))

cal_scores = []
cal_labels = []
cal_book = []
risk_hits = 0

for aidx, entry in enumerate(align):
    if not entry.get("is_match"): continue
    ci = int(entry["corpus_idx"])
    if ci >= len(sanad_col): continue
    s = sanad_col[ci]
    if pd.isna(s): continue
    try: chain = json.loads(s.replace("'", '"'))
    except:
        try: chain = ast.literal_eval(s)
        except: continue
    names = [n.strip() for n in chain if len(n.strip()) > 3]
    if len(names) < 2: continue

    scores = []
    for n in names:
        nkey = norm(n)
        base = name_to_score.get(nkey, 0.632)
        risk = narrator_risk.get(nkey, 1.0)
        if risk < 1.0: risk_hits += 1
        scores.append(base * risk)

    chain_score = float(min(scores))
    cal_scores.append(chain_score)
    cal_labels.append(eval_df.iloc[aidx]["label"])
    cal_book.append(eval_df.iloc[aidx].get("book_name", ""))

n = len(cal_scores)
print(f"  Evaluated: {n:,} hadiths | Risk applied: {risk_hits:,}")

# ── 5. Results ──
print(f"\n[5/5] Results:")

print(f"\n  Score by actual grade:")
for g in ["Sahih","Hasan","Daif","Matruk"]:
    mask = [l == g for l in cal_labels]
    if sum(mask) > 0:
        sg = [cal_scores[i] for i in range(n) if mask[i]]
        print(f"    {g:10s}: mean={np.mean(sg):.4f}, std={np.std(sg):.4f}, n={len(sg)}")

predicted = [to_grade(s) for s in cal_scores]
actual = cal_labels
grades = ["Sahih","Hasan","Daif","Matruk"]

cm = confusion_matrix(actual, predicted, labels=grades)
header = "           " + " ".join(f"{g:>8s}" for g in grades)
print(f"\n  Confusion Matrix:")
print(header)
for i, g in enumerate(grades):
    row = " ".join(f"{cm[i][j]:>8d}" for j in range(len(grades)))
    print(f"  {g:10s} {row}")

acc = accuracy_score(actual, predicted)
f1 = f1_score(actual, predicted, average="macro")
kappa = cohen_kappa_score(actual, predicted, weights="quadratic")

# Relaxed accuracy
def relaxed_acc(a, p, max_diff=1):
    tier = {"Sahih":4,"Hasan":3,"Daif":2,"Matruk":1}
    return sum(1 for x,y in zip(a,p) if abs(tier.get(x,2)-tier.get(y,2))<=max_diff)/max(len(a),1)

print(f"\n  Accuracy (exact):         {acc:.4f}")
print(f"  Accuracy (±1 grade):      {relaxed_acc(actual, predicted, 1):.4f}")
print(f"  F1 Macro:                 {f1:.4f}")
print(f"  Cohen's κ (quadratic):    {kappa:.4f}")

# Compare vs GNN-only
print(f"\n  {'=' * 55}")
print(f"  HYBRID vs GNN-ONLY COMPARISON")
print(f"  {'=' * 55}")
print(f"  {'Metric':<30s} {'GNN-Only':>10s} {'Hybrid':>10s}")
print(f"  {'-' * 50}")
# Load previous results for comparison
try:
    prev = json.load(open(OUT_DIR / "recalibrated_summary.json", encoding="utf-8"))
    for m, gnn_val in [("Exact Accuracy", prev["accuracy_exact"]),
                        ("±1 Grade Acc", prev["accuracy_relaxed_1"]),
                        ("F1 Macro", prev["f1_macro"]),
                        ("Cohen's κ", prev["cohen_kappa"])]:
        hybrid_val = {"Exact Accuracy": acc, "±1 Grade Acc": relaxed_acc(actual,predicted,1),
                      "F1 Macro": f1, "Cohen's κ": kappa}[m]
        delta = hybrid_val - gnn_val
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "—"
        print(f"  {m:<30s} {gnn_val:>10.4f} {hybrid_val:>10.4f} {arrow} {abs(delta):.4f}")
except:
    pass

print(f"\n  By book (hybrid):")
for book in sorted(set(cal_book)):
    mask = [b == book for b in cal_book]
    if sum(mask) < 5: continue
    ba = accuracy_score([actual[i] for i in range(n) if mask[i]],
                         [predicted[i] for i in range(n) if mask[i]])
    print(f"    {book[:35]:35s}: {ba:.4f} (n={sum(mask)})")

# ── Save ──
summary = {
    "method": "hybrid_arabert_gnn",
    "n_calibration": n,
    "arabert_coverage": arabert_hits,
    "arabert_coverage_pct": round(arabert_hits/len(all_kg)*100, 1),
    "accuracy_exact": round(acc, 4),
    "accuracy_relaxed": round(relaxed_acc(actual, predicted, 1), 4),
    "f1_macro": round(f1, 4),
    "cohen_kappa": round(kappa, 4),
    "by_book": {b: round(float(accuracy_score(
        [actual[i] for i in range(n) if cal_book[i]==b],
        [predicted[i] for i in range(n) if cal_book[i]==b]
    )), 4) for b in set(cal_book) if sum(1 for x in cal_book if x==b) >= 5},
    "score_by_grade": {g: round(float(np.mean([cal_scores[i] for i in range(n) if cal_labels[i]==g])), 4)
                       for g in grades if sum(1 for x in cal_labels if x==g) > 0},
}
with open(OUT_DIR / "hybrid_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {OUT_DIR / 'hybrid_summary.json'}")

print(f"\n{'=' * 65}")
print(f"  HYBRID EVALUATION COMPLETE")
print(f"  AraBERT coverage: {arabert_hits:,}/{len(all_kg):,} narrators")
print(f"  Accuracy: {acc:.4f} | κ: {kappa:.4f}")
print(f"{'=' * 65}")
