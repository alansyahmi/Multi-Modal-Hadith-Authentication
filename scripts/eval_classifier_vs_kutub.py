"""
Evaluate 5-class classifier scores against Kutub as-Sitta canonical grades.

Compares: GNN-only vs Classifier chain scores on the 1,022 aligned evaluation hadiths.
"""
import json, ast, sys, re, time, math
from collections import defaultdict, Counter
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score

SANADSET = Path("Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv")
NODE_NAMES = Path("output/stage2/node_names.json")
GNN_PRED = Path("output/stage3/narrator_predictions.csv")
CLASSIFIER_SCORES = Path("output/track_a_classifier/narrator_scores_classifier.json")
RISK_MOD = Path("output/risk_modifiers/risk_modifiers.json")
ALIAS = Path("output/shamela/alias_index.json")
EVAL_CSV = Path("output/stage1/kutub_sitta_evaluation.csv")
ALIGNMENT = Path("output/stage1/alignment_results.json")
OUT_DIR = Path("output/final_evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

print("=" * 65)
print("  CLASSIFIER vs CANONICAL GRADES (KUTUB AS-SITTA)")
print("=" * 65)

# ── Build both score lookups ──
print("\nBuilding score lookups...")

# Load node names
with open(NODE_NAMES, encoding="utf-8") as f:
    id_to_name = json.load(f)

# GNN scores
gnn = {int(r["node_id"]): float(r["predicted_score"]) for _, r in pd.read_csv(GNN_PRED).iterrows()}
gnn_name = {norm(v): gnn.get(int(k), 0.632) for k, v in id_to_name.items()}

# Classifier scores (keyed by string node_id)
cls_scores = json.load(open(CLASSIFIER_SCORES, encoding="utf-8"))
cls_name = {}
for nid_str, name in id_to_name.items():
    nid = int(nid_str)
    score = cls_scores.get(str(nid), cls_scores.get(nid_str, 0.632))
    cls_name[norm(name)] = float(score)

# Risk modifiers
risk_mods = json.load(open(RISK_MOD, encoding="utf-8"))
entry_risk = {int(k): min(m["modifier"] for m in v) for k, v in risk_mods.items()}
alias_data = json.load(open(ALIAS, encoding="utf-8"))
narrator_risk = {}
for ak, eids in alias_data.items():
    if eids[0] in entry_risk:
        narrator_risk[norm(ak)] = entry_risk[eids[0]]

# ── Load evaluation hadiths ──
eval_df = pd.read_csv(EVAL_CSV)
grade_labels = {"Ṣaḥīḥ": "Sahih", "Ḥasan": "Hasan", "Ḍaʿīf": "Daif", "Mawḍūʿ": "Matruk"}
eval_df["label"] = eval_df["grade_normalised"].map(grade_labels).fillna("Daif")
print(f"Eval set: {len(eval_df)} hadiths, grades: {dict(eval_df['label'].value_counts())}")

# Load Sanadset
sanad_col = []
for chunk in pd.read_csv(SANADSET, usecols=["Sanad"], chunksize=100_000, dtype_backend="pyarrow"):
    sanad_col.extend(chunk["Sanad"].tolist())

align = json.load(open(ALIGNMENT, encoding="utf-8"))

# ── Score chains (both methods) ──
print(f"\nScoring {sum(1 for a in align if a['is_match'])} aligned hadiths...")

grades_order = ["Sahih", "Hasan", "Daif", "Matruk"]

def score_chain(names, score_lookup):
    """Weakest-link + risk modifiers."""
    scores = []
    for n in names:
        nkey = norm(n)
        base = score_lookup.get(nkey, 0.632)
        risk = narrator_risk.get(nkey, 1.0)
        scores.append(base * risk)
    return float(min(scores)) if scores else 0.632

def chain_to_grade(score):
    if score >= 0.70: return "Sahih"
    elif score >= 0.55: return "Hasan"
    elif score >= 0.35: return "Daif"
    else: return "Matruk"

results = []
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

    gnn_score = score_chain(names, gnn_name)
    cls_score = score_chain(names, cls_name)

    row = eval_df.iloc[aidx]
    results.append({
        "book": row.get("book_name", ""),
        "actual_grade": row["label"],
        "gnn_score": round(gnn_score, 4),
        "gnn_grade": chain_to_grade(gnn_score),
        "cls_score": round(cls_score, 4),
        "cls_grade": chain_to_grade(cls_score),
        "chain_len": len(names),
    })

df = pd.DataFrame(results)
print(f"  Scored {len(df):,} hadiths")

# ── RESULTS ──
print(f"\n{'=' * 65}")
print(f"  CANONICAL GRADE COMPARISON ({len(df):,} hadiths)")
print(f"{'=' * 65}")

# Score by actual grade
print(f"\n  Mean chain score by canonical grade:")
for g in grades_order:
    grp = df[df["actual_grade"] == g]
    if len(grp) > 0:
        print(f"    {g:10s} (n={len(grp):>4}): GNN={grp['gnn_score'].mean():.4f} | Classifier={grp['cls_score'].mean():.4f}")

# Confusion matrices side by side
for method, col in [("GNN-Only", "gnn_grade"), ("Classifier", "cls_grade")]:
    print(f"\n  ── {method} ──")
    actual = df["actual_grade"].tolist()
    pred = df[col].tolist()

    cm = confusion_matrix(actual, pred, labels=grades_order)
    header = "             " + " ".join(f"{g:>8s}" for g in grades_order)
    print(header)
    for i, g in enumerate(grades_order):
        row = " ".join(f"{cm[i][j]:>8d}" for j in range(len(grades_order)))
        print(f"    {g:8s}   {row}")

    acc = accuracy_score(actual, pred)
    f1 = f1_score(actual, pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(actual, pred, weights="quadratic")

    def relaxed(a, p, d=1):
        tier = {"Sahih":4,"Hasan":3,"Daif":2,"Matruk":1}
        return sum(1 for x,y in zip(a,p) if abs(tier.get(x,2)-tier.get(y,2))<=d)/max(len(a),1)

    print(f"\n    Exact Accuracy:    {acc:.4f}")
    print(f"    ±1 Grade Acc:      {relaxed(actual, pred, 1):.4f}")
    print(f"    F1 Macro:          {f1:.4f}")
    print(f"    Cohen's κ (quad):  {kappa:.4f}")

    # Per-grade recall
    for g in grades_order:
        mask = [a == g for a in actual]
        if sum(mask) > 0:
            grade_acc = sum(1 for i in range(len(actual)) if mask[i] and pred[i] == g) / sum(mask)
            print(f"    {g:10s} recall:    {grade_acc:.4f}")

# ── By book ──
print(f"\n  ── Accuracy by Book ──")
print(f"  {'Book':35s} {'n':>5s} {'GNN':>8s} {'Classifier':>8s} {'Winner'}")
print(f"  {'-' * 60}")
for book in sorted(set(df["book"])):
    grp = df[df["book"] == book]
    if len(grp) < 5: continue
    gnn_acc = accuracy_score(grp["actual_grade"], grp["gnn_grade"])
    cls_acc = accuracy_score(grp["actual_grade"], grp["cls_grade"])
    winner = "GNN" if gnn_acc > cls_acc else "CLS" if cls_acc > gnn_acc else "TIE"
    print(f"  {book[:35]:35s} {len(grp):>5d} {gnn_acc:>8.4f} {cls_acc:>8.4f} {winner:>8s}")

# ── Direct comparison table ──
print(f"\n{'=' * 65}")
print(f"  GNN vs CLASSIFIER — HEAD TO HEAD")
print(f"{'=' * 65}")
print(f"  {'Metric':<30s} {'GNN-Only':>10s} {'Classifier':>10s} {'Δ':>8s}")
print(f"  {'-' * 60}")

gnn_actual = df["actual_grade"].tolist()
gnn_pred = df["gnn_grade"].tolist()
cls_pred = df["cls_grade"].tolist()

for metric_name in ["Exact Accuracy", "±1 Grade Acc", "F1 Macro", "Cohen's κ"]:
    if metric_name == "Exact Accuracy":
        g = accuracy_score(gnn_actual, gnn_pred)
        c = accuracy_score(gnn_actual, cls_pred)
    elif metric_name == "±1 Grade Acc":
        def r(p, d=1): return relaxed(gnn_actual, p, d)
        g = r(gnn_pred)
        c = r(cls_pred)
    elif metric_name == "F1 Macro":
        g = f1_score(gnn_actual, gnn_pred, average="macro", zero_division=0)
        c = f1_score(gnn_actual, cls_pred, average="macro", zero_division=0)
    else:
        g = cohen_kappa_score(gnn_actual, gnn_pred, weights="quadratic")
        c = cohen_kappa_score(gnn_actual, cls_pred, weights="quadratic")
    delta = c - g
    arrow = "↑" if delta > 0 else "↓" if delta < 0 else "—"
    print(f"  {metric_name:<30s} {g:>10.4f} {c:>10.4f} {arrow} {abs(delta):.4f}")

# ── Show representative predictions ──
print(f"\n{'=' * 65}")
print(f"  SAMPLE PREDICTIONS (first 10 Sahih hadiths)")
print(f"{'=' * 65}")
sahih = df[df["actual_grade"] == "Sahih"].head(10)
print(f"  {'Book':30s} {'Actual':>8s} {'GNN':>8s} {'CLS':>8s} {'Score (CLS)':>12s}")
print(f"  {'-' * 72}")
for _, r in sahih.iterrows():
    print(f"  {r['book'][:30]:30s} {r['actual_grade']:>8s} {r['gnn_grade']:>8s} {r['cls_grade']:>8s} {r['cls_score']:>12.4f}")

# ── Save ──
df.to_csv(OUT_DIR / "classifier_vs_canonical.csv", index=False)
summary = {
    "n_hadiths": len(df),
    "gnn": {"accuracy": accuracy_score(gnn_actual, gnn_pred), "f1_macro": f1_score(gnn_actual, gnn_pred, average="macro", zero_division=0),
            "cohen_kappa": cohen_kappa_score(gnn_actual, gnn_pred, weights="quadratic")},
    "classifier": {"accuracy": accuracy_score(gnn_actual, cls_pred), "f1_macro": f1_score(gnn_actual, cls_pred, average="macro", zero_division=0),
                  "cohen_kappa": cohen_kappa_score(gnn_actual, cls_pred, weights="quadratic")},
}
with open(OUT_DIR / "classifier_vs_canonical.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved: {OUT_DIR / 'classifier_vs_canonical.csv'}, {OUT_DIR / 'classifier_vs_canonical.json'}")
