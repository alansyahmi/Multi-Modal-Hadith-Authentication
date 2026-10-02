"""Evaluate the 3K grade-balanced set with our chain scoring pipeline."""
import sys, re, ast, json, os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"c:\Projects\FYP Project")

def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

# ── Build narrator scores ──
GRADE_MAP = {"صحابي":1.0,"ثقة":0.85,"صدوق":0.70,"مقبول":0.40,"ضعيف":0.15,"متروك":0.05,"كذاب":0.0}
enc = json.load(open("output/shamela/narrator_encyclopedia.json", encoding="utf-8"))
alias = json.load(open("output/shamela/alias_index.json", encoding="utf-8"))
with open("output/stage2/node_names.json", encoding="utf-8") as f: id_to_name = json.load(f)
classifier = json.load(open("output/track_a_classifier/narrator_scores_classifier.json", encoding="utf-8"))

entry_best = {}
for e in enc:
    for g in e.get("grades", []):
        for t in sorted(GRADE_MAP, key=len, reverse=True):
            if t in g: entry_best[e["entry_id"]] = max(entry_best.get(e["entry_id"],0), GRADE_MAP[t]); break

name_score = {}
for nid_str, name in id_to_name.items():
    nid, nk = int(nid_str), norm(name)
    score = 0.632
    if nk in alias:
        eid = alias[nk][0]
        if eid in entry_best: score = entry_best[eid]
    if score == 0.632:
        cv = classifier.get(nid_str, classifier.get(str(nid)))
        if cv and float(cv) != 0.632: score = float(cv)
    name_score[nk] = score

# ── Load eval set ──
eval_df = pd.read_csv("output/final_evaluation/stage1_aligned_balanced_4k.csv")
print(f"Loaded {len(eval_df):,} evaluation hadiths")
for g, c in Counter(eval_df["grade_4"]).most_common():
    print(f"  {g}: {c}")

# ── Load Sanadset and score chains ──
sanad_col = []
for chunk in pd.read_csv(
    "Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv",
    usecols=["Sanad"], chunksize=100_000, dtype_backend="pyarrow",
):
    sanad_col.extend(chunk["Sanad"].tolist())

T = (0.45, 0.425, 0.15)
actual, preds = [], []

for _, row in eval_df.iterrows():
    ci = int(row.get("corpus_idx", -1))
    if ci < 0 or ci >= len(sanad_col): continue
    s = sanad_col[ci]
    if pd.isna(s): continue
    try: ch = json.loads(s.replace("'", '"'))
    except:
        try: ch = ast.literal_eval(s)
        except: continue
    names = [n.strip() for n in ch if len(n.strip()) > 3]
    if len(names) < 2: continue

    sc = sorted([name_score.get(norm(n), 0.632) for n in names])
    cs = (sc[0]+sc[1])/2 if len(sc)>=2 else sc[0]
    g = "Sahih" if cs>=T[0] else "Hasan" if cs>=T[1] else "Daif" if cs>=T[2] else "Matruk"
    actual.append(str(row["grade_4"]))
    preds.append(g)

n = len(actual)
bl = max(Counter(actual).values())/max(n,1)
acc = accuracy_score(actual, preds)
f1 = f1_score(actual, preds, average="macro", zero_division=0)
k = cohen_kappa_score(actual, preds, weights="quadratic")

print(f"\n{'=' * 60}")
print(f"GRADE-BALANCED EVALUATION (n={n})")
print(f"{'=' * 60}")
print(f"  Baseline (always majority): {bl:.4f}")
print(f"  Accuracy:  {acc:.4f}")
print(f"  F1 Macro:  {f1:.4f}")
print(f"  Cohen's κ: {k:.4f}")

# Confusion matrix
grades = ["Sahih","Hasan","Daif","Matruk"]
print(f"\n  {'':>10s} " + " ".join(f"{g:>8s}" for g in grades))
for a in grades:
    row = [sum(1 for i in range(n) if actual[i]==a and preds[i]==p) for p in grades]
    print(f"  {a:>8s} " + " ".join(f"{c:>8d}" for c in row))

# Per-grade recall
print(f"\n  Per-grade recall:")
for g in grades:
    correct = sum(1 for i in range(n) if actual[i]==g and preds[i]==g)
    total = sum(1 for a in actual if a==g)
    print(f"    {g:10s}: {correct}/{total} ({correct/max(total,1)*100:.1f}%)")
