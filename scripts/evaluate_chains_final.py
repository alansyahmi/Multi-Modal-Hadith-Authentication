"""
FINAL CHAIN EVALUATION — combining all critical improvements:

1. Weakest-link scoring (classical hadith principle)
2. Edge risk modifiers (tadlis ×0.80, ikhtilat ×0.50, etc.)
3. Calibrated thresholds (isotonic regression on Kutub as-Sitta)
4. Proper grade distribution output

Usage:
  python scripts/evaluate_chains_final.py
"""
import json, ast, sys, re, time
from collections import defaultdict
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

SANADSET = Path("Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv")
NODE_NAMES_PATH = Path("output/stage2/node_names.json")
GNN_PRED = Path("output/stage3/narrator_predictions.csv")
RISK_MODIFIERS = Path("output/risk_modifiers/risk_modifiers.json")
ALIAS_INDEX = Path("output/shamela/alias_index.json")
EVAL_CSV = Path("output/stage1/kutub_sitta_evaluation.csv")
ALIGNMENT = Path("output/stage1/alignment_results.json")
OUT_DIR = Path("output/final_evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Grade thresholds (will be calibrated)
GRADE_BINS = {"Sahih": (0.85, 1.0), "Hasan": (0.70, 0.85),
              "Maqbul": (0.50, 0.70), "Daif": (0.15, 0.50), "Matruk": (0.0, 0.15)}

def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

print("=" * 65)
print("  FINAL CHAIN EVALUATION")
print("  Weakest-Link + Risk Modifiers + Calibrated Thresholds")
print("=" * 65)

# ── 1. Build hybrid narrator scoring ──
print("\n[1/5] Building narrator scores...")

preds = pd.read_csv(GNN_PRED)
gnn = {int(r["node_id"]): float(r["predicted_score"]) for _, r in preds.iterrows()}
with open(NODE_NAMES_PATH, encoding="utf-8") as f:
    id_to_name = json.load(f)

name_to_score = {}
for nid, name in id_to_name.items():
    name_to_score[norm(name)] = gnn.get(int(nid), 0.632)
print(f"  GNN scores for {len(name_to_score):,} narrators")

# ── 2. Build risk modifier lookup ──
print("\n[2/5] Building risk modifier lookup...")

risk_mods = json.load(open(RISK_MODIFIERS, encoding="utf-8"))
# Map entry_id → worst risk modifier
entry_risk = {}
for eid, mods in risk_mods.items():
    entry_risk[int(eid)] = min(m["modifier"] for m in mods)

# Extend to KG narrators via alias index
alias_index = json.load(open(ALIAS_INDEX, encoding="utf-8"))
narrator_risk = {}
for alias_key, eids in alias_index.items():
    eid = eids[0]
    if eid in entry_risk:
        narrator_risk[norm(alias_key)] = entry_risk[eid]

print(f"  {len(entry_risk):,} encyclopedia entries with risk modifiers")
print(f"  {len(narrator_risk):,} KG narrators mapped to risk modifiers")

# ── 3. Calibrate thresholds using Kutub as-Sitta grades ──
print("\n[3/5] Calibrating thresholds...")

eval_df = pd.read_csv(EVAL_CSV)
grade_map = {"Ṣaḥīḥ": 4, "Ḥasan": 3, "Ḍaʿīf": 2, "Mawḍūʿ": 1}
eval_df["tier"] = eval_df["grade_normalised"].map(grade_map).fillna(0)

# Score evaluation set chains using weakest-link
sanad_col = []
for chunk in pd.read_csv(SANADSET, usecols=["Sanad"], chunksize=100_000, dtype_backend="pyarrow"):
    sanad_col.extend(chunk["Sanad"].tolist())

align = json.load(open(ALIGNMENT, encoding="utf-8"))
eval_scores = []
eval_tiers = []
eval_grades = []

for aidx, entry in enumerate(align):
    if not entry.get("is_match"): continue
    corpus_idx = int(entry["corpus_idx"])
    if corpus_idx >= len(sanad_col): continue
    sanad_str = sanad_col[corpus_idx]
    if pd.isna(sanad_str): continue
    try: chain = json.loads(sanad_str.replace("'", '"'))
    except:
        try: chain = ast.literal_eval(sanad_str)
        except: continue
    names = [n.strip() for n in chain if len(n.strip()) > 3]
    if len(names) < 2: continue

    scores = []
    for n in names:
        nkey = norm(n)
        base = name_to_score.get(nkey, 0.632)
        risk = narrator_risk.get(nkey, 1.0)
        scores.append(base * risk)

    chain_score = float(min(scores))  # Weakest-link
    eval_scores.append(chain_score)
    eval_tiers.append(eval_df.iloc[aidx]["tier"])
    eval_grades.append(eval_df.iloc[aidx]["grade_normalised"])

print(f"  Calibration set: {len(eval_scores):,} aligned hadiths")
print(f"  Grade distribution: {dict(pd.Series(eval_grades).value_counts())}")

# Isotonic calibration: predict grade tier from chain score
iso = IsotonicRegression(y_min=0.5, y_max=4.5, out_of_bounds="clip", increasing=True)
iso.fit(eval_scores, eval_tiers)

# Find optimal thresholds for each grade boundary
score_bounds = np.linspace(min(eval_scores), max(eval_scores), 100)
calibrated_tiers = iso.predict(score_bounds)

# Extract thresholds where tier crosses boundaries
thresholds = {}
for boundary in [3.5, 2.5, 1.5, 0.5]:  # Sahih/Hasan, Hasan/Maqbul, etc.
    crossing = np.where(np.diff(calibrated_tiers >= boundary))[0]
    if len(crossing) > 0:
        idx = crossing[-1]
        thresholds[boundary] = float(score_bounds[idx])

print(f"\n  Calibrated thresholds:")
for b, t in sorted(thresholds.items(), reverse=True):
    labels = {3.5: "Sahih/Hasan", 2.5: "Hasan/Maqbul", 1.5: "Maqbul/Daif", 0.5: "Daif/Matruk"}
    print(f"    {labels[b]:20s}: ≥ {t:.4f}")

# ── 4. Score all 650K chains ──
print(f"\n[4/5] Scoring 650K chains (weakest-link + risk modifiers)...")

total, nerror = 0, 0
risk_applied = 0
results = []
start = time.time()

for chunk_idx, chunk in enumerate(pd.read_csv(
    SANADSET, usecols=["Sanad"], chunksize=100_000, dtype_backend="pyarrow",
)):
    t0 = time.time()
    for _, row in chunk.iterrows():
        sanad = row["Sanad"]
        if pd.isna(sanad): continue
        try: chain = json.loads(sanad.replace("'", '"'))
        except:
            try: chain = ast.literal_eval(sanad)
            except: nerror += 1; continue
        if not isinstance(chain, list) or len(chain) < 2: continue
        names = [n.strip() for n in chain if len(n.strip()) > 3]
        if len(names) < 2: continue

        scores = []
        for n in names:
            nkey = norm(n)
            base_score = name_to_score.get(nkey, 0.632)
            risk = narrator_risk.get(nkey, 1.0)
            if risk < 1.0:
                risk_applied += 1
            scores.append(base_score * risk)

        chain_score = float(min(scores))  # Weakest-link
        results.append({"cl": len(names), "sc": round(chain_score, 4)})
        total += 1

    print(f"  Chunk {chunk_idx+1}: {total:,} chains ({time.time()-t0:.0f}s, {risk_applied:,} risk modifiers applied)")

ttime = time.time() - start
df = pd.DataFrame(results)
print(f"\n  Processed: {len(df):,} chains, {ttime:.0f}s, {risk_applied:,} risk modifiers applied")

# ── 5. Grade distribution with calibrated thresholds ──
print(f"\n[5/5] Grade distribution (calibrated)...")

# Use calibrated thresholds if available, else defaults
T = {
    "Sahih": thresholds.get(3.5, 0.78),
    "Hasan": thresholds.get(2.5, 0.68),
    "Maqbul": thresholds.get(1.5, 0.55),
    "Daif": thresholds.get(0.5, 0.20),
}

def to_grade(s):
    if s >= T["Sahih"]: return "Sahih"
    elif s >= T["Hasan"]: return "Hasan"
    elif s >= T["Maqbul"]: return "Maqbul"
    elif s >= T["Daif"]: return "Daif"
    else: return "Matruk"

df["gr"] = df["sc"].apply(to_grade)

print(f"\n{'=' * 65}")
print(f"  RESULTS: {len(df):,} chains")
print(f"{'=' * 65}")

print(f"\nScore: mean={df['sc'].mean():.4f}, std={df['sc'].std():.4f}")

print(f"\nThresholds used:")
for grade in ["Sahih", "Hasan", "Maqbul", "Daif"]:
    print(f"  {grade:10s}: ≥ {T[grade]:.4f}")

print(f"\nGrade distribution:")
for grade in ["Sahih", "Hasan", "Maqbul", "Daif", "Matruk"]:
    cnt = len(df[df["gr"] == grade])
    print(f"  {grade:10s}: {cnt:>8,} ({cnt/len(df)*100:.1f}%)")

# ── Save ──
df.to_csv(OUT_DIR / "chain_results_final.csv", index=False, encoding="utf-8-sig")
print(f"\nSaved: {OUT_DIR / 'chain_results_final.csv'} ({len(df):,} rows)")

summary = {
    "method": "weakest_link_with_risk",
    "total_chains": len(df),
    "mean_score": round(float(df["sc"].mean()), 4),
    "std_score": round(float(df["sc"].std()), 4),
    "thresholds": T,
    "grade_distribution": {g: int(c) for g, c in df["gr"].value_counts().items()},
    "risk_modifiers_applied": risk_applied,
    "time_seconds": ttime,
    "calibration": {
        "n_calibration_samples": len(eval_scores),
        "isotonic_regression": True,
    },
}
with open(OUT_DIR / "summary_final.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {OUT_DIR / 'summary_final.json'}")

print(f"\n{'=' * 65}")
print(f"  COMPLETE")
print(f"  Weakest-link + Risk Modifiers + Calibrated Thresholds")
print(f"  {df['gr'].value_counts().to_dict()}")
print(f"{'=' * 65}")
