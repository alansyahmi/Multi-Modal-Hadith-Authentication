"""
Experiment: Do machine-scored Sahih hadiths align with Mutawatir/Muttafaqun alayhi?

Uses the Stage 1 alignment results to cross-reference evaluation hadiths with
their Sanadset chains, then compares GNN chain scores across scholarly consensus tiers.

Usage:
  python scripts/sahih_vs_mutawatir_experiment.py
"""
import sys, json, re, ast
from pathlib import Path
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

# ── Paths ──
EVAL_PATH = Path("output/stage1/kutub_sitta_evaluation.csv")
ALIGNMENT_PATH = Path("output/stage1/alignment_results.json")
SANADSET_RAW = Path("Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv")
NODE_NAMES_PATH = Path("output/stage2/node_names.json")
GNN_PREDICTIONS = Path("output/stage3/narrator_predictions.csv")
OUTPUT_DIR = Path("output/consensus_experiment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

print("=" * 60)
print("  CONSENSUS TIER ANALYSIS")
print("=" * 60)

# ── Build KG score lookup ──
print(f"\nBuilding narrator score lookup...")
with open(NODE_NAMES_PATH, encoding="utf-8") as f:
    id_to_name = json.load(f)
preds_df = pd.read_csv(GNN_PREDICTIONS)
pred_map = {int(row["node_id"]): float(row["predicted_score"]) for _, row in preds_df.iterrows()}
name_to_score = {}
for nid_str, name in id_to_name.items():
    nid = int(nid_str)
    if nid in pred_map:
        name_to_score[norm(name)] = pred_map[nid]
default_score = float(np.mean(list(name_to_score.values())))
print(f"  {len(name_to_score):,} names mapped, mean={default_score:.4f}")

# ── Load eval set and alignment ──
print(f"\nLoading evaluation set and alignment...")
eval_df = pd.read_csv(EVAL_PATH)
align = json.load(open(ALIGNMENT_PATH, encoding="utf-8"))

grade_map = {"Ṣaḥīḥ": 4, "Ḥasan": 3, "Ḍaʿīf": 2, "Mawḍūʿ": 1, "Unknown": 0}
eval_df["grade_tier"] = eval_df["grade_normalised"].map(grade_map).fillna(2)
print(f"  {len(eval_df)} eval hadiths, {sum(a['is_match'] for a in align)} aligned")

# ── Load raw Sanadset as list for fast index access ──
print(f"\nLoading Sanadset rows by index...")
sanad_col = []
for chunk in pd.read_csv(SANADSET_RAW, usecols=["Sanad"], chunksize=100_000, dtype_backend="pyarrow"):
    sanad_col.extend(chunk["Sanad"].tolist())
print(f"  {len(sanad_col):,} rows loaded")

# ── Score each aligned hadith ──
print(f"\nScoring aligned chains...")
results = []
matched = 0
for aidx, entry in enumerate(align):
    if not entry.get("is_match"):
        continue
    corpus_idx = int(entry["corpus_idx"])
    if corpus_idx >= len(sanad_col):
        continue

    sanad_str = sanad_col[corpus_idx]
    if pd.isna(sanad_str):
        continue

    try:
        chain = json.loads(sanad_str.replace("'", '"'))
    except:
        try:
            chain = ast.literal_eval(sanad_str)
        except:
            continue

    names = [n.strip() for n in chain if len(n.strip()) > 3]
    if len(names) < 2:
        continue

    scores = [name_to_score.get(norm(n), default_score) for n in names]
    chain_score = float(np.mean(scores))

    row = eval_df.iloc[aidx]
    results.append({
        "book": row["book_name"],
        "grade": row["grade_normalised"],
        "grade_tier": row["grade_tier"],
        "chain_score": round(chain_score, 4),
        "chain_length": len(names),
        "aligned": True,
    })
    matched += 1

df = pd.DataFrame(results)
print(f"  Scored: {len(df):,}/{matched} aligned hadiths")

# ── Analysis ──
print(f"\n{'=' * 60}")
print(f"  RESULTS")
print(f"{'=' * 60}")

print(f"\n1. By Traditional Grade:")
for grade, grp in df.groupby("grade"):
    scores = grp["chain_score"]
    gname = {"Ṣaḥīḥ": "Sahih", "Ḥasan": "Hasan", "Ḍaʿīf": "Daif", "Mawḍūʿ": "Mawdu", "Unknown": "Unknown"}.get(grade, grade)
    print(f"  {gname:10s}: n={len(grp):>4}, mean={scores.mean():.4f}, median={scores.median():.4f}, std={scores.std():.4f}")

print(f"\n2. By Book (Scholarly Consensus Proxy):")
def book_tier(b):
    if "Bukhari" in b: return "Bukhari"
    if "Muslim" in b: return "Muslim"
    if "Tirmidhi" in b: return "Tirmidhi"
    if "Abu Dawud" in b: return "Abu Dawud"
    if "Nasai" in b: return "Nasai"
    if "Ibn Majah" in b: return "Ibn Majah"
    return b

df["book_tier"] = df["book"].apply(book_tier)
for book, grp in sorted(df.groupby("book_tier")):
    print(f"  {book:15s}: n={len(grp):>4}, "
          f"mean={grp['chain_score'].mean():.4f}, median={grp['chain_score'].median():.4f}")

# Sahih/Hasan only by book
sahih = df[df["grade_tier"] >= 3]
print(f"\n3. Sahih/Hasan by Book (n={len(sahih)}):")
for book, grp in sorted(sahih.groupby("book_tier")):
    print(f"  {book:15s}: n={len(grp):>4}, mean={grp['chain_score'].mean():.4f}")

print(f"\n4. Consensus Tier Comparison (Sahih/Hasan only):")
high = sahih[sahih["book_tier"].isin(["Bukhari", "Muslim"])]
low = sahih[~sahih["book_tier"].isin(["Bukhari", "Muslim"])]
print(f"  Bukhari+Muslim (= muttafaqun proxy): n={len(high)}, mean={high['chain_score'].mean():.4f}")
print(f"  Other books:                         n={len(low)},  mean={low['chain_score'].mean():.4f}")
diff = high["chain_score"].mean() - low["chain_score"].mean()
print(f"  Difference: {diff:.4f}")
if diff > 0:
    print(f"  ✅ Our scores AGREE — higher consensus = higher reliability score")
else:
    print(f"  ❌ Our scores do not differentiate by consensus level")

# ── Save ──
df.to_csv(OUTPUT_DIR / "consensus_scores.csv", index=False, encoding="utf-8-sig")
print(f"\nSaved: {OUTPUT_DIR / 'consensus_scores.csv'} ({len(df):,} rows)")

summary = {
    "matched": int(matched),
    "scored": len(df),
    "sahih_score_mean": round(float(df[df["grade_tier"] >= 3]["chain_score"].mean()), 4),
    "hasan_score_mean": round(float(df[df["grade"] == "Hasan"]["chain_score"].mean()), 4) if len(df[df["grade"] == "Hasan"]) > 0 else None,
    "daif_score_mean": round(float(df[df["grade_tier"] <= 2]["chain_score"].mean()), 4) if len(df[df["grade_tier"] <= 2]) > 0 else None,
    "bukhari_muslim_mean": round(float(high["chain_score"].mean()), 4) if len(high) > 0 else None,
    "other_books_mean": round(float(low["chain_score"].mean()), 4) if len(low) > 0 else None,
    "consensus_difference": round(float(diff), 4),
}
with open(OUTPUT_DIR / "consensus_findings.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {OUTPUT_DIR / 'consensus_findings.json'}")

print(f"\n{'=' * 60}")
print(f"  COMPLETE")
print(f"{'=' * 60}")
