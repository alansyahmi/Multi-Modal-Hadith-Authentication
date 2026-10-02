"""
Chain-level reliability scoring on ALL 650K Sanadset hadiths.

Uses GNN + Track A (AraBERT) scores via the 262K-node KG name lookup.

Usage:
  python scripts/evaluate_650k_chains.py

Output:
  output/evaluation_fullscale/
    chain_results.csv       Per-chain scores + grades
    chain_summary.json      Distribution stats
"""

import json, ast, sys, re, time
from collections import Counter, defaultdict
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import partial_ratio

# ── Paths ──
SANADSET_PATH = Path("Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv")
NODE_NAMES = Path("output/stage2/node_names.json")
GNN_PREDICTIONS = Path("output/stage3/narrator_predictions.csv")
OUTPUT_DIR = Path("output/evaluation_fullscale")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Thresholds (from Experiment 8) ──
def score_to_grade(score: float) -> str:
    if score >= 0.85: return "Sahih/Hasan"
    elif score >= 0.70: return "Marginally Accepted"
    elif score >= 0.50: return "Uncertain"
    else: return "Da'if/Rejected"

def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

# ── Build name → score lookup from KG ──
print("Building narrator score lookup...")

# Load node names
with open(NODE_NAMES, encoding="utf-8") as f:
    id_to_name: dict[str, str] = json.load(f)
print(f"  KG narrators: {len(id_to_name):,}")

# Load GNN predictions ({node_id → score})
preds_df = pd.read_csv(GNN_PREDICTIONS)
pred_map: dict[int, float] = {}
for _, row in preds_df.iterrows():
    pred_map[int(row["node_id"])] = float(row["predicted_score"])
print(f"  GNN predictions: {len(pred_map):,}")

# Build normalized name → score dict (exact match)
name_to_score: dict[str, float] = {}
for nid_str, name in id_to_name.items():
    nid = int(nid_str)
    if nid in pred_map:
        name_to_score[norm(name)] = pred_map[nid]

print(f"  Name→score entries: {len(name_to_score):,}")
print(f"  Score range: {min(name_to_score.values()):.3f} – {max(name_to_score.values()):.3f}")

DEFAULT_SCORE = float(np.mean(list(name_to_score.values())))
print(f"  Default (mean) score: {DEFAULT_SCORE:.4f}")

# ── Process 650K chains ──
print(f"\nScoring {SANADSET_PATH.name}...")
total = 0
scored = 0
nerror = 0
results = []

start = time.time()

for chunk_idx, chunk in enumerate(pd.read_csv(
    SANADSET_PATH,
    usecols=["Sanad", "Sanad_Length"],
    chunksize=100_000,
    dtype_backend="pyarrow",
)):
    t0 = time.time()

    for _, row in chunk.iterrows():
        sanad_str = row["Sanad"]
        if pd.isna(sanad_str):
            continue
        try:
            chain = json.loads(sanad_str.replace("'", '"'))
        except:
            try:
                chain = ast.literal_eval(sanad_str)
            except:
                nerror += 1
                continue

        if not isinstance(chain, list) or len(chain) < 2:
            continue

        chain = [n.strip() for n in chain if len(n.strip()) > 3]
        if len(chain) < 2:
            continue

        # Score each narrator in the chain
        narrator_scores = []
        matched = 0
        for name in chain:
            nkey = norm(name)
            if nkey in name_to_score:
                narrator_scores.append(name_to_score[nkey])
                matched += 1
            else:
                narrator_scores.append(DEFAULT_SCORE)

        # Chain score = mean of narrator scores (simpler than multiplicative)
        mean_score = float(np.mean(narrator_scores))

        results.append({
            "chain_length": len(chain),
            "narrators_matched": matched,
            "narrator_count": len(chain),
            "score": round(mean_score, 4),
            "grade": score_to_grade(mean_score),
        })
        total += 1

    elapsed = time.time() - t0
    rate = len(chunk) / elapsed if elapsed > 0 else 0
    print(f"  Chunk {chunk_idx+1}: {len(chunk):,} rows → {total:,} chains ({elapsed:.0f}s, {rate:.0f} rows/s)")

total_time = time.time() - start
print(f"\n  Total: {total:,} chains, {nerror} parse errors, {total_time:.0f}s")
print(f"  Rate: {total/total_time:.0f} chains/sec")

# ── Analysis ──
df = pd.DataFrame(results)

print(f"\n{'=' * 60}")
print(f"  CHAIN-LEVEL EVALUATION: {len(df):,} hadiths")
print(f"{'=' * 60}")

print(f"\nScore distribution:")
print(f"  Mean:  {df['score'].mean():.4f}")
print(f"  Std:   {df['score'].std():.4f}")
print(f"  Min:   {df['score'].min():.4f}")
print(f"  Max:   {df['score'].max():.4f}")

print(f"\nGrade distribution:")
grade_dist = df["grade"].value_counts()
for grade, cnt in grade_dist.items():
    print(f"  {grade:25s}: {cnt:>8,} ({cnt/len(df)*100:.1f}%)")

print(f"\nChain length stats:")
print(f"  Mean: {df['chain_length'].mean():.1f}")
print(f"  Median: {df['chain_length'].median():.0f}")
print(f"  Max: {df['chain_length'].max()}")

# Narrator match rate
avg_match = df["narrators_matched"].sum() / max(df["narrator_count"].sum(), 1)
print(f"\n  Narrator KG coverage: {avg_match*100:.1f}%")

# ── Export ──
df.to_csv(OUTPUT_DIR / "chain_results.csv", index=False, encoding="utf-8-sig")
print(f"\nSaved: {OUTPUT_DIR / 'chain_results.csv'} ({len(df):,} rows)")

# Sample output
samples = df.sample(min(20, len(df)), random_state=42)
with open(OUTPUT_DIR / "chain_samples.txt", "w", encoding="utf-8") as f:
    f.write("Sample Chain Results (random 20)\n")
    f.write("=" * 50 + "\n")
    for _, row in samples.iterrows():
        f.write(f"  length={row['chain_length']:>2}, matched={row['narrators_matched']:>2}/"
                f"{row['narrator_count']:>2}, score={row['score']:.4f}, "
                f"grade={row['grade']}\n")

summary = {
    "total_chains": len(df),
    "parse_errors": nerror,
    "time_seconds": total_time,
    "chains_per_second": round(total / total_time, 1),
    "mean_score": round(float(df["score"].mean()), 4),
    "std_score": round(float(df["score"].std()), 4),
    "min_score": round(float(df["score"].min()), 4),
    "max_score": round(float(df["score"].max()), 4),
    "grade_distribution": {g: int(c) for g, c in grade_dist.items()},
    "avg_chain_length": round(float(df["chain_length"].mean()), 1),
    "kg_coverage_pct": round(avg_match * 100, 1),
}
with open(OUTPUT_DIR / "chain_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {OUTPUT_DIR / 'chain_summary.json'}")

print(f"\n{'=' * 60}")
print(f"  EVALUATION COMPLETE — {len(df):,} hadiths scored")
print(f"{'=' * 60}")
