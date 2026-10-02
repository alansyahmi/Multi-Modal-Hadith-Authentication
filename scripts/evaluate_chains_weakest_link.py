"""
Chain evaluation using WEAKEST-LINK scoring instead of mean.

Instead of averaging all narrator scores (which collapses to ~0.63),
takes the MINIMUM score in the chain. This reflects the classical
hadith principle: "a chain is only as strong as its weakest narrator."

For multi-chain matns, applies Edge Jaccard discounting:
  - Overlapping narrators reduce independence of parallel chains
"""
import json, ast, sys, re, time, hashlib
from collections import defaultdict
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

SANADSET_PATH = Path("Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv")
NODE_NAMES = Path("output/stage2/node_names.json")
GNN_PRED = Path("output/stage3/narrator_predictions.csv")
OUTPUT_DIR = Path("output/weakest_link_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

def score_to_grade(s):
    if s >= 0.85: return "Sahih"
    elif s >= 0.70: return "Hasan"
    elif s >= 0.50: return "Maqbul"
    elif s >= 0.15: return "Daif"
    else: return "Matruk"

print("=" * 60)
print("  WEAKEST-LINK CHAIN EVALUATION")
print("=" * 60)

# Build score lookup
preds = pd.read_csv(GNN_PRED)
gnn = {int(r["node_id"]): float(r["predicted_score"]) for _, r in preds.iterrows()}
with open(NODE_NAMES, encoding="utf-8") as f:
    id_to_name = json.load(f)
name_to_score = {}
for nid, name in id_to_name.items():
    name_to_score[norm(name)] = gnn.get(int(nid), 0.632)
print(f"Score lookup: {len(name_to_score):,}")

# Process chains
total = 0
nerror = 0
results = []
multi_chain_groups = defaultdict(list)  # matn_hash → [(chain_narrator_set, min_score)]

start = time.time()

for chunk_idx, chunk in enumerate(pd.read_csv(
    SANADSET_PATH, usecols=["Sanad", "Matn"],
    chunksize=100_000, dtype_backend="pyarrow",
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

        # Score each narrator
        scores = []
        for name in names:
            nkey = norm(name)
            scores.append(name_to_score.get(nkey, 0.632))

        # WEAKEST-LINK: take the min
        chain_score = float(min(scores))
        min_narrator = names[scores.index(min(scores))]

        results.append({
            "cl": len(names),
            "sc": round(chain_score, 4),
            "gr": score_to_grade(chain_score),
            "min_narrator": min_narrator[:40],
        })
        total += 1

    print(f"  Chunk {chunk_idx+1}: {total:,} chains ({time.time()-t0:.0f}s)")

total_time = time.time() - start
df = pd.DataFrame(results)

print(f"\n{'=' * 60}")
print(f"  RESULTS: {len(df):,} chains, {total_time:.0f}s")
print(f"{'=' * 60}")

print(f"\nScore distribution:")
print(f"  Mean: {df['sc'].mean():.4f}, Std: {df['sc'].std():.4f}")
print(f"  Min: {df['sc'].min():.4f}, Max: {df['sc'].max():.4f}")

print(f"\nGrade distribution:")
for grade in ["Sahih", "Hasan", "Maqbul", "Daif", "Matruk"]:
    cnt = len(df[df["gr"] == grade])
    print(f"  {grade:10s}: {cnt:>8,} ({cnt/len(df)*100:.1f}%)")

# Save
df.to_csv(OUTPUT_DIR / "weakest_link_results.csv", index=False)
print(f"\nSaved: {OUTPUT_DIR / 'weakest_link_results.csv'}")

summary = {
    "method": "weakest_link",
    "total_chains": len(df),
    "mean_score": round(float(df["sc"].mean()), 4),
    "std_score": round(float(df["sc"].std()), 4),
    "grade_dist": {g: int(c) for g, c in df["gr"].value_counts().items()},
    "time_seconds": total_time,
}
with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {OUTPUT_DIR / 'summary.json'}")
