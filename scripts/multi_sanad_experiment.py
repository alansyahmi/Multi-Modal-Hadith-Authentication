"""
Multi-Sanad Matn Experiment (Full-Scale)
=========================================
Finds matn texts appearing with MULTIPLE different chains across all 650K
hadiths, scores each chain using the 262K-node KG score lookup, and checks
whether independent chains for the same matn converge or diverge.

Usage:
  python scripts/multi_sanad_experiment.py

Output:
  output/multi_sanad_experiment/convergence.csv
  output/multi_sanad_experiment/notable_cases.txt
  output/multi_sanad_experiment/findings.json
"""
import sys, json, re, ast, hashlib, time
from pathlib import Path
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

# ── Config ──
SANADSET_PATH = Path("Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv")
NODE_NAMES_PATH = Path("output/stage2/node_names.json")
GNN_PREDICTIONS = Path("output/stage3/narrator_predictions.csv")
OUTPUT_DIR = Path("output/multi_sanad_experiment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CHAINS = 3

def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

def norm_matn(text: str) -> str:
    if not text or not isinstance(text, str): return ""
    text = re.sub(r"[ً-ٌٍ-َّْٰٓ]", "", text)
    text = text.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    text = text.replace("ى","ي").replace("ة","ه")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("=" * 60)
print("  MULTI-SANAD EXPERIMENT (FULL 650K)")
print("=" * 60)

# ── Build name→score lookup ──
print(f"\n[1/3] Building KG score lookup...")
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
print(f"  {len(name_to_score):,} name→score entries, mean={default_score:.4f}")

# ── Pass 1: Find multi-chain matns ──
print(f"\n[2/3] Scanning 650K for multi-chain matns...")
start = time.time()

# First pass: build matn_hash → [chain_list] (store chain text, not scores yet)
groups = defaultdict(list)
total = 0
nerror = 0

for chunk_idx, chunk in enumerate(pd.read_csv(
    SANADSET_PATH,
    usecols=["Matn", "Sanad"],
    chunksize=100_000,
    dtype_backend="pyarrow",
)):
    t0 = time.time()
    for _, row in chunk.iterrows():
        matn_text = row.get("Matn", "")
        sanad_str = row.get("Sanad", "")
        if pd.isna(matn_text) or pd.isna(sanad_str):
            continue
        matn_key = norm_matn(matn_text)
        if len(matn_key) < 10:
            continue
        h = hashlib.md5(matn_key.encode("utf-8")).hexdigest()

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
        # Store raw chain for later scoring
        groups[h].append(chain)
        total += 1

    print(f"  Chunk {chunk_idx+1}: {total:,} total ({time.time()-t0:.0f}s)")

scan_time = time.time() - start
print(f"\n  Scan: {total:,} hadiths, {len(groups):,} unique matns, {nerror} errors, {scan_time:.0f}s")

# ── Pass 2: Score multi-chain groups ──
print(f"\n[3/3] Scoring {len([h for h,v in groups.items() if len(v)>=MIN_CHAINS]):,} multi-chain matns...")
start = time.time()

rows = []
for h, chains in groups.items():
    if len(chains) < MIN_CHAINS:
        continue

    scores = []
    for chain in chains:
        name_scores = []
        for name in chain:
            n = name.strip()
            if len(n) < 3: continue
            nkey = norm(n)
            score = name_to_score.get(nkey, default_score)
            name_scores.append(score)
        chain_score = float(np.mean(name_scores)) if name_scores else default_score
        scores.append(chain_score)

    mean_s = float(np.mean(scores))
    std_s = float(np.std(scores))
    cv = std_s / max(mean_s, 1e-6)

    rows.append({
        "matn_hash": h,
        "n_chains": len(chains),
        "mean_score": round(mean_s, 4),
        "std_score": round(std_s, 4),
        "cv": round(cv, 4),
        "min_score": round(float(min(scores)), 4),
        "max_score": round(float(max(scores)), 4),
    })

elapsed = time.time() - start
conv_df = pd.DataFrame(rows)
conv_df = conv_df.sort_values("std_score", ascending=False)

print(f"\n  Scored {len(conv_df):,} groups in {elapsed:.0f}s")
print(f"  Total chain-score pairs: {conv_df['n_chains'].sum():,}")

# ── Results ──
print(f"\n{'=' * 60}")
print(f"  CONVERGENCE ANALYSIS")
print(f"{'=' * 60}")
print(f"  Groups:                     {len(conv_df):,}")
print(f"  Mean score (all):           {conv_df['mean_score'].mean():.4f}")
print(f"  Mean std (within groups):   {conv_df['std_score'].mean():.4f}")
print(f"  Mean CV:                    {conv_df['cv'].mean():.4f}")

print(f"\n  Convergence distribution:")
for thresh in [0.001, 0.005, 0.01, 0.02, 0.05]:
    cnt = len(conv_df[conv_df["std_score"] <= thresh])
    print(f"    std ≤ {thresh:.3f}: {cnt:>6,} ({cnt/len(conv_df)*100:.1f}%)")

# Top diverging cases
print(f"\n  Top 5 diverging matns (highest std):")
for _, row in conv_df.head(5).iterrows():
    print(f"    std={row['std_score']:.4f}, n={row['n_chains']}, "
          f"mean={row['mean_score']:.4f}, range=[{row['min_score']:.4f}, {row['max_score']:.4f}]")

# ── Save ──
conv_df.to_csv(OUTPUT_DIR / "convergence.csv", index=False, encoding="utf-8-sig")
print(f"\nSaved: {OUTPUT_DIR / 'convergence.csv'} ({len(conv_df):,} groups)")

# Notable cases
high_var = conv_df.head(10)
low_var = conv_df[conv_df["std_score"] <= conv_df["std_score"].quantile(0.01)].head(10)

with open(OUTPUT_DIR / "notable_cases.txt", "w", encoding="utf-8") as f:
    f.write("MULTI-SANAD EXPERIMENT — NOTABLE CASES\n")
    f.write("=" * 60 + "\n\n")
    f.write("HIGHEST VARIANCE (diverging chains):\n" + "-" * 40 + "\n")
    for _, row in high_var.iterrows():
        f.write(f"  Hash: {row['matn_hash']}\n  Chains: {row['n_chains']}, "
                f"Mean: {row['mean_score']:.4f}, Std: {row['std_score']:.4f}, "
                f"Range: [{row['min_score']:.4f}, {row['max_score']:.4f}]\n\n")
    f.write("\nLOWEST VARIANCE (converging chains):\n" + "-" * 40 + "\n")
    for _, row in low_var.iterrows():
        f.write(f"  Hash: {row['matn_hash']}\n  Chains: {row['n_chains']}, "
                f"Mean: {row['mean_score']:.4f}, Std: {row['std_score']:.4f}\n\n")

print(f"Saved: {OUTPUT_DIR / 'notable_cases.txt'}")

findings = {
    "total_hadiths_scanned": total,
    "parse_errors": nerror,
    "unique_matns": len(groups),
    "matns_with_multi_chains": len(conv_df),
    "multi_chain_pairs": int(conv_df["n_chains"].sum()),
    "convergence": {
        "mean_score_all": round(float(conv_df["mean_score"].mean()), 4),
        "mean_std_within": round(float(conv_df["std_score"].mean()), 4),
        "mean_cv": round(float(conv_df["cv"].mean()), 4),
        "pct_std_below_0_01": round(len(conv_df[conv_df["std_score"] <= 0.01]) / max(len(conv_df), 1) * 100, 1),
        "pct_std_below_0_005": round(len(conv_df[conv_df["std_score"] <= 0.005]) / max(len(conv_df), 1) * 100, 1),
    },
}
with open(OUTPUT_DIR / "findings.json", "w") as f:
    json.dump(findings, f, indent=2)
print(f"Saved: {OUTPUT_DIR / 'findings.json'}")

print(f"\n{'=' * 60}")
print(f"  MULTI-SANAD EXPERIMENT COMPLETE")
print(f"  {len(conv_df):,} matns with ≥{MIN_CHAINS} chains analyzed")
print(f"{'=' * 60}")
