"""
Hybrid chain evaluation: AraBERT (Track A) for narrators with bio text,
GNN predictions for the rest. Then score all 650K chains.

Usage:
  python scripts/score_and_evaluate_chains.py

Output:
  output/final_evaluation/chain_results.csv
  output/final_evaluation/summary.json
"""
import json, ast, sys, re, time
from pathlib import Path
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

SANADSET_PATH = Path("Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv")
NODE_NAMES_PATH = Path("output/stage2/node_names.json")
GNN_PREDICTIONS = Path("output/stage3/narrator_predictions.csv")
ALIAS_PATH = Path("output/shamela/alias_index.json")
ENCYCLOPEDIA_PATH = Path("output/shamela/narrator_encyclopedia.json")
MODEL_DIR = Path("output/track_a/arabert_regression_model")
OUTPUT_DIR = Path("output/final_evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128  # Larger batch for name-only inference (short sequences)
MAX_LENGTH = 128

def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

def score_to_grade(s):
    if s >= 0.85: return "Sahih/Hasan"
    elif s >= 0.70: return "Marginally Accepted"
    elif s >= 0.50: return "Uncertain"
    else: return "Daif"

print("=" * 60)
print("  HYBRID CHAIN EVALUATION")
print("=" * 60)

# ── Step 1: Build narrator score lookup ──
print(f"\n[1/4] Building narrator score lookup...")

# Load GNN predictions (fallback)
preds_df = pd.read_csv(GNN_PREDICTIONS)
gnn_scores = {int(row["node_id"]): float(row["predicted_score"]) for _, row in preds_df.iterrows()}

# Load node names
with open(NODE_NAMES_PATH, encoding="utf-8") as f:
    id_to_name = json.load(f)

# Build normalized name → id mapping
name_to_id = {}
for nid, name in id_to_name.items():
    name_to_id[norm(name)] = int(nid)

print(f"  GNN scores: {len(gnn_scores):,}")
print(f"  KG names:   {len(name_to_id):,}")

# ── Step 2: Score narrators with bio text using AraBERT ──
print(f"\n[2/4] AraBERT scoring for narrators with bio text...")

# Load alias + encyclopedia
alias_index = json.load(open(ALIAS_PATH, encoding="utf-8"))
enc = json.load(open(ENCYCLOPEDIA_PATH, encoding="utf-8"))
entry_text = {}
for entry in enc:
    eid, text = entry.get("entry_id"), entry.get("full_text", "")
    if eid and text and len(text) > 30:
        entry_text[eid] = text[:2000]

# Build name → bio text mapping from alias
name_to_bio = {}
for alias_key, eids in alias_index.items():
    eid = eids[0]
    if eid in entry_text:
        name_to_bio[alias_key] = entry_text[eid]
print(f"  Name→bio text: {len(name_to_bio):,} entries")

# Load AraBERT model
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(DEVICE)
model.eval()
print(f"  AraBERT model loaded")

# Score narrators that have bio text
all_names = list(name_to_id.keys())
arabert_scores = {}
scored_count = 0

with torch.no_grad():
    for i in tqdm(range(0, len(all_names), BATCH_SIZE), desc="AraBERT scoring"):
        batch = all_names[i:i + BATCH_SIZE]
        texts = [name_to_bio.get(n, n) for n in batch]  # Bio text if available, else name
        has_bio = [n in name_to_bio for n in batch]

        encoded = tokenizer(texts, truncation=True, padding=True,
                           max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
        outputs = model(**encoded)
        scores = torch.sigmoid(outputs.logits).squeeze(-1).cpu().numpy()

        for j, (name, hb) in enumerate(zip(batch, has_bio)):
            node_id = name_to_id[name]
            if hb:
                arabert_scores[node_id] = float(scores[j])
                scored_count += 1

print(f"  AraBERT-scored narrators: {scored_count:,}/{len(all_names):,} ({scored_count/len(all_names)*100:.1f}%)")

# ── Step 3: Hybrid lookupscore = AraBERT if available, else GNN ──
print(f"\n[3/4] Building hybrid score lookup...")
hybrid_scores = {}
for node_id in range(1, len(id_to_name) + 1):
    if node_id in arabert_scores:
        hybrid_scores[node_id] = arabert_scores[node_id]
    elif node_id in gnn_scores:
        hybrid_scores[node_id] = gnn_scores[node_id]
    else:
        hybrid_scores[node_id] = 0.632  # Global mean

# Normalized name → score
name_to_score = {}
for nid, name in id_to_name.items():
    nkey = norm(name)
    node_id = int(nid)
    name_to_score[nkey] = hybrid_scores.get(node_id, 0.632)

vals = list(name_to_score.values())
print(f"  Hybrid scores: {len(name_to_score):,}")
print(f"  Range: {min(vals):.3f} – {max(vals):.3f}, Mean: {np.mean(vals):.3f}, Std: {np.std(vals):.3f}")

# ── Step 4: Score all 650K chains ──
print(f"\n[4/4] Scoring 650K chains...")

default = float(np.mean(vals))
total = 0
nerror = 0
results = []
start = time.time()

for chunk_idx, chunk in enumerate(pd.read_csv(
    SANADSET_PATH, usecols=["Sanad"],
    chunksize=100_000, dtype_backend="pyarrow",
)):
    t0 = time.time()
    for _, row in chunk.iterrows():
        sanad = row["Sanad"]
        if pd.isna(sanad):
            continue
        try:
            chain = json.loads(sanad.replace("'", '"'))
        except:
            try:
                chain = ast.literal_eval(sanad)
            except:
                nerror += 1
                continue
        if not isinstance(chain, list) or len(chain) < 2:
            continue
        names = [n.strip() for n in chain if len(n.strip()) > 3]
        if len(names) < 2:
            continue

        scores = [name_to_score.get(norm(n), default) for n in names]
        mean_score = float(np.mean(scores))
        results.append({
            "cl": len(names),
            "sc": round(mean_score, 4),
            "gr": score_to_grade(mean_score),
        })
        total += 1

    print(f"  Chunk {chunk_idx+1}: {total:,} chains ({time.time()-t0:.0f}s)")

total_time = time.time() - start
print(f"\n  Processed: {total:,} chains, {nerror} errors, {total_time:.0f}s")
print(f"  Rate: {total/total_time:.0f} chains/sec")

# ── Analysis ──
df = pd.DataFrame(results)
print(f"\n{'=' * 60}")
print(f"  RESULTS: {len(df):,} chains")
print(f"{'=' * 60}")

print(f"\nScore distribution:")
print(f"  Mean: {df['sc'].mean():.4f}, Std: {df['sc'].std():.4f}")
print(f"  Min: {df['sc'].min():.4f}, Max: {df['sc'].max():.4f}")

print(f"\nGrade distribution:")
for grade in ["Sahih/Hasan", "Marginally Accepted", "Uncertain", "Daif"]:
    cnt = len(df[df["gr"] == grade])
    print(f"  {grade:25s}: {cnt:>8,} ({cnt/len(df)*100:.1f}%)")

# Chain length
print(f"\nChain length: mean={df['cl'].mean():.1f}, median={df['cl'].median():.0f}, max={df['cl'].max()}")

# ── Save ──
df.to_csv(OUTPUT_DIR / "chain_results.csv", index=False, encoding="utf-8-sig")
print(f"\nSaved: {OUTPUT_DIR / 'chain_results.csv'} ({len(df):,} rows)")

summary = {
    "total_chains": len(df),
    "parse_errors": nerror,
    "time_seconds": total_time,
    "chains_per_second": round(total / total_time, 1),
    "mean_score": round(float(df["sc"].mean()), 4),
    "std_score": round(float(df["sc"].std()), 4),
    "min_score": round(float(df["sc"].min()), 4),
    "max_score": round(float(df["sc"].max()), 4),
    "grade_dist": {g: int(c) for g, c in df["gr"].value_counts().items()},
    "avg_chain_length": round(float(df["cl"].mean()), 1),
    "arabert_coverage_pct": round(scored_count / len(all_names) * 100, 1),
}
with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {OUTPUT_DIR / 'summary.json'}")

print(f"\n{'=' * 60}")
print(f"  EVALUATION COMPLETE")
print(f"{'=' * 60}")
