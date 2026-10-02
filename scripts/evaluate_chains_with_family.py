"""
Chain evaluation with family graph resolution for "أبيه", "أخيه", etc.

When a chain has "... → سفيان → أبيه → ...", it resolves "أبيه" to
سفيان's father using the father-son graph, then uses the father's score.

Usage:
  python scripts/evaluate_chains_with_family.py
"""
import json, ast, sys, re, time
from collections import defaultdict
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

SANADSET_PATH = Path("Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv")
NODE_NAMES_PATH = Path("output/stage2/node_names.json")
GNN_PREDICTIONS = Path("output/stage3/narrator_predictions.csv")
FATHER_SON_PATH = Path("output/family_graph/father_son_resolved.json")
VIRTUAL_NARRATORS_PATH = Path("output/family_graph/virtual_narrators.json")
OUTPUT_DIR = Path("output/final_evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

# Relational terms in sanad
ABIH_RE = re.compile(r"أبيه|أبوه|أبى|والده")
AKHIH_RE = re.compile(r"أخيه|أخوه|أخاه")
AMMIH_RE = re.compile(r"عمه|عمه|عمى")
JADDIH_RE = re.compile(r"جده|جدى")

print("=" * 60)
print("  CHAIN EVAL WITH FAMILY RESOLUTION")
print("=" * 60)

# Build narrator score lookup
print("\n[1] Building narrator scores...")
preds_df = pd.read_csv(GNN_PREDICTIONS)
gnn = {int(row["node_id"]): float(row["predicted_score"]) for _, row in preds_df.iterrows()}
with open(NODE_NAMES_PATH, encoding="utf-8") as f:
    id_to_name = json.load(f)
name_to_id = {norm(v): int(k) for k, v in id_to_name.items()}
score_lookup = {}
for nid, name in id_to_name.items():
    score_lookup[norm(name)] = gnn.get(int(nid), 0.632)
print(f"  Scores: {len(score_lookup):,}")

# Load father-son graph
father_son = json.load(open(FATHER_SON_PATH, encoding="utf-8"))
son_to_father = {int(k): v for k, v in father_son.items()}
virtual = json.load(open(VIRTUAL_NARRATORS_PATH, encoding="utf-8"))
print(f"  Father-son pairs: {len(son_to_father):,}")
print(f"  Virtual narrators: {len(virtual):,}")

# Build GNN score for fathers
virtual_scores = {}
for fname, data in virtual.items():
    son_ids = data["son_entry_ids"]
    sons_in_kg = []
    for sid in son_ids:
        son_entry = None
        for nkey, nid in name_to_id.items():
            if str(sid) in nkey:  # Crude: check if entry_id appears in normalized name
                break
        # Use the encyclopedia to look up actual entry name
        pass
    # Average score of sons
    virtual_scores[norm(fname)] = 0.632  # Default for now

# Load encyclopedia for entry → name mapping
enc = json.load(open("output/shamela/narrator_encyclopedia.json", encoding="utf-8"))
entry_name = {e["entry_id"]: e.get("name", "") for e in enc}

# Build entry_id → score lookup (for resolved fathers)
entry_to_score = {}
for e in enc:
    eid = e["entry_id"]
    name = e.get("name", "")
    nkey = norm(name)
    if nkey in score_lookup:
        entry_to_score[eid] = score_lookup[nkey]

# Resolve virtual narrator scores from sons
for fname, data in virtual.items():
    sons = data["son_entry_ids"]
    son_scores = [entry_to_score.get(sid) for sid in sons if sid in entry_to_score]
    if son_scores:
        virtual_scores[norm(fname)] = float(np.mean(son_scores))
    else:
        virtual_scores[norm(fname)] = 0.632

print(f"  Entry→scores: {len(entry_to_score):,}")
print(f"  Virtual scores: {len(virtual_scores):,}")

# Score chains with family resolution
print("\n[2] Scoring 650K chains (with family resolution)...")

# Patterns to extract father fragment from narrator names
FATHER_FRAG_RE = re.compile(r"(?:بن|ابن)\s+(.*)")

def resolve_narrator(name: str, prev_name: str = "") -> str:
    """Resolve a narrator name, handling family references."""
    nkey = norm(name)

    # Direct match in KG
    if nkey in score_lookup:
        return name

    # "أبيه" / "أخيه" / "عمه" — resolve via previous narrator
    if prev_name:
        pkey = norm(prev_name)

        if re.search(r"أبيه|أبوه|أبى|والده", name):
            # Find father of prev_name
            for pkey_candidate, fid in son_to_father.items():
                # Check if prev_name's entry_id matches
                pass  # TODO: integrate with entry_id matching

        elif re.search(r"أخيه|أخوه", name):
            pass  # Brother resolution via shared teacher

    return name  # Fallback: use as-is

# Process chains
total = 0
nerror = 0
family_resolved = 0
results = []
start = time.time()

for chunk_idx, chunk in enumerate(pd.read_csv(
    SANADSET_PATH, usecols=["Sanad"],
    chunksize=100_000, dtype_backend="pyarrow",
)):
    t0 = time.time()
    for _, row in chunk.iterrows():
        sanad = row["Sanad"]
        if pd.isna(sanad): continue
        try:
            chain = json.loads(sanad.replace("'", '"'))
        except:
            try: chain = ast.literal_eval(sanad)
            except: nerror += 1; continue
        if not isinstance(chain, list) or len(chain) < 2: continue

        names = [n.strip() for n in chain if len(n.strip()) > 3]
        if len(names) < 2: continue

        # Score with family resolution
        prev_name = ""
        narrator_scores = []
        for name in names:
            nkey = norm(name)
            resolved = False

            # If it's a relational term like "أبيه"
            if prev_name and re.search(r"أبيه|أبوه$|أبى|والده", name):
                # Find father of prev narrator
                pkey = norm(prev_name)
                # Try to match prev_name to an entry with known father
                for nid_key, score in score_lookup.items():
                    if pkey in nid_key:
                        for entry_nkey, eid in [(norm(en), eid) for eid, en in entry_name.items()]:
                            if entry_nkey == nid_key and eid in son_to_father:
                                father_id = son_to_father[eid]
                                if father_id in entry_to_score:
                                    narrator_scores.append(entry_to_score[father_id])
                                    family_resolved += 1
                                    resolved = True
                                    break
                        if resolved:
                            break

            if not resolved:
                if nkey in score_lookup:
                    narrator_scores.append(score_lookup[nkey])
                else:
                    narrator_scores.append(0.632)

            prev_name = name

        chain_score = float(np.mean(narrator_scores))
        results.append({"cl": len(names), "sc": round(chain_score, 4)})
        total += 1

    print(f"  Chunk {chunk_idx+1}: {total:,} chains, {family_resolved:,} family refs resolved ({time.time()-t0:.0f}s)")

total_time = time.time() - start
df = pd.DataFrame(results)
print(f"\nResults: {len(df):,} chains, {family_resolved:,} family refs ({total_time:.0f}s)")
print(f"Score: mean={df['sc'].mean():.4f}, std={df['sc'].std():.4f}")

df.to_csv(OUTPUT_DIR / "chain_results_family.csv", index=False)
print(f"Saved: {OUTPUT_DIR / 'chain_results_family.csv'}")
