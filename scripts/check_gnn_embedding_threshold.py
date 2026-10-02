"""
Check GNN embedding threshold: find the 5th percentile of
pairwise similarities between KNOWN reliable teacher-student pairs.
"""
import sys, re, json
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","a").replace("ة","ه").replace("ى","ي").strip()

# Load embeddings
embeddings = np.load("output/stage3/narrator_embeddings.npy")
print(f"Embeddings: {embeddings.shape}")

# Load node names
with open("output/stage2/node_names.json", encoding="utf-8") as f:
    id_to_name = json.load(f)

# Load KG edges (known teacher-student pairs)
edges = pd.read_csv("output/stage2/graph_edges.csv")
print(f"KG edges: {len(edges):,}")

# Sample edges and compute embedding similarities
SAMPLE = min(10000, len(edges))
sampled = edges.sample(SAMPLE, random_state=42)

similarities = []
for _, row in sampled.iterrows():
    src, tgt = int(row["source_id"]), int(row["target_id"])
    if 1 <= src-1 < embeddings.shape[0] and 1 <= tgt-1 < embeddings.shape[0]:
        z1 = embeddings[src-1]
        z2 = embeddings[tgt-1]
        dot = np.sum(z1 * z2)
        sim = 1.0 / (1.0 + np.exp(-dot))
        similarities.append(sim)

sims = np.array(similarities)
print(f"\nSampled {len(sims):,} known teacher-student pairs")
print(f"  Mean: {sims.mean():.4f}")
print(f"  Median: {np.median(sims):.4f}")
print(f"  Std: {sims.std():.4f}")
print(f"\n  Distribution:")
for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"    P{q:>2d}: {np.percentile(sims, q):.4f}")

# Recommended threshold = 1st percentile (only 1% of real pairs will be penalized)
p1 = np.percentile(sims, 1)
p5 = np.percentile(sims, 5)
print(f"\n  Recommended threshold: {p1:.4f} (P1, very strict)")
print(f"  Conservative threshold: {p5:.4f} (P5, safe)")
print(f"  Current threshold (0.25): {np.mean(sims < 0.25)*100:.1f}% of known pairs would be penalized")
