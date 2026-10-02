"""
Build a larger knowledge graph with more Sanadset chains for denser topology.
"""
import json, ast, sys, os
from collections import defaultdict
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

# Load Sanadset (sample 100K rows for a much denser graph)
NROWS = 100_000
print(f"Loading {NROWS:,} Sanadset rows...")
sanadset = pd.read_csv(
    r"output/stage1/sanadset_cleaned.csv",
    nrows=NROWS,
)
print(f"  Columns: {list(sanadset.columns)}")

# Build graph: extract all narrator chains
nodes: dict[str, int] = {}
edges_raw = []

for idx, row in sanadset.iterrows():
    if idx % 20000 == 0:
        print(f"  Processing row {idx:,}/{NROWS:,}... ({len(nodes):,} narrators, {len(edges_raw):,} edges)")

    try:
        chain = ast.literal_eval(row["Sanad"]) if isinstance(row["Sanad"], str) else []
    except (ValueError, SyntaxError):
        continue

    if len(chain) < 2:
        continue

    # Trim very long chains
    chain = chain[:30]

    for i in range(len(chain) - 1):
        src = chain[i].strip()
        tgt = chain[i + 1].strip()
        if not src or not tgt or len(src) < 3 or len(tgt) < 3:
            continue

        # Assign node IDs
        if src not in nodes:
            nodes[src] = len(nodes) + 1
        if tgt not in nodes:
            nodes[tgt] = len(nodes) + 1

        edges_raw.append((nodes[src], nodes[tgt], i))

print(f"\nGraph built:")
print(f"  Nodes:       {len(nodes):,}")
print(f"  Raw edges:   {len(edges_raw):,}")

# Aggregate edges
edge_buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
for sid, tid, pos in edges_raw:
    edge_buckets[(sid, tid)].append(pos)

print(f"  Unique edges: {len(edge_buckets):,}")

# Build node features
n = len(nodes)
feats = np.zeros((n, 8), dtype=np.float32)

# Compute in/out degree and position stats
id_to_name = {v: k for k, v in nodes.items()}
in_deg = defaultdict(int)
out_deg = defaultdict(int)
appearances = defaultdict(int)
pos_sum = defaultdict(float)
pos_count = defaultdict(int)

for (sid, tid), positions in edge_buckets.items():
    freq = len(positions)
    out_deg[sid] += freq
    in_deg[tid] += freq
    appearances[sid] += freq
    appearances[tid] += freq
    for p in positions:
        pos_sum[tid] += p
        pos_count[tid] += 1

# Populate features
for node_id in range(1, n + 1):
    idx = node_id - 1
    feats[idx, 1] = np.log1p(appearances[node_id])
    feats[idx, 2] = np.log1p(in_deg[node_id])
    feats[idx, 3] = np.log1p(out_deg[node_id])
    if pos_count[node_id] > 0:
        feats[idx, 4] = pos_sum[node_id] / pos_count[node_id]
    feats[idx, 5] = 0.0  # has_grade — will fill if available
    feats[idx, 6] = 0.0
    feats[idx, 7] = 0.0

# Normalize columns 1-4
for col in [1, 2, 3, 4]:
    vals = feats[:, col]
    vmin, vmax = vals.min(), vals.max()
    if vmax - vmin > 1e-8:
        feats[:, col] = (vals - vmin) / (vmax - vmin)

# Build edge_index
edge_list = list(edge_buckets.keys())
edge_index = np.array([[s, t] for s, t in edge_list], dtype=np.int64).T  # [2, E]
edge_attr = np.array([
    [0.5, np.mean(positions) / 30.0, np.log1p(len(positions))]
    for (sid, tid), positions in edge_buckets.items()
], dtype=np.float32)

print(f"\nFeatures: {feats.shape}")
print(f"Edge index: {edge_index.shape}")
print(f"Edge attr: {edge_attr.shape}")

# Save
out = Path(r"output/stage2/pyg_graph_large.npz")
np.savez(out, x=feats, edge_index=edge_index, edge_attr=edge_attr)
print(f"\nSaved: {out}")

# Also save node name mapping
name_map_path = Path(r"output/stage2/node_names.json")
with open(name_map_path, "w", encoding="utf-8") as f:
    json.dump(id_to_name, f, ensure_ascii=False)
print(f"Saved: {name_map_path}")

print(f"\nDone — {len(nodes):,} nodes, {len(edge_list):,} edges")
