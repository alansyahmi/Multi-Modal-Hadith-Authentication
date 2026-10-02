"""
Build a full-scale Knowledge Graph from ALL 650K Sanadset chains.
Uses the original sanadset.csv directly with chunked loading.
"""
import json, ast, sys, time, gc
from collections import defaultdict
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

# ── Config ──
SANADSET_PATH = r"Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv"
CHUNK_SIZE = 50_000  # rows per chunk to manage memory
MAX_CHAIN = 30
OUTPUT_DIR = Path(r"output/stage2_full")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  FULL-SCALE KG: 650K Sanadset Chains")
print("=" * 60)
print(f"  Source: {SANADSET_PATH}")
print(f"  Chunk size: {CHUNK_SIZE:,}")

total_bytes = Path(SANADSET_PATH).stat().st_size
print(f"  File size: {total_bytes / 1e9:.2f} GB")

# ── Pass 1: Build all nodes + edges ──
print(f"\n=== Pass 1: Extracting narrators and edges ===")
start = time.time()

nodes: dict[str, int] = {}
edges_raw: list[tuple[int, int, int]] = []
total_rows = 0

for chunk_idx, chunk in enumerate(pd.read_csv(
    SANADSET_PATH,
    usecols=["Sanad", "Sanad_Length"],
    chunksize=CHUNK_SIZE,
    dtype_backend="pyarrow",
)):
    t0 = time.time()
    chunk_rows = 0

    for _, row in chunk.iterrows():
        sanad_str = row["Sanad"]
        if pd.isna(sanad_str) or sanad_str == "No SANAD":
            continue

        # json.loads after quote replacement is ~5x faster than ast.literal_eval
        try:
            chain = json.loads(sanad_str.replace("'", '"'))
        except (json.JSONDecodeError, ValueError):
            try:
                chain = ast.literal_eval(sanad_str)
            except (ValueError, SyntaxError):
                continue

        if not isinstance(chain, list) or len(chain) < 2:
            continue

        chain = [n.strip() for n in chain[:MAX_CHAIN]]

        for i in range(len(chain) - 1):
            src, tgt = chain[i], chain[i + 1]
            if len(src) < 3 or len(tgt) < 3:
                continue
            if src not in nodes:
                nodes[src] = len(nodes) + 1
            if tgt not in nodes:
                nodes[tgt] = len(nodes) + 1
            edges_raw.append((nodes[src], nodes[tgt], i))

        chunk_rows += 1

    total_rows += chunk_rows
    elapsed = time.time() - t0
    pct = total_rows / 650_000 * 100
    print(f"  Chunk {chunk_idx+1}: {chunk_rows:,} chains, "
          f"{len(nodes):,} narrators, {len(edges_raw):,} edges "
          f"({elapsed:.0f}s, {pct:.1f}%)")

    del chunk
    gc.collect()

pass1_time = time.time() - start
print(f"\n=== Pass 1 complete: {total_rows:,} chains processed ===")
print(f"  Narrators: {len(nodes):,}")
print(f"  Raw edges: {len(edges_raw):,}")
print(f"  Time: {pass1_time:.0f}s")

# ── Aggregate edges ──
print(f"\n=== Aggregating edges ===")
edge_buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
for sid, tid, pos in edges_raw:
    edge_buckets[(sid, tid)].append(pos)

print(f"  Unique edges: {len(edge_buckets):,}")

del edges_raw
gc.collect()

# ── Build node features ──
print(f"\n=== Building node features ===")
n = len(nodes)
feats = np.zeros((n, 8), dtype=np.float32)

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

for node_id in range(1, n + 1):
    idx = node_id - 1
    feats[idx, 1] = np.log1p(appearances[node_id])
    feats[idx, 2] = np.log1p(in_deg[node_id])
    feats[idx, 3] = np.log1p(out_deg[node_id])
    if pos_count[node_id] > 0:
        feats[idx, 4] = pos_sum[node_id] / pos_count[node_id]
    feats[idx, 5] = 0.3  # default grade
    feats[idx, 6] = 0.0
    feats[idx, 7] = 0.0

for col in [1, 2, 3, 4]:
    vals = feats[:, col]
    vmin, vmax = vals.min(), vals.max()
    if vmax - vmin > 1e-8:
        feats[:, col] = (vals - vmin) / (vmax - vmin)

edge_list = list(edge_buckets.keys())
edge_index = np.array([[s, t] for s, t in edge_list], dtype=np.int64).T
edge_attr = np.array([
    [0.5, np.mean(positions) / 30.0, np.log1p(len(positions))]
    for (sid, tid), positions in edge_buckets.items()
], dtype=np.float32)

print(f"\n  Features:     {feats.shape}")
print(f"  Edge index:   {edge_index.shape}")
print(f"  Edge attr:    {edge_attr.shape}")

# ── Save ──
print(f"\n=== Saving ===")
out = OUTPUT_DIR / "pyg_graph_full.npz"
np.savez(out, x=feats, edge_index=edge_index, edge_attr=edge_attr)
print(f"  Saved: {out}")

name_map_path = OUTPUT_DIR / "node_names.json"
with open(name_map_path, "w", encoding="utf-8") as f:
    json.dump(id_to_name, f, ensure_ascii=False)
print(f"  Saved: {name_map_path}")

meta = {
    "n_chains": total_rows,
    "n_nodes": n,
    "n_edges": len(edge_list),
    "n_features": 8,
    "pass1_time_seconds": pass1_time,
}
with open(OUTPUT_DIR / "graph_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"  Saved: metadata")

total_time = time.time() - start
print(f"\n{'=' * 60}")
print(f"  FULL-SCALE KG COMPLETE")
print(f"  {n:,} narrators, {len(edge_list):,} edges")
print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"{'=' * 60}")
