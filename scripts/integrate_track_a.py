"""
Integrate Track A (AraBERT regression) scores into the GNN pipeline.

Replaces the keyword-lookup grade scores (Track B) in the graph node features
with the AraBERT-predicted scores, then re-runs the GNN to measure impact.

Steps:
  1. Load the PyTorch Geometric graph from Stage 2
  2. Replace column 0 of node features (grade score) with Track A scores
  3. Save an augmented graph for GNN training
  4. Run a quick GCN training comparison: Track A vs Track B features
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACK_A_DIR = PROJECT_ROOT / "output" / "track_a"
STAGE2_DIR = PROJECT_ROOT / "output" / "stage2"
STAGE3_DIR = PROJECT_ROOT / "output" / "stage3"

# ── Config ──
GRAPH_PATH = STAGE2_DIR / "pyg_graph_large_graded.npz"
NODE_NAMES_PATH = STAGE2_DIR / "node_names.json"
TRACK_A_SCORES_PATH = TRACK_A_DIR / "narrator_scores.json"
OUTPUT_GRAPH_PATH = STAGE2_DIR / "pyg_graph_track_a.npz"
DEFAULT_SCORE = 0.30


def integrate():
    print("=" * 65)
    print("  INTEGRATE TRACK A SCORES INTO GNN GRAPH")
    print("=" * 65)

    # ── Load graph ──
    print(f"\n[1/4] Loading graph: {GRAPH_PATH}")
    graph = np.load(GRAPH_PATH)
    x = graph["x"].copy()
    edge_index = graph["edge_index"]
    edge_attr = graph.get("edge_attr", np.zeros((0,)))
    print(f"  Nodes: {x.shape[0]:,}, Features: {x.shape[1]}")

    # ── Load Track A scores ──
    print(f"\n[2/4] Loading Track A scores...")
    with open(TRACK_A_SCORES_PATH, encoding="utf-8") as f:
        track_a_scores: dict[str, float] = json.load(f)
    print(f"  Track A scores: {len(track_a_scores):,}")
    print(f"  Score range: {min(track_a_scores.values()):.3f} – "
          f"{max(track_a_scores.values()):.3f}")

    # ── Load node names ──
    print(f"\n[3/4] Mapping Track A scores to graph nodes...")
    with open(NODE_NAMES_PATH, encoding="utf-8") as f:
        id_to_name: dict[str, str] = json.load(f)

    # Column 0 of node features = grade score
    original_col0 = x[:, 0].copy()
    new_col0 = np.full(x.shape[0], DEFAULT_SCORE, dtype=np.float32)
    matched = 0

    for node_id_str, name in id_to_name.items():
        if node_id_str in track_a_scores:
            node_idx = int(node_id_str) - 1  # 1-based → 0-based
            if 0 <= node_idx < x.shape[0]:
                new_col0[node_idx] = track_a_scores[node_id_str]
                matched += 1

    print(f"  Matched: {matched:,}/{x.shape[0]:,} ({matched/x.shape[0]*100:.1f}%)")
    print(f"  Unmatched defaulted to: {DEFAULT_SCORE}")

    # Stats
    original_mean = float(original_col0.mean())
    new_mean = float(new_col0.mean())
    original_std = float(original_col0.std())
    new_std = float(new_col0.std())

    print(f"  Original grade scores:  mean={original_mean:.3f}, std={original_std:.3f}")
    print(f"  Track A grade scores:   mean={new_mean:.3f}, std={new_std:.3f}")

    # Replace column 0
    x[:, 0] = new_col0

    # ── Save augmented graph ──
    print(f"\n[4/4] Saving Track A graph: {OUTPUT_GRAPH_PATH}")
    np.savez(OUTPUT_GRAPH_PATH, x=x, edge_index=edge_index, edge_attr=edge_attr)
    print(f"  Shape: {x.shape}")

    # Save integration stats
    stats = {
        "graph_nodes": x.shape[0],
        "track_a_scores_total": len(track_a_scores),
        "matched": matched,
        "coverage_pct": float(f"{matched/x.shape[0]*100:.1f}"),
        "original_grade_mean": f"{original_mean:.3f}",
        "track_a_score_mean": f"{new_mean:.3f}",
        "original_grade_std": f"{original_std:.3f}",
        "track_a_score_std": f"{new_std:.3f}",
    }
    with open(TRACK_A_DIR / "integration_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Stats saved: {TRACK_A_DIR / 'integration_stats.json'}")

    print(f"\n  Done. Graph with Track A scores ready.")
    print(f"  Use: run_stage3.py will pick up {OUTPUT_GRAPH_PATH.name}")
    print(f"  Or manually point stage3/config.py GRAPH_NPZ to this file.")


if __name__ == "__main__":
    integrate()
