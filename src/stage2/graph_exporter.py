"""
Graph Exporter — serializes the Knowledge Graph to PyTorch Geometric format.

Produces a torch_geometric.data.Data object with:
  - x: Node feature matrix [N, F_node]
  - edge_index: Adjacency [2, E]
  - edge_attr: Edge features [E, F_edge]
  - y: Optional node labels (for supervised learning in Stage 3)

Node features (dim=8):
  0: grade_score         (Ibn Hajar reliability, 0-1)
  1: total_appearances   (normalized log)
  2: in_degree           (normalized log)
  3: out_degree          (normalized log)
  4: avg_chain_position  (normalized)
  5: has_grade           (binary: 1=annotated, 0=unknown)
  6: has_death_info      (binary)
  7: has_tabaqa          (binary)

Edge features (dim=4):
  0: method_code          (0=direct, 1=ambiguous, 2=unknown)
  1: position             (normalized position in chain)
  2: frequency            (normalized log)
  3: log_frequency
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import STAGE2_DIR, GRAPH_NODES_CSV, GRAPH_EDGES_CSV, GRAPH_PYG


def _log_norm(values: np.ndarray) -> np.ndarray:
    """Log-normalize: log1p then min-max scale to [0, 1]."""
    logged = np.log1p(values.astype(np.float64))
    mi, ma = logged.min(), logged.max()
    if ma - mi < 1e-8:
        return np.zeros_like(logged)
    return ((logged - mi) / (ma - mi)).astype(np.float32)


def build_node_features(nodes_df: pd.DataFrame) -> np.ndarray:
    """Build node feature matrix from node DataFrame."""
    n = len(nodes_df)
    feats = np.zeros((n, 8), dtype=np.float32)

    # 0: grade_score (already 0-1)
    feats[:, 0] = nodes_df["grade_score"].fillna(0.30).values.astype(np.float32)

    # 1: total_appearances (log-normalized)
    feats[:, 1] = _log_norm(nodes_df["total_appearances"].fillna(0).values)

    # 2: in_degree
    feats[:, 2] = _log_norm(nodes_df["in_degree"].fillna(0).values)

    # 3: out_degree
    feats[:, 3] = _log_norm(nodes_df["out_degree"].fillna(0).values)

    # 4: avg_chain_position
    pos = nodes_df["avg_chain_position"].fillna(0).values.astype(np.float64)
    pos_max = pos.max()
    feats[:, 4] = ((pos / pos_max) if pos_max > 0 else pos).astype(np.float32)

    # 5: has_grade
    feats[:, 5] = (nodes_df["grade"].fillna("").str.len() > 0).astype(np.float32)

    # 6: has_death_info
    feats[:, 6] = (nodes_df["death_info"].fillna("").str.len() > 0).astype(np.float32)

    # 7: has_tabaqa
    feats[:, 7] = (nodes_df["tabaqa"].fillna("").str.len() > 0).astype(np.float32)

    return feats


def build_edge_features(edges_df: pd.DataFrame) -> np.ndarray:
    """Build edge feature matrix from edge DataFrame."""
    n = len(edges_df)
    feats = np.zeros((n, 3), dtype=np.float32)

    # 0: method_code (already 0/1/2)
    feats[:, 0] = edges_df["method_code"].fillna(2).values.astype(np.float32) / 2.0

    # 1: position (normalized, 0=collector end)
    pos = edges_df["position"].fillna(0).values.astype(np.float64)
    pos_max = pos.max()
    feats[:, 1] = ((pos / pos_max) if pos_max > 0 else pos).astype(np.float32)

    # 2: frequency (log-normalized)
    feats[:, 2] = _log_norm(edges_df["frequency"].fillna(1).values)

    return feats


def export_to_pyg(
    nodes_path: Path = GRAPH_NODES_CSV,
    edges_path: Path = GRAPH_EDGES_CSV,
    output_path: Path = GRAPH_PYG,
) -> Optional[Path]:
    """
    Build and save a torch_geometric.data.Data object.

    Falls back to NumPy .npz if torch/torch_geometric are not installed.
    """
    print("\n=== EXPORTING TO GRAPH FORMAT ===")

    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    print(f"  Nodes: {len(nodes_df):,}")
    print(f"  Edges: {len(edges_df):,}")

    # Build features
    x = build_node_features(nodes_df)
    edge_attr = build_edge_features(edges_df)

    # Build edge_index [2, E]
    edge_index = np.array([
        edges_df["source_id"].values,
        edges_df["target_id"].values,
    ], dtype=np.int64)

    print(f"  Node features: {x.shape}")
    print(f"  Edge index:    {edge_index.shape}")
    print(f"  Edge features: {edge_attr.shape}")

    # Try PyTorch Geometric
    try:
        import torch
        from torch_geometric.data import Data

        data = Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_attr),
        )
        torch.save(data, output_path)
        print(f"  Saved PyG Data: {output_path}")
        return output_path

    except ImportError:
        print("  torch_geometric not installed — saving as NumPy .npz")
        npz_path = output_path.with_suffix(".npz")
        np.savez(
            npz_path,
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        print(f"  Saved NumPy archive: {npz_path}")
        return npz_path


def write_metadata(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> None:
    """Write graph metadata JSON with schema documentation."""
    meta = {
        "num_nodes": len(nodes_df),
        "num_edges": len(edges_df),
        "node_feature_dim": 8,
        "edge_feature_dim": 3,
        "node_features": [
            "grade_score (Ibn Hajar 0-1)",
            "total_appearances (log-norm)",
            "in_degree (log-norm)",
            "out_degree (log-norm)",
            "avg_chain_position (norm)",
            "has_grade (binary)",
            "has_death_info (binary)",
            "has_tabaqa (binary)",
        ],
        "edge_features": [
            "method_code (0=direct, 1=ambiguous, scaled to 0-1)",
            "position (norm, 0=collector end)",
            "frequency (log-norm)",
        ],
        "grade_distribution": nodes_df["grade"].value_counts().to_dict(),
        "avg_grade_score": round(float(nodes_df["grade_score"].mean()), 4),
        "nodes_with_grade": int((nodes_df["grade"].notna() & (nodes_df["grade"] != "")).sum()),
        "nodes_with_death": int((nodes_df["death_info"].notna() & (nodes_df["death_info"] != "")).sum()),
    }
    meta_path = STAGE2_DIR / "graph_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Metadata: {meta_path}")
