"""
Graph Exporter — serializes the Knowledge Graph to PyTorch Geometric and NumPy formats.

ZERO TARGET LEAKAGE DESIGN:
All ground-truth reliability grades (Ibn Hajar annotations) are strictly isolated
into target label vectors (y, y_score) and evaluation masks (train_mask, val_mask, test_mask).
They are NEVER included in the input feature matrix X.

Input Node Features (dim=8, pure topological + chronological metadata):
  0: total_appearances   (normalized log)
  1: in_degree           (normalized log — citations as student/target)
  2: out_degree          (normalized log — citations as teacher/source)
  3: net_flow            (out_deg - in_deg) / (out_deg + in_deg + 1.0) -> [-1, 1]
  4: avg_chain_position  (normalized [0, 1] from early narrator to collector)
  5: has_death_info      (binary flag: 1=recorded, 0=unknown)
  6: has_tabaqa          (binary flag: 1=recorded, 0=unknown)
  7: tabaqa_num          (normalized [0, 1] generation index: 1st=0.083 ... 12th=1.0)

Ground Truth Labels (isolated in y / masks):
  y_score: Continuous reliability score (0.0 to 1.0) or NaN for unlabeled nodes
  y_class: Ordinal 5-class index (0: matruk, 1: daif, 2: maqbul, 3: saduq, 4: thiqah, -1: unlabeled)
  has_target: Boolean mask of nodes with verified ground truth
  train_mask / val_mask / test_mask: Stratified 70/15/15 split on annotated nodes (fixed seed=42)

Edge Features (dim=3):
  0: method_code          (0=direct, 0.5=ambiguous, 1=unknown)
  1: position             (normalized position in chain)
  2: frequency            (normalized log)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import STAGE2_DIR, GRAPH_NODES_CSV, GRAPH_EDGES_CSV, GRAPH_PYG

RANDOM_SEED = 42

CLASS_TO_SCORE = {0: 0.05, 1: 0.15, 2: 0.40, 3: 0.70, 4: 0.85}

TABAQA_ORDER = {
    "أول": 1, "اول": 1, "الأول": 1,
    "ثاني": 2, "الثاني": 2,
    "ثالث": 3, "الثالث": 3,
    "رابع": 4, "الرابع": 4,
    "خامس": 5, "الخامس": 5,
    "سادس": 6, "السادس": 6,
    "سابع": 7, "السابع": 7,
    "ثامن": 8, "الثامن": 8,
    "تاسع": 9, "التاسع": 9,
    "عاشر": 10, "العاشر": 10, "عا": 10,
    "حادي": 11, "الحادي": 11,
    "ثاني عشر": 12, "الثاني عشر": 12,
}


def _log_norm(values: np.ndarray) -> np.ndarray:
    """Log-normalize: log1p then min-max scale to [0, 1]."""
    logged = np.log1p(values.astype(np.float64))
    mi, ma = logged.min(), logged.max()
    if ma - mi < 1e-8:
        return np.zeros_like(logged, dtype=np.float32)
    return ((logged - mi) / (ma - mi)).astype(np.float32)


def _parse_tabaqa(t_str: object) -> float:
    """Extract and normalize generation index to [0.0, 1.0]."""
    if not isinstance(t_str, str) or not t_str.strip():
        return 0.0
    for term, val in sorted(TABAQA_ORDER.items(), key=lambda x: len(x[0]), reverse=True):
        if term in t_str:
            return round(val / 12.0, 4)
    return 0.0


def build_node_features(nodes_df: pd.DataFrame) -> np.ndarray:
    """
    Build clean node feature matrix from node DataFrame.
    STRICT ZERO TARGET LEAKAGE:
    Contains ONLY graph topology and historical/chronological indicators.
    """
    n = len(nodes_df)
    feats = np.zeros((n, 8), dtype=np.float32)

    # 0: total_appearances (log-normalized)
    feats[:, 0] = _log_norm(nodes_df["total_appearances"].fillna(0).values)

    # 1: in_degree (student citations, log-normalized)
    in_deg = nodes_df["in_degree"].fillna(0).values.astype(np.float32)
    feats[:, 1] = _log_norm(in_deg)

    # 2: out_degree (teacher citations, log-normalized)
    out_deg = nodes_df["out_degree"].fillna(0).values.astype(np.float32)
    feats[:, 2] = _log_norm(out_deg)

    # 3: net_flow (-1 to +1): relative position in transmission chain
    feats[:, 3] = ((out_deg - in_deg) / (out_deg + in_deg + 1.0)).astype(np.float32)

    # 4: avg_chain_position (normalized [0, 1])
    pos = nodes_df["avg_chain_position"].fillna(0).values.astype(np.float64)
    pos_max = pos.max()
    feats[:, 4] = ((pos / pos_max) if pos_max > 0 else pos).astype(np.float32)

    # 5: has_death_info (binary flag)
    feats[:, 5] = (nodes_df["death_info"].fillna("").str.strip().str.len() > 0).astype(np.float32)

    # 6: has_tabaqa (binary flag)
    feats[:, 6] = (nodes_df["tabaqa"].fillna("").str.strip().str.len() > 0).astype(np.float32)

    # 7: tabaqa_num (normalized generation index [0.0, 1.0])
    feats[:, 7] = nodes_df["tabaqa"].apply(_parse_tabaqa).values.astype(np.float32)

    return feats


def build_node_labels(
    nodes_df: pd.DataFrame,
    seed: int = RANDOM_SEED,
    cohort_path: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Isolate ground truth reliability into target arrays and stratified train/val/test masks.

    If cohort_path is provided or exists at `output/hadith_viz_match/frozen_evaluation_cohort.csv`,
    uses the canonical multi-critic consensus ground truth classes and locked 70/15/15 splits.
    """
    n = len(nodes_df)
    y_score = np.full(n, np.nan, dtype=np.float32)
    y_class = np.full(n, -1, dtype=np.int64)
    has_target = np.zeros(n, dtype=bool)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    # Check for canonical frozen cohort
    default_cohort = Path(__file__).resolve().parent.parent.parent / "output/hadith_viz_match/frozen_evaluation_cohort.csv"
    target_cohort = cohort_path if cohort_path is not None else (default_cohort if default_cohort.exists() else None)

    if target_cohort and target_cohort.exists():
        print(f"  [Ground Truth] Loading canonical frozen evaluation cohort: {target_cohort}")
        cohort_df = pd.read_csv(target_cohort)
        for _, row in cohort_df.iterrows():
            nid = int(row["node_id"])
            idx = nid - 1  # 0-based indexing
            if 0 <= idx < n:
                cls_val = int(row["ground_truth_class"])
                split = str(row["split"]).strip()
                if split in ("train", "val", "test") and cls_val >= 0:
                    has_target[idx] = True
                    y_class[idx] = cls_val
                    y_score[idx] = CLASS_TO_SCORE.get(cls_val, 0.5)
                    if split == "train":
                        train_mask[idx] = True
                    elif split == "val":
                        val_mask[idx] = True
                    elif split == "test":
                        test_mask[idx] = True

        print(f"  [Cohort Loaded] Target nodes: {has_target.sum():,} | Train={train_mask.sum():,} | Val={val_mask.sum():,} | Test={test_mask.sum():,}")
        return y_score, y_class, has_target, train_mask, val_mask, test_mask

    # Fallback to local Taqrīb grades if cohort file is absent
    for i, row in nodes_df.iterrows():
        grade_str = str(row.get("grade", "")).strip()
        raw_score = row.get("grade_score", np.nan)
        if pd.notna(raw_score) and grade_str and grade_str != "nan":
            score = float(raw_score)
            y_score[i] = score
            has_target[i] = True

            # 5-class ordinal mapping (consistent with Track A AraBERT)
            if score >= 0.85:
                y_class[i] = 4  # thiqah
            elif score >= 0.60:
                y_class[i] = 3  # saduq
            elif score >= 0.35:
                y_class[i] = 2  # maqbul
            elif score >= 0.15:
                y_class[i] = 1  # daif
            else:
                y_class[i] = 0  # matruk

    target_indices = np.where(has_target)[0]
    target_classes = y_class[target_indices]

    if len(target_indices) >= 20:
        # Check minimum class count for stratified split
        _, counts = np.unique(target_classes, return_counts=True)
        can_stratify = bool(counts.min() >= 2)

        stratify_labels = target_classes if can_stratify else None
        train_idx, temp_idx = train_test_split(
            target_indices,
            test_size=0.30,
            random_state=seed,
            stratify=stratify_labels,
        )

        temp_stratify = y_class[temp_idx] if can_stratify else None
        if can_stratify:
            _, c_temp = np.unique(temp_stratify, return_counts=True)
            if c_temp.min() < 2:
                temp_stratify = None

        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            random_state=seed,
            stratify=temp_stratify,
        )

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
    else:
        train_mask[target_indices] = True

    return y_score, y_class, has_target, train_mask, val_mask, test_mask


def build_edge_features(edges_df: pd.DataFrame) -> np.ndarray:
    """Build edge feature matrix from edge DataFrame."""
    n = len(edges_df)
    feats = np.zeros((n, 3), dtype=np.float32)

    # 0: method_code (already 0/1/2) -> scaled to [0, 1]
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
    cohort_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Build and save a leakage-free torch_geometric.data.Data object and NumPy archive.
    """
    print("\n=== EXPORTING LEAKAGE-FREE GRAPH ===")

    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    print(f"  Nodes: {len(nodes_df):,}")
    print(f"  Edges: {len(edges_df):,}")

    # Build clean features and isolated labels
    x = build_node_features(nodes_df)
    edge_attr = build_edge_features(edges_df)
    y_score, y_class, has_target, train_mask, val_mask, test_mask = build_node_labels(
        nodes_df, cohort_path=cohort_path
    )

    # Build edge_index [2, E] (1-based to 0-based indexing)
    edge_index = np.array([
        edges_df["source_id"].values - 1,
        edges_df["target_id"].values - 1,
    ], dtype=np.int64)

    print(f"  Clean node features (X): {x.shape} (ZERO target leakage)")
    print(f"  Edge index:              {edge_index.shape}")
    print(f"  Edge features:           {edge_attr.shape}")
    print(f"  Annotated targets:       {has_target.sum():,} / {len(nodes_df):,}")
    print(f"  Stratified Split:        Train={train_mask.sum():,} | Val={val_mask.sum():,} | Test={test_mask.sum():,}")

    # Always save clean NumPy archive
    npz_path = output_path.with_suffix(".npz")
    np.savez(
        npz_path,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y_class,
        y_score=y_score,
        has_target=has_target,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    print(f"  Saved clean NumPy archive: {npz_path}")

    # Try PyTorch Geometric Data object
    try:
        import torch
        from torch_geometric.data import Data

        data = Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_attr),
            y=torch.from_numpy(y_class),
            y_score=torch.from_numpy(y_score),
            has_target=torch.from_numpy(has_target),
            train_mask=torch.from_numpy(train_mask),
            val_mask=torch.from_numpy(val_mask),
            test_mask=torch.from_numpy(test_mask),
        )
        torch.save(data, output_path)
        print(f"  Saved clean PyG Data: {output_path}")
        return output_path

    except ImportError:
        print("  torch_geometric not installed — NumPy .npz output complete.")
        return npz_path


def write_metadata(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> None:
    """Write graph metadata JSON with complete schema documentation and leakage audit status."""
    y_score, y_class, has_target, train_mask, val_mask, test_mask = build_node_labels(nodes_df)

    meta = {
        "num_nodes": len(nodes_df),
        "num_edges": len(edges_df),
        "target_leakage_status": "VERIFIED_ZERO_LEAKAGE (grade scores excluded from X)",
        "node_feature_dim": 8,
        "edge_feature_dim": 3,
        "node_features": [
            "total_appearances (log-norm)",
            "in_degree (log-norm, student citations)",
            "out_degree (log-norm, teacher citations)",
            "net_flow ((out-in)/(out+in+1), directional chain position)",
            "avg_chain_position (normalized [0, 1])",
            "has_death_info (binary flag)",
            "has_tabaqa (binary flag)",
            "tabaqa_num (normalized generation index 1/12..12/12)",
        ],
        "edge_features": [
            "method_code (0=direct, 0.5=ambiguous, 1=unknown)",
            "position (normalized chain position)",
            "frequency (log-norm transmission weight)",
        ],
        "target_distribution": {
            "annotated_nodes": int(has_target.sum()),
            "unannotated_nodes": int((~has_target).sum()),
            "classes": {
                "0_matruk": int((y_class == 0).sum()),
                "1_daif": int((y_class == 1).sum()),
                "2_maqbul": int((y_class == 2).sum()),
                "3_saduq": int((y_class == 3).sum()),
                "4_thiqah": int((y_class == 4).sum()),
            },
            "splits": {
                "train_count": int(train_mask.sum()),
                "val_count": int(val_mask.sum()),
                "test_count": int(test_mask.sum()),
            }
        },
        "grade_distribution": nodes_df["grade"].value_counts().to_dict(),
        "avg_grade_score": round(float(nodes_df["grade_score"].dropna().mean()), 4),
    }
    meta_path = STAGE2_DIR / "graph_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Metadata: {meta_path}")
