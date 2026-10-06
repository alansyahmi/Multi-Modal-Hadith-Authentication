"""
GNN Training and Evaluation Pipeline.

Trains supervised GCN/GAT/Fusion models for 5-class narrator reliability
classification using zero-leakage isnād transmission topology.
Evaluates using the locked primary metric: Macro-F1 (PO2).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Optional
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix,
    cohen_kappa_score,
)
from scipy.stats import spearmanr, pearsonr

from .config import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    PATIENCE,
    TRAIN_RATIO,
    VAL_RATIO,
    RANDOM_SEED,
    STAGE3_DIR,
    MODEL_CHECKPOINT,
    PREDICTIONS_CSV,
    EVALUATION_JSON,
    GRAPH_NPZ,
)
from .gnn_model import build_model, CLASS_TO_SCORE, CLASS_NAMES


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_graph(npz_path: Path = GRAPH_NPZ) -> Data:
    """Load graph from Stage 2 PyG .pt file or NumPy archive."""
    if npz_path.suffix == ".pt" or (not npz_path.exists() and npz_path.with_suffix(".pt").exists()):
        pt_path = npz_path if npz_path.suffix == ".pt" else npz_path.with_suffix(".pt")
        return torch.load(pt_path, weights_only=False)

    arrs = np.load(npz_path)
    x = torch.from_numpy(arrs["x"]).float()
    edge_index = torch.from_numpy(arrs["edge_index"]).long()

    # Convert 1-based to 0-based only if needed
    if edge_index.numel() > 0 and edge_index.min() > 0:
        edge_index = edge_index - 1

    edge_attr = torch.from_numpy(arrs["edge_attr"]).float() if "edge_attr" in arrs else None
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    for key in ["y", "y_score", "has_target", "train_mask", "val_mask", "test_mask"]:
        if key in arrs:
            setattr(data, key, torch.from_numpy(arrs[key]))

    return data


@torch.no_grad()
def evaluate_link_prediction(
    model: nn.Module,
    data: Data,
    pos_edge_index: torch.Tensor,
) -> dict[str, float]:
    """Evaluate structural edge compatibility as a diagnostic metric."""
    model.eval()
    if hasattr(model, "encode"):
        z = model.encode(data.x, data.edge_index, getattr(data, "edge_attr", None))
    else:
        out = model(data.x, data.edge_index)
        z = out[1] if isinstance(out, tuple) else out

    # Positive predictions
    pos_pred = torch.sigmoid((z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)).cpu().numpy()

    # Negative sampling
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1),
    )
    neg_pred = torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)).cpu().numpy()

    y_true = np.concatenate([np.ones_like(pos_pred), np.zeros_like(neg_pred)])
    y_pred = np.concatenate([pos_pred, neg_pred])

    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = float(roc_auc_score(y_true, y_pred))
    ap = float(average_precision_score(y_true, y_pred))

    return {"auc": auc, "ap": ap, "n": int(pos_edge_index.size(1))}


def train(
    data: Data,
    model_type: str = "gcn",
    num_epochs: int = NUM_EPOCHS,
    patience: int = PATIENCE,
    verbose: bool = True,
) -> tuple[nn.Module, dict]:
    """Train supervised GNN node classifier for narrator reliability (PO2)."""
    set_seed()

    has_supervised = hasattr(data, "y") and hasattr(data, "train_mask") and hasattr(data, "val_mask") and hasattr(data, "test_mask")
    if not has_supervised:
        raise ValueError(
            "Graph Data object missing required supervised fields ('y', 'train_mask', 'val_mask', 'test_mask').\n"
            "Run Stage 2 (run_stage2.py) to generate clean zero-leakage splits."
        )

    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    y = data.y.long()

    # Compute inverse frequency class weights to address severe class imbalance
    train_labels = y[train_mask].cpu().numpy()
    class_counts = Counter(train_labels)
    max_cnt = max(class_counts.values())
    class_weights = torch.tensor(
        [max_cnt / max(class_counts.get(i, 1), 1) for i in range(5)],
        dtype=torch.float,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Build supervised classifier model
    model = build_model(
        in_channels=data.x.size(1),
        model_type=model_type,
        num_classes=5,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10,
    )

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    if verbose:
        print(f"\n{'='*55}")
        print(f"TRAINING SUPERVISED GNN CLASSIFIER ({model_type.upper()})")
        print(f"{'='*55}")
        print(f"  Nodes: {data.num_nodes:,} | Edges: {data.num_edges:,} | Features: {data.x.size(1)}")
        print(f"  Splits: Train={train_mask.sum().item():,} | Val={val_mask.sum().item():,} | Test={test_mask.sum().item():,}")
        print(f"  Class Weights (inverse-freq): {[round(w, 2) for w in class_weights.tolist()]}")
        print(f"  Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, num_epochs + 1):
        # Training step
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, getattr(data, "edge_attr", None))
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index, getattr(data, "edge_attr", None))
            val_preds = val_out[val_mask].argmax(dim=-1).cpu().numpy()
            val_true = y[val_mask].cpu().numpy()
            val_f1 = float(f1_score(val_true, val_preds, average="macro", zero_division=0))
            val_acc = float(accuracy_score(val_true, val_preds))

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (best Val Macro-F1: {best_val_f1:.4f})")
            break

        if verbose and (epoch % 20 == 0 or epoch == 1):
            train_acc = float((logits[train_mask].argmax(dim=-1) == y[train_mask]).float().mean().item())
            print(f"  Epoch {epoch:>3}: train_loss={loss.item():.4f}, train_acc={train_acc:.4f}, "
                  f"val_acc={val_acc:.4f}, val_f1_macro={val_f1:.4f}")

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Final Test Evaluation (Locked Metric: Macro-F1) ──
    model.eval()
    with torch.no_grad():
        all_logits, z = model(data.x, data.edge_index, getattr(data, "edge_attr", None), return_embeddings=True)
        all_preds = all_logits.argmax(dim=-1).cpu().numpy()

        test_preds = all_preds[test_mask.cpu().numpy()]
        test_true = y[test_mask].cpu().numpy()

        test_f1_macro = float(f1_score(test_true, test_preds, average="macro", zero_division=0))
        test_acc = float(accuracy_score(test_true, test_preds))
        test_kappa = float(cohen_kappa_score(test_true, test_preds, weights="quadratic"))

        # Map to continuous scores for ranking correlation
        y_score_test = data.y_score[test_mask].cpu().numpy() if hasattr(data, "y_score") else np.array([CLASS_TO_SCORE[c] for c in test_true])
        pred_score_test = np.array([CLASS_TO_SCORE[c] for c in test_preds], dtype=np.float32)
        spear_r, _ = spearmanr(y_score_test, pred_score_test)
        pear_r, _ = pearsonr(y_score_test, pred_score_test)

    # Calculate Structural Anomaly Score from embeddings
    src, dst = data.edge_index[0], data.edge_index[1]
    edge_sims = torch.sigmoid((z[src] * z[dst]).sum(dim=-1))
    node_sum = torch.zeros(data.num_nodes, device=z.device)
    node_count = torch.zeros(data.num_nodes, device=z.device)
    node_sum.index_add_(0, src, edge_sims)
    node_count.index_add_(0, src, torch.ones_like(edge_sims))
    node_sum.index_add_(0, dst, edge_sims)
    node_count.index_add_(0, dst, torch.ones_like(edge_sims))
    consistency = node_sum / torch.clamp(node_count, min=1.0)
    consistency[node_count == 0] = 1.0
    anomaly = 1.0 - consistency

    # Continuous score predictions
    predicted_scores = np.array([CLASS_TO_SCORE[c] for c in all_preds], dtype=np.float32)
    ground_truth_scores = data.y_score.cpu().numpy() if hasattr(data, "y_score") else np.full(data.num_nodes, np.nan)

    # Link prediction diagnostic
    link_diag = evaluate_link_prediction(model, data, data.edge_index)

    test_metrics = {
        "primary_metric": "f1_macro",
        "f1_macro": round(test_f1_macro, 4),
        "accuracy": round(test_acc, 4),
        "cohen_kappa": round(test_kappa, 4),
        "spearman_r": round(float(spear_r), 4),
        "pearson_r": round(float(pear_r), 4),
        "confusion_matrix": confusion_matrix(test_true, test_preds).tolist(),
    }

    results = {
        "model_type": model_type,
        "primary_metric": "f1_macro",
        "test": test_metrics,
        "best_val_f1_macro": round(best_val_f1, 4),
        "link_prediction_diagnostic": link_diag,
        "embeddings": z.cpu().numpy().tolist(),
        "predicted_classes": all_preds.tolist(),
        "predicted_scores": predicted_scores.tolist(),
        "anomaly_scores": anomaly.cpu().numpy().tolist(),
        "ground_truth": ground_truth_scores.tolist(),
    }

    if verbose:
        print(f"\n{'='*55}")
        print(f"PO2 EVALUATION RESULTS ({model_type.upper()})")
        print(f"{'='*55}")
        print(f"  Locked Primary Metric (PO2):")
        print(f"    Macro-F1:     {test_f1_macro:.4f}")
        print(f"  Secondary Metrics:")
        print(f"    Accuracy:     {test_acc:.4f}")
        print(f"    Cohen's kappa:{test_kappa:.4f}")
        print(f"    Spearman rho: {spear_r:.4f}")
        print(f"    Pearson r:    {pear_r:.4f}")
        print(f"\n  Classification Report (Test Set):")
        print(classification_report(test_true, test_preds, target_names=CLASS_NAMES, zero_division=0))

    return model, results


def save_results(model: nn.Module, results: dict) -> None:
    """Save model checkpoint, node embeddings, predictions CSV, and evaluation JSON."""
    STAGE3_DIR.mkdir(parents=True, exist_ok=True)

    # Model checkpoint
    torch.save(model.state_dict(), MODEL_CHECKPOINT)
    print(f"  Model checkpoint: {MODEL_CHECKPOINT}")

    # Embeddings
    emb_path = STAGE3_DIR / "narrator_embeddings.npy"
    embeddings = np.array(results["embeddings"], dtype=np.float32)
    np.save(emb_path, embeddings)
    print(f"  Embeddings:       {emb_path} ({embeddings.shape})")

    # Predictions CSV
    preds = results["predicted_scores"]
    pred_classes = results.get("predicted_classes", [0] * len(preds))
    anom = results["anomaly_scores"]
    true = results["ground_truth"]
    out_df = pd.DataFrame({
        "node_id": range(1, len(preds) + 1),
        "predicted_class": pred_classes,
        "predicted_grade": [CLASS_NAMES[c] for c in pred_classes],
        "predicted_score": preds,
        "anomaly_score": anom,
        "consistency_score": [1.0 - a for a in anom],
        "ground_truth_score": true,
    })
    out_df.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8")
    print(f"  Predictions:      {PREDICTIONS_CSV} ({len(out_df):,} rows)")

    # Evaluation summary
    eval_out = {k: v for k, v in results.items() if k not in ("embeddings", "anomaly_scores", "predicted_scores", "predicted_classes", "ground_truth")}
    with open(EVALUATION_JSON, "w", encoding="utf-8") as f:
        json.dump(eval_out, f, indent=2, ensure_ascii=False)
    print(f"  Evaluation JSON:  {EVALUATION_JSON}")
