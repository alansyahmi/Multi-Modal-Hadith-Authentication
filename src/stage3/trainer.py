"""
GNN Training and Evaluation Pipeline.

Trains a GCN/GAT encoder via Graph Autoencoder (GAE) / Link Prediction
to learn structural embeddings, edge compatibility, and structural anomaly scores.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

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
from .gnn_model import build_model


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_graph(npz_path: Path = GRAPH_NPZ) -> Data:
    """Load graph from Stage 2 NumPy archive."""
    if npz_path.suffix == ".pt":
        return torch.load(npz_path, weights_only=False)

    arrs = np.load(npz_path)
    x = torch.from_numpy(arrs["x"]).float()
    edge_index = torch.from_numpy(arrs["edge_index"]).long()

    # Convert from 1-based (CSV) to 0-based (PyG)
    edge_index = edge_index - 1

    edge_attr = torch.from_numpy(arrs["edge_attr"]).float() if "edge_attr" in arrs else None

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


@torch.no_grad()
def evaluate_link_prediction(
    model: nn.Module,
    data: Data,
    pos_edge_index: torch.Tensor,
) -> dict[str, float]:
    """Evaluate link prediction on a given set of positive edges."""
    model.eval()
    z = model(data.x, data.edge_index)

    # Positive predictions
    pos_pred = torch.sigmoid((z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)).cpu().numpy()

    # Sample negative edges
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


def train_epoch(
    model: nn.Module,
    data: Data,
    train_edge_index: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Single training epoch using link prediction loss."""
    model.train()
    optimizer.zero_grad()

    z = model(data.x, data.edge_index)

    # Positive scores
    pos_pred = torch.sigmoid((z[train_edge_index[0]] * z[train_edge_index[1]]).sum(dim=-1))

    # Negative sampling
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=train_edge_index.size(1),
    )
    neg_pred = torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1))

    # BCE Loss
    loss = -torch.log(pos_pred + 1e-15).mean() - torch.log(1.0 - neg_pred + 1e-15).mean()
    loss.backward()
    optimizer.step()

    return loss.item()


def train(
    data: Data,
    model_type: str = "gcn",
    num_epochs: int = NUM_EPOCHS,
    patience: int = PATIENCE,
    verbose: bool = True,
) -> tuple[nn.Module, dict]:
    """Train GNN as a representation learner (GAE / Link Prediction)."""
    set_seed()

    # Split edges for evaluation
    num_edges = data.edge_index.size(1)
    indices = torch.randperm(num_edges)
    n_val = int(num_edges * VAL_RATIO)
    n_test = int(num_edges * (1 - TRAIN_RATIO - VAL_RATIO))
    n_train = num_edges - n_val - n_test

    train_edge_index = data.edge_index[:, indices[:n_train]]
    val_edge_index = data.edge_index[:, indices[n_train:n_train + n_val]]
    test_edge_index = data.edge_index[:, indices[n_train + n_val:]]

    # Build model (encoder)
    model = build_model(
        in_channels=data.x.size(1),
        model_type=model_type,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_ap = 0.0
    best_state = None
    patience_counter = 0

    if verbose:
        print(f"\n{'='*50}")
        print(f"TRAINING STRUCTURAL ENCODER ({model_type.upper()})")
        print(f"{'='*50}")
        print(f"  Nodes: {data.num_nodes:,}")
        print(f"  Edges: {data.num_edges:,}")
        print(f"  Features: {data.x.size(1)}")
        print(f"  Train Edges: {train_edge_index.size(1):,}")
        print(f"  Val Edges:   {val_edge_index.size(1):,}")
        print(f"  Test Edges:  {test_edge_index.size(1):,}")
        print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, data, train_edge_index, optimizer)
        val_metrics = evaluate_link_prediction(model, data, val_edge_index)
        val_ap = val_metrics["ap"]

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

        if verbose and epoch % 25 == 0:
            print(f"  Epoch {epoch:>4}: train_loss={loss:.4f}, "
                  f"val_auc={val_metrics['auc']:.4f}, val_ap={val_ap:.4f}")

    # Restore best state
    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation
    test_metrics = evaluate_link_prediction(model, data, test_edge_index)
    all_metrics = evaluate_link_prediction(model, data, data.edge_index)

    # Compute node embeddings
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)

    # Calculate Structural Anomaly Score
    # anomaly = 1 - average connection compatibility with neighbors
    src, dst = data.edge_index[0], data.edge_index[1]
    edge_sims = torch.sigmoid((z[src] * z[dst]).sum(dim=-1))

    node_sum = torch.zeros(data.num_nodes, device=z.device)
    node_count = torch.zeros(data.num_nodes, device=z.device)
    node_sum.index_add_(0, src, edge_sims)
    node_count.index_add_(0, src, torch.ones_like(edge_sims))
    node_sum.index_add_(0, dst, edge_sims)
    node_count.index_add_(0, dst, torch.ones_like(edge_sims))

    consistency = node_sum / torch.clamp(node_count, min=1.0)
    consistency[node_count == 0] = 1.0  # default for isolated nodes
    anomaly = 1.0 - consistency

    # Downstream readout: Train KNeighborsRegressor on structural embeddings of graded nodes
    from sklearn.neighbors import KNeighborsRegressor
    
    sent_csv = Path("output/track_a_classifier/narrator_scores_sentiment.csv")
    if sent_csv.exists():
        print(f"  Loading sentiment-based reliability scores for downstream GNN propagation...")
        sent_df = pd.read_csv(sent_csv)
        base_scores = np.full(data.num_nodes, 0.632, dtype=np.float32)
        has_grade = np.zeros(data.num_nodes, dtype=bool)
        for _, row in sent_df.iterrows():
            nid = int(row["node_id"])
            idx = nid - 1
            if 0 <= idx < data.num_nodes and row["critic_count"] > 0:
                base_scores[idx] = float(row["reliability_score"])
                has_grade[idx] = True
        print(f"    Priors loaded for {has_grade.sum():,} nodes.")
    else:
        base_scores = data.x[:, 0].numpy()
        has_grade = (data.x[:, 5] > 0.5).numpy() if data.x.size(1) > 5 else (base_scores > 0.0)
    
    z_np = z.cpu().numpy()
    X_train = z_np[has_grade]
    y_train = base_scores[has_grade]
    
    if len(y_train) > 0:
        knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
        knn.fit(X_train, y_train)
        predicted_base = knn.predict(z_np)
        predicted_base = np.clip(predicted_base, 0.0, 1.0)
    else:
        predicted_base = np.where(has_grade, base_scores, 0.632)
        
    anomaly_np = anomaly.cpu().numpy()
    fused_scores = predicted_base * (1.0 - 0.25 * anomaly_np)

    results = {
        "model_type": model_type,
        "test": test_metrics,
        "all": all_metrics,
        "best_val_ap": best_val_ap,
        "embeddings": z.cpu().numpy().tolist(),
        "anomaly_scores": anomaly_np.tolist(),
        "predicted_scores": fused_scores.tolist(),
        "ground_truth": base_scores.tolist(),
    }

    if verbose:
        print(f"\n=== Test Results (Link Prediction) ===")
        print(f"  ROC-AUC: {test_metrics['auc']:.4f}")
        print(f"  Avg Prec: {test_metrics['ap']:.4f}")

    return model, results


def save_results(model: nn.Module, results: dict) -> None:
    """Save model checkpoint, node embeddings, and anomaly/reliability scores."""
    STAGE3_DIR.mkdir(parents=True, exist_ok=True)

    # Model checkpoint
    torch.save(model.state_dict(), MODEL_CHECKPOINT)
    print(f"  Model: {MODEL_CHECKPOINT}")

    # Save raw embeddings as numpy array
    emb_path = STAGE3_DIR / "narrator_embeddings.npy"
    embeddings = np.array(results["embeddings"], dtype=np.float32)
    np.save(emb_path, embeddings)
    print(f"  Embeddings: {emb_path} ({embeddings.shape})")

    # Save predictions CSV
    preds = results["predicted_scores"]
    anom = results["anomaly_scores"]
    true = results["ground_truth"]
    out_df = pd.DataFrame({
        "node_id": range(1, len(preds) + 1),
        "predicted_score": preds,
        "anomaly_score": anom,
        "consistency_score": [1.0 - a for a in anom],
        "ground_truth": true,
    })
    out_df.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8")
    print(f"  Predictions: {PREDICTIONS_CSV} ({len(out_df):,} rows)")

    # Evaluation summary
    eval_out = {k: v for k, v in results.items() if k not in ("embeddings", "anomaly_scores", "predicted_scores", "ground_truth")}
    with open(EVALUATION_JSON, "w", encoding="utf-8") as f:
        json.dump(eval_out, f, indent=2, ensure_ascii=False)
    print(f"  Evaluation: {EVALUATION_JSON}")
