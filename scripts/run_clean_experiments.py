"""
Clean Controlled Experiments Engine for FYP2.

Executes the audited 13-arm experiment matrix across 25 paired cross-validation folds:
  - 5-fold Stratified Cross-Validation x 5 Random Seeds (Seeds: [42, 123, 456, 789, 2024]).
  - One canonical narrator identity per row; outer train/test row partitions are disjoint.
  - Target labels are not input features. Text uses the configured date-verified critic allowlist and snippet-level compiler substring filter.
  - Graph arms are transductive: all cohort nodes and graph edges are present during message passing in every fold.
  - Target label: Ibn Hajar's canonical 3-tier reliability grade.
  - Consistent inverse-frequency class weighting across all arms.
  - Paired evaluation: Every arm is trained and evaluated on the exact same 25 fold partitions.

Experimental Arms:
  1. B0: Majority Baseline (Floor Accuracy)
  2. B1: Stratified Random Baseline (Chance Macro-F1 Floor)
  3. B2: Metadata / Name Probe (LR on Name tokens + biographical metadata)
  4. P1-A: Lexical TF-IDF Baseline + Logistic Regression (PO1 lexical)
  5. P1-B: Contextual Semantic NLP - AraBERT (768-dim) + Logistic Regression (PO1 semantic)
  6. P1-B_MLP: Contextual Semantic NLP - AraBERT (768-dim) + 2-layer MLP
  7. G1: Non-GNN Graph Features (5 Topo + 3 Meta) + 2-layer MLP (no edges)
  8. G2: Graph Message Passing (5 Topo + 3 Meta + edges) + 2-layer GCN
  9. P2-A: Early Multimodal Fusion (Text + 8 Tabular) + 2-layer MLP (no edges) [Tests H_1a]
  10. P2-B_Concat: Full Multimodal GNN (Text + 8 Tabular + edges) [Tests H_1b]
  11. P2-B_Attn: Multimodal GNN with Gated Attention (Text + 8 Tabular + edges)
  12. P2-C: Topology Isolation (Text + 5 Topo) + 2-layer MLP (no edges)
  13. P2-D: Metadata Isolation (Text + 3 Meta) + 2-layer MLP (no edges)

Outputs:
  - `output/cleaned_cohort/clean_experiments_results.json` (Full 25-fold granular data)
  - `output/cleaned_cohort/clean_experiments_summary.json` (Summary stats, CIs, paired tests)
  - `FYP2/clean_experiments_report.md` (Formal academic report ready for Chapter 6)
"""

import os
import sys
import json
import time
import math
from pathlib import Path
from collections import Counter

# Set console output encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, confusion_matrix
from scipy import stats
from src.evaluation_stats import corrected_repeated_cv_summary, corrected_repeated_cv_test

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 456, 789, 2024]
N_SPLITS = 5
NUM_CLASSES = 3
CLASS_NAMES = ["below_saduq", "saduq", "thiqah"]

# ── Neural Network Modules ──

class TabularMLP(nn.Module):
    """2-layer MLP classifier for tabular or concatenated features."""
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index=None):
        h = F.relu(self.fc1(x))
        h = self.drop(h)
        return self.fc2(h)

class PureGCN(nn.Module):
    """2-layer Graph Convolutional Network."""
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.drop(h)
        return self.conv2(h, edge_index)

class MultimodalGNNConcat(nn.Module):
    """Multimodal GNN fusing graph topological propagation and text semantics via concatenation."""
    def __init__(self, graph_dim: int = 8, text_dim: int = 768, hidden_dim: int = 64, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.graph_conv = GCNConv(graph_dim, hidden_dim)
        self.text_linear = nn.Linear(text_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fusion_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_graph, edge_index, x_text):
        h_g = F.relu(self.graph_conv(x_graph, edge_index))
        h_t = F.relu(self.text_linear(x_text))
        h_fused = torch.cat([h_g, h_t], dim=-1)
        h = F.relu(self.fusion_fc(h_fused))
        h = self.drop(h)
        return self.classifier(h)

class MultimodalGNNAttention(nn.Module):
    """Multimodal GNN fusing graph topology and text semantics via cross-modal gated attention."""
    def __init__(self, graph_dim: int = 8, text_dim: int = 768, hidden_dim: int = 64, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.graph_conv = GCNConv(graph_dim, hidden_dim)
        self.text_linear = nn.Linear(text_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_graph, edge_index, x_text):
        h_g = F.relu(self.graph_conv(x_graph, edge_index))
        h_t = F.relu(self.text_linear(x_text))
        combined = torch.cat([h_g, h_t], dim=-1)
        alpha = self.gate(combined)
        h_fused = alpha * h_g + (1.0 - alpha) * h_t
        h = self.drop(h_fused)
        return self.classifier(h)

# ── Training Helpers for Neural Models ──

def train_neural_model(
    model: nn.Module,
    forward_fn,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    y: torch.Tensor,
    class_weights: torch.Tensor,
    epochs: int = 80,
    lr: float = 0.005,
    weight_decay: float = 1e-4,
    patience: int = 20,
    return_probabilities: bool = False,
):
    """Train with inner validation early stopping and return test predictions.

    When requested, also return the softmax probability matrix in class-index
    order. The default preserves the historical prediction-only interface.
    """
    model = model.to(DEVICE)
    y_dev = y.to(DEVICE)
    weights_dev = class_weights.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights_dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = -1.0
    best_weights = None
    patience_cnt = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = forward_fn(model)
        loss = criterion(logits[train_idx], y_dev[train_idx])
        loss.backward()
        optimizer.step()

        # Inner validation check
        model.eval()
        with torch.no_grad():
            eval_logits = forward_fn(model)
            val_preds = eval_logits[val_idx].argmax(dim=-1).cpu().numpy()
            val_true = y[val_idx].numpy()
            val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        final_logits = forward_fn(model)
        test_logits = final_logits[test_idx]
        test_preds = test_logits.argmax(dim=-1).cpu().numpy()
        test_probabilities = torch.softmax(test_logits, dim=-1).cpu().numpy()

    if return_probabilities:
        return test_preds, test_probabilities
    return test_preds

# ── Metric Computation ──

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, is_consensus: np.ndarray = None) -> dict:
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    res = {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "quadratic_kappa": kappa,
        "f1_below_saduq": float(per_class_f1[0]) if len(per_class_f1) > 0 else 0.0,
        "f1_saduq": float(per_class_f1[1]) if len(per_class_f1) > 1 else 0.0,
        "f1_thiqah": float(per_class_f1[2]) if len(per_class_f1) > 2 else 0.0,
    }

    if is_consensus is not None and np.sum(is_consensus) > 0:
        c_true = y_true[is_consensus == 1]
        c_pred = y_pred[is_consensus == 1]
        res["consensus_macro_f1"] = float(f1_score(c_true, c_pred, average="macro", zero_division=0))
        res["consensus_accuracy"] = float(accuracy_score(c_true, c_pred))
    else:
        res["consensus_macro_f1"] = macro_f1
        res["consensus_accuracy"] = acc

    return res

# ── Main Experiment Runner ──

def main():
    print("=" * 80)
    print("FYP2 CLEAN CONTROLLED EXPERIMENTS: 13-ARM EVALUATION MATRIX")
    print(f"Device: {DEVICE} | 5 Seeds x 5 Folds = 25 Paired Evaluations")
    print("=" * 80)

    # 1. Load Data
    cohort_csv = ROOT / "output/cleaned_cohort/clean_narrators_cohort.csv"
    graph_npz = ROOT / "output/cleaned_cohort/clean_graph.npz"
    texts_json = ROOT / "output/cleaned_cohort/clean_texts.json"
    emb_npy = ROOT / "output/cleaned_cohort/clean_arabert_embeddings.npy"

    df_cohort = pd.read_csv(cohort_csv)
    npz_data = np.load(graph_npz)
    with open(texts_json, "r", encoding="utf-8") as f:
        texts_dict = json.load(f)
    x_arabert_raw = np.load(emb_npy)

    n_nodes = len(df_cohort)
    print(f"Loaded cohort: {n_nodes} unique narrator identities.")

    y = npz_data["y"]  # (335,)
    is_consensus = npz_data["is_consensus"]  # (335,)
    edge_index = npz_data["edge_index"]      # (2, 354)
    x_topo_raw = npz_data["x_topo"]          # (335, 5)
    x_meta_raw = npz_data["x_meta"]          # (335, 3)

    # Prepare raw text list ordered by identity_id 0..334
    id_to_text = {}
    for entry in texts_dict.values():
        id_to_text[int(entry["identity_id"])] = entry["earlier_critic_text"]
    raw_texts = [id_to_text[i].strip() if id_to_text[i].strip() else "لا توجد ترجمة" for i in range(n_nodes)]
    canonical_names = df_cohort["canonical_name"].tolist()

    y_tensor = torch.from_numpy(y).long()
    edge_index_tensor = torch.from_numpy(edge_index).long().to(DEVICE)
    x_arabert_tensor = torch.from_numpy(x_arabert_raw).float().to(DEVICE)

    # Arms to benchmark
    ARM_NAMES = [
        "B0 (Majority Baseline)",
        "B1 (Stratified Random)",
        "B2 (Name + Meta Probe)",
        "P1-A (Lexical TF-IDF LR)",
        "P1-B (AraBERT Semantic LR)",
        "P1-B_MLP (AraBERT Semantic MLP)",
        "G1 (Graph Features MLP)",
        "G2 (Graph GCN)",
        "P2-A (Early Fusion MLP)",
        "P2-B_Concat (Multimodal GCN Concat)",
        "P2-B_Attn (Multimodal GCN Attention)",
        "P2-C (Topology Isolation MLP)",
        "P2-D (Metadata Isolation MLP)",
    ]

    # Store all raw run metrics: {arm_name: [metric_dict_for_each_of_25_folds]}
    fold_results = {arm: [] for arm in ARM_NAMES}
    fold_predictions = {arm: [] for arm in ARM_NAMES}
    fold_ground_truths = []

    total_runs = len(SEEDS) * N_SPLITS
    run_idx = 0
    t_start = time.time()

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n_nodes), y)):
            run_idx += 1
            print(f"\n[Run {run_idx:02d}/{total_runs}] Seed {seed} | Fold {fold + 1}/{N_SPLITS} (Train={len(train_idx)}, Test={len(test_idx)})...")

            # Ground truth for test fold
            y_test = y[test_idx]
            consensus_test = is_consensus[test_idx]
            fold_ground_truths.append(y_test.tolist())

            # Inner validation split (80% inner-train, 20% inner-val) for neural early stopping
            skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + fold)
            inner_train_idx, inner_val_idx = next(skf_inner.split(train_idx, y[train_idx]))
            sub_train_idx = train_idx[inner_train_idx]
            sub_val_idx = train_idx[inner_val_idx]

            # Compute inverse-frequency class weights from outer training fold
            train_labels = y[train_idx]
            cnts = Counter(train_labels)
            weights = np.array([len(train_labels) / (NUM_CLASSES * max(cnts.get(c, 1), 1)) for c in range(NUM_CLASSES)], dtype=np.float32)
            # Clip weights to prevent instability
            weights = np.clip(weights, 0.2, 10.0)
            class_weights_tensor = torch.from_numpy(weights).float()

            # Preprocessing: Fit Scalers strictly on outer train_idx
            scaler_topo = StandardScaler()
            x_topo_scaled = x_topo_raw.copy()
            x_topo_scaled[train_idx] = scaler_topo.fit_transform(x_topo_raw[train_idx])
            x_topo_scaled[test_idx] = scaler_topo.transform(x_topo_raw[test_idx])

            scaler_meta = StandardScaler()
            x_meta_scaled = x_meta_raw.copy()
            x_meta_scaled[train_idx] = scaler_meta.fit_transform(x_meta_raw[train_idx])
            x_meta_scaled[test_idx] = scaler_meta.transform(x_meta_raw[test_idx])

            x_all_scaled = np.concatenate([x_topo_scaled, x_meta_scaled], axis=1)

            x_topo_tensor = torch.from_numpy(x_topo_scaled).float().to(DEVICE)
            x_meta_tensor = torch.from_numpy(x_meta_scaled).float().to(DEVICE)
            x_all_tensor = torch.from_numpy(x_all_scaled).float().to(DEVICE)

            # Combined tabular + text features for MLP arms
            x_fusion_all = torch.cat([x_arabert_tensor, x_all_tensor], dim=-1)   # (335, 776)
            x_fusion_topo = torch.cat([x_arabert_tensor, x_topo_tensor], dim=-1) # (335, 773)
            x_fusion_meta = torch.cat([x_arabert_tensor, x_meta_tensor], dim=-1) # (335, 771)

            # ── Arm 1: B0 Majority Baseline ──
            majority_class = Counter(train_labels).most_common(1)[0][0]
            pred_b0 = np.full(len(test_idx), majority_class)
            fold_results["B0 (Majority Baseline)"].append(compute_metrics(y_test, pred_b0, consensus_test))
            fold_predictions["B0 (Majority Baseline)"].append(pred_b0.tolist())

            # ── Arm 2: B1 Stratified Random Baseline ──
            np.random.seed(seed + fold)
            p_dist = [cnts.get(c, 0) / len(train_labels) for c in range(NUM_CLASSES)]
            pred_b1 = np.random.choice(NUM_CLASSES, size=len(test_idx), p=p_dist)
            fold_results["B1 (Stratified Random)"].append(compute_metrics(y_test, pred_b1, consensus_test))
            fold_predictions["B1 (Stratified Random)"].append(pred_b1.tolist())

            # ── Arm 3: B2 Name + Meta Probe (LR) ──
            tfidf_name = TfidfVectorizer(ngram_range=(1, 2), max_features=100)
            X_name_train = tfidf_name.fit_transform([canonical_names[i] for i in train_idx]).toarray()
            X_name_test = tfidf_name.transform([canonical_names[i] for i in test_idx]).toarray()
            X_b2_train = np.hstack([X_name_train, x_meta_scaled[train_idx]])
            X_b2_test = np.hstack([X_name_test, x_meta_scaled[test_idx]])
            clf_b2 = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
            clf_b2.fit(X_b2_train, train_labels)
            pred_b2 = clf_b2.predict(X_b2_test)
            fold_results["B2 (Name + Meta Probe)"].append(compute_metrics(y_test, pred_b2, consensus_test))
            fold_predictions["B2 (Name + Meta Probe)"].append(pred_b2.tolist())

            # ── Arm 4: P1-A Lexical TF-IDF LR ──
            tfidf_critic = TfidfVectorizer(ngram_range=(1, 2), max_features=1000, sublinear_tf=True)
            X_text_train = tfidf_critic.fit_transform([raw_texts[i] for i in train_idx])
            X_text_test = tfidf_critic.transform([raw_texts[i] for i in test_idx])
            clf_p1a = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
            clf_p1a.fit(X_text_train, train_labels)
            pred_p1a = clf_p1a.predict(X_text_test)
            fold_results["P1-A (Lexical TF-IDF LR)"].append(compute_metrics(y_test, pred_p1a, consensus_test))
            fold_predictions["P1-A (Lexical TF-IDF LR)"].append(pred_p1a.tolist())

            # ── Arm 5: P1-B AraBERT Semantic LR ──
            clf_p1b = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=seed)
            clf_p1b.fit(x_arabert_raw[train_idx], train_labels)
            pred_p1b = clf_p1b.predict(x_arabert_raw[test_idx])
            fold_results["P1-B (AraBERT Semantic LR)"].append(compute_metrics(y_test, pred_p1b, consensus_test))
            fold_predictions["P1-B (AraBERT Semantic LR)"].append(pred_p1b.tolist())

            # ── Arm 6: P1-B_MLP AraBERT Semantic MLP ──
            torch.manual_seed(seed + fold)
            model_p1b_mlp = TabularMLP(in_dim=768, hidden_dim=64, num_classes=NUM_CLASSES, dropout=0.3)
            pred_p1b_mlp = train_neural_model(
                model_p1b_mlp,
                lambda m: m(x_arabert_tensor),
                sub_train_idx, sub_val_idx, test_idx,
                y_tensor, class_weights_tensor
            )
            fold_results["P1-B_MLP (AraBERT Semantic MLP)"].append(compute_metrics(y_test, pred_p1b_mlp, consensus_test))
            fold_predictions["P1-B_MLP (AraBERT Semantic MLP)"].append(pred_p1b_mlp.tolist())

            # ── Arm 7: G1 Non-GNN Graph Features MLP ──
            torch.manual_seed(seed + fold)
            model_g1 = TabularMLP(in_dim=8, hidden_dim=64, num_classes=NUM_CLASSES, dropout=0.3)
            pred_g1 = train_neural_model(
                model_g1,
                lambda m: m(x_all_tensor),
                sub_train_idx, sub_val_idx, test_idx,
                y_tensor, class_weights_tensor
            )
            fold_results["G1 (Graph Features MLP)"].append(compute_metrics(y_test, pred_g1, consensus_test))
            fold_predictions["G1 (Graph Features MLP)"].append(pred_g1.tolist())

            # ── Arm 8: G2 Graph GCN ──
            torch.manual_seed(seed + fold)
            model_g2 = PureGCN(in_dim=8, hidden_dim=64, num_classes=NUM_CLASSES, dropout=0.3)
            pred_g2 = train_neural_model(
                model_g2,
                lambda m: m(x_all_tensor, edge_index_tensor),
                sub_train_idx, sub_val_idx, test_idx,
                y_tensor, class_weights_tensor
            )
            fold_results["G2 (Graph GCN)"].append(compute_metrics(y_test, pred_g2, consensus_test))
            fold_predictions["G2 (Graph GCN)"].append(pred_g2.tolist())

            # ── Arm 9: P2-A Early Fusion MLP ──
            torch.manual_seed(seed + fold)
            model_p2a = TabularMLP(in_dim=776, hidden_dim=64, num_classes=NUM_CLASSES, dropout=0.3)
            pred_p2a = train_neural_model(
                model_p2a,
                lambda m: m(x_fusion_all),
                sub_train_idx, sub_val_idx, test_idx,
                y_tensor, class_weights_tensor
            )
            fold_results["P2-A (Early Fusion MLP)"].append(compute_metrics(y_test, pred_p2a, consensus_test))
            fold_predictions["P2-A (Early Fusion MLP)"].append(pred_p2a.tolist())

            # ── Arm 10: P2-B_Concat Multimodal GCN Concat ──
            torch.manual_seed(seed + fold)
            model_p2b_c = MultimodalGNNConcat(graph_dim=8, text_dim=768, hidden_dim=64, num_classes=NUM_CLASSES, dropout=0.3)
            pred_p2b_c = train_neural_model(
                model_p2b_c,
                lambda m: m(x_all_tensor, edge_index_tensor, x_arabert_tensor),
                sub_train_idx, sub_val_idx, test_idx,
                y_tensor, class_weights_tensor
            )
            fold_results["P2-B_Concat (Multimodal GCN Concat)"].append(compute_metrics(y_test, pred_p2b_c, consensus_test))
            fold_predictions["P2-B_Concat (Multimodal GCN Concat)"].append(pred_p2b_c.tolist())

            # ── Arm 11: P2-B_Attn Multimodal GCN Attention ──
            torch.manual_seed(seed + fold)
            model_p2b_a = MultimodalGNNAttention(graph_dim=8, text_dim=768, hidden_dim=64, num_classes=NUM_CLASSES, dropout=0.3)
            pred_p2b_a = train_neural_model(
                model_p2b_a,
                lambda m: m(x_all_tensor, edge_index_tensor, x_arabert_tensor),
                sub_train_idx, sub_val_idx, test_idx,
                y_tensor, class_weights_tensor
            )
            fold_results["P2-B_Attn (Multimodal GCN Attention)"].append(compute_metrics(y_test, pred_p2b_a, consensus_test))
            fold_predictions["P2-B_Attn (Multimodal GCN Attention)"].append(pred_p2b_a.tolist())

            # ── Arm 12: P2-C Topology Isolation MLP ──
            torch.manual_seed(seed + fold)
            model_p2c = TabularMLP(in_dim=773, hidden_dim=64, num_classes=NUM_CLASSES, dropout=0.3)
            pred_p2c = train_neural_model(
                model_p2c,
                lambda m: m(x_fusion_topo),
                sub_train_idx, sub_val_idx, test_idx,
                y_tensor, class_weights_tensor
            )
            fold_results["P2-C (Topology Isolation MLP)"].append(compute_metrics(y_test, pred_p2c, consensus_test))
            fold_predictions["P2-C (Topology Isolation MLP)"].append(pred_p2c.tolist())

            # ── Arm 13: P2-D Metadata Isolation MLP ──
            torch.manual_seed(seed + fold)
            model_p2d = TabularMLP(in_dim=771, hidden_dim=64, num_classes=NUM_CLASSES, dropout=0.3)
            pred_p2d = train_neural_model(
                model_p2d,
                lambda m: m(x_fusion_meta),
                sub_train_idx, sub_val_idx, test_idx,
                y_tensor, class_weights_tensor
            )
            fold_results["P2-D (Metadata Isolation MLP)"].append(compute_metrics(y_test, pred_p2d, consensus_test))
            fold_predictions["P2-D (Metadata Isolation MLP)"].append(pred_p2d.tolist())

    elapsed_total = time.time() - t_start
    print(f"\nAll 25 evaluations complete in {elapsed_total:.2f} seconds ({elapsed_total/total_runs:.2f}s per fold).")

    # ── 3. Summary Statistics and Hypothesis Testing ──
    print("\n" + "=" * 80)
    print("AGGREGATED BENCHMARK RESULTS (25 EVALUATIONS)")
    print("=" * 80)

    summary = {}
    metrics_keys = ["macro_f1", "accuracy", "quadratic_kappa", "f1_below_saduq", "f1_saduq", "f1_thiqah", "consensus_macro_f1"]

    for arm in ARM_NAMES:
        arm_runs = fold_results[arm]
        summary[arm] = {}
        for k in metrics_keys:
            vals = [r[k] for r in arm_runs]
            cv_stats = corrected_repeated_cv_summary(vals, n_splits=N_SPLITS)
            m = cv_stats["mean"]
            s = cv_stats["std"]
            summary[arm][k] = {
                "mean": round(m, 4),
                "std": round(s, 4),
                "ci95_low": round(cv_stats["ci_low"], 4),
                "ci95_high": round(cv_stats["ci_high"], 4),
            }

    # Print summary table
    print(f"{'Arm':<38} | {'Macro-F1 (Mean±SD)':<20} | {'95% CI':<16} | {'Acc (%)':<9} | {'Kappa':<8} | {'Consensus F1':<12}")
    print("-" * 115)
    for arm in ARM_NAMES:
        f1_info = summary[arm]["macro_f1"]
        acc_info = summary[arm]["accuracy"]
        kap_info = summary[arm]["quadratic_kappa"]
        con_info = summary[arm]["consensus_macro_f1"]
        f1_str = f"{f1_info['mean']:.4f} ± {f1_info['std']:.4f}"
        ci_str = f"[{f1_info['ci95_low']:.4f}, {f1_info['ci95_high']:.4f}]"
        print(f"{arm:<38} | {f1_str:<20} | {ci_str:<16} | {acc_info['mean']*100:6.2f}%   | {kap_info['mean']:6.4f} | {con_info['mean']:.4f}")

    # ── 4. Formal Paired Hypothesis Tests ──
    print("\n" + "=" * 80)
    print("DEPENDENCE-CORRECTED REPEATED-CV HYPOTHESIS TESTS (df = 24)")
    print("=" * 80)

    comparisons = [
        ("P1-B (AraBERT Semantic LR)", "P1-A (Lexical TF-IDF LR)", "PO1: AraBERT vs TF-IDF Baseline"),
        ("G2 (Graph GCN)", "G1 (Graph Features MLP)", "PO2: Message Passing vs Tabular Features Alone"),
        ("P2-A (Early Fusion MLP)", "P1-B_MLP (AraBERT Semantic MLP)", "H1a: Does Tabular/Network Features Improve Text?"),
        ("P2-B_Concat (Multimodal GCN Concat)", "P2-A (Early Fusion MLP)", "H1b: Does GCN Message Passing Improve Beyond Non-GNN Fusion?"),
        ("P2-B_Attn (Multimodal GCN Attention)", "P2-A (Early Fusion MLP)", "H1b: Does Attention GCN Improve Beyond Non-GNN Fusion?"),
        ("P2-A (Early Fusion MLP)", "P2-C (Topology Isolation MLP)", "Ablation: Contribution of Metadata over Pure Topology"),
    ]

    paired_tests = {}
    for arm_a, arm_b, desc in comparisons:
        f1_a = np.array([r["macro_f1"] for r in fold_results[arm_a]])
        f1_b = np.array([r["macro_f1"] for r in fold_results[arm_b]])
        diff = f1_a - f1_b
        mean_diff = float(np.mean(diff))
        cv_test = corrected_repeated_cv_test(diff, n_splits=N_SPLITS)
        std_diff = cv_test["std"]
        ci_low = cv_test["ci_low"]
        ci_high = cv_test["ci_high"]
        t_stat = cv_test["t_statistic"]
        p_val = cv_test["p_value"]

        paired_tests[desc] = {
            "arm_a": arm_a,
            "arm_b": arm_b,
            "mean_difference": round(mean_diff, 4),
            "std_difference": round(std_diff, 4),
            "ci95_low": round(ci_low, 4),
            "ci95_high": round(ci_high, 4),
            "t_statistic": round(float(t_stat), 4),
            "p_value_parametric": round(float(p_val), 5),
            "test_method": "Nadeau-Bengio corrected repeated 5-fold CV t-test",
            "correction_factor": round(cv_test["correction_factor"], 5),
            "decision": "Significant Difference (p < 0.05)" if p_val < 0.05 else "Fail to Reject Null (p >= 0.05)"
        }

        print(f"\nTest: {desc}")
        print(f"  Comparison: {arm_a} vs. {arm_b}")
        print(f"  Mean Delta: {mean_diff:+.4f} (95% CI: [{ci_low:+.4f}, {ci_high:+.4f}])")
        print(f"  Corrected repeated-CV t-test: t = {t_stat:+.4f}, p = {p_val:.5f} ({paired_tests[desc]['decision']})")

    # ── 5. Save Artifacts ──
    out_dir = ROOT / "output/cleaned_cohort"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "clean_experiments_results.json", "w", encoding="utf-8") as f:
        json.dump(fold_results, f, indent=2)

    summary_payload = {
        "metadata": {
            "n_identities": n_nodes,
            "n_seeds": len(SEEDS),
            "n_splits": N_SPLITS,
            "total_paired_evaluations": total_runs,
            "elapsed_seconds": round(elapsed_total, 2),
            "device": str(DEVICE),
            "inference_method": "Nadeau-Bengio corrected repeated 5-fold CV t-test and intervals",
        },
        "arms_summary": summary,
        "paired_hypothesis_tests": paired_tests,
    }

    with open(out_dir / "clean_experiments_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    print(f"\nSaved granular results to {out_dir / 'clean_experiments_results.json'}")
    print(f"Saved summary and hypothesis tests to {out_dir / 'clean_experiments_summary.json'}")

    # ── 6. Generate Formal Academic Report in FYP2/ ──
    generate_report(summary, paired_tests, total_runs, n_nodes, y, is_consensus, out_dir / "clean_experiments_summary.json")

def generate_report(summary: dict, paired_tests: dict, total_runs: int, n_nodes: int, y: np.ndarray, is_consensus: np.ndarray, summary_path: Path):
    report_path = ROOT / "FYP2/clean_experiments_report.md"

    n_consensus = int(np.sum(is_consensus))
    n_below = int(np.sum(y == 0))
    n_saduq = int(np.sum(y == 1))
    n_thiqah = int(np.sum(y == 2))

    lines = [
        "# Empirical Benchmark Report: 13-Arm Controlled Cross-Validation Study",
        "",
        "## 1. Overview of Experimental Protocol",
        "",
        f"- **Cohort Size:** {n_nodes} unique historical narrator identities (1 identity = 1 node).",
        f"- **Evaluation Protocol:** 5-fold Stratified Cross-Validation repeated across 5 distinct random seeds ({total_runs} total paired evaluations).",
        "- **Target Ground Truth:** Ibn Ḥajar's canonical *Taqrīb al-Tahdhīb* 3-tier synthesis grade (*Thiqah*, *Ṣadūq*, *Below Ṣadūq*).",
        "- **Feature Separation:** Target labels are excluded from text and tabular inputs. Text screening retains the configured date-verified core critics and removes configured snippet-level compiler references.",
        "- **Graph Evaluation:** Transductive node classification. The full cohort graph, including train-test edges, is available to graph models in each fold; loss and scoring remain restricted to the appropriate train and test rows.",
        "- **Row Partitioning:** Stratified folds have disjoint canonical identity rows. This does not make the graph evaluation inductive.",
        "- **Metric Reporting:** Mean and standard deviation across folds; corrected 95% confidence intervals and paired tests use the Nadeau-Bengio repeated-CV variance correction.",
        "",
        "---",
        "",
        "## 2. Benchmark Results Across All Experimental Arms",
        "",
        f"| Arm ID | Experimental Arm | Model Architecture | Macro-F1 (Mean ± SD) | 95% Confidence Interval | Accuracy (%) | Quadratic $\\kappa$ | Consensus F1 ($N={n_consensus}$) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for arm, d in summary.items():
        f1 = d["macro_f1"]
        acc = d["accuracy"]
        kap = d["quadratic_kappa"]
        con = d["consensus_macro_f1"]
        lines.append(
            f"| **{arm.split(' ')[0]}** | {arm} | {get_arch_desc(arm)} | **{f1['mean']:.4f} ± {f1['std']:.4f}** | [{f1['ci95_low']:.4f}, {f1['ci95_high']:.4f}] | {acc['mean']*100:.2f}% | {kap['mean']:.4f} | {con['mean']:.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Per-Class F1 Breakdown Across Tiers",
        "",
        f"| Arm ID | Experimental Arm | Below Ṣadūq ($N={n_below}$) | Ṣadūq ($N={n_saduq}$) | Thiqah ($N={n_thiqah}$) |",
        "|---|---|---|---|---|",
    ])

    for arm, d in summary.items():
        f0 = d["f1_below_saduq"]["mean"]
        f1 = d["f1_saduq"]["mean"]
        f2 = d["f1_thiqah"]["mean"]
        lines.append(f"| **{arm.split(' ')[0]}** | {arm} | {f0:.4f} | {f1:.4f} | {f2:.4f} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Dependence-Corrected Repeated-CV Hypothesis Testing ($df = 24$)",
        "",
        "| Research Hypothesis / Comparison | Arm A vs. Arm B | Mean Difference $\\bar{\\Delta}$ | Corrected 95% CI of Difference | Corrected $t$-statistic | $p$-value | Decision |",
        "|---|---|---|---|---|---|---|",
    ])

    for desc, t in paired_tests.items():
        lines.append(
            f"| **{desc}** | {t['arm_a'].split(' ')[0]} vs. {t['arm_b'].split(' ')[0]} | **{t['mean_difference']:+.4f}** | [{t['ci95_low']:+.4f}, {t['ci95_high']:+.4f}] | $t = {t['t_statistic']:+.4f}$ | $p = {t['p_value_parametric']:.5f}$ | {t['decision']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Scientific Findings and Examiner Synthesis",
        "",
        "The empirical findings from this 25-fold controlled study provide clear answers to the project objectives:",
        "",
        "1. **Lexical vs. Contextual Semantic NLP (PO1)**:",
        "   - Compares earlier critic text representations without target contamination.",
        "   - Eliminates artificial keyword masking artifacts.",
        "",
        "2. **Contribution of Network Information Beyond Text ($H_{1a}$)**:",
        "   - Directly tests whether augmenting biographical texts with network topology and biographical metadata yields statistically significant gains.",
        "",
        "3. **Graph Neural Network Message Passing vs. Non-GNN Tabular Fusion ($H_{1b}$)**:",
        "   - Isolates whether message passing across isnād edges adds measurable predictive value over combining node features directly.",
        "",
        "4. **Dual-Synthesizer Consensus Alignment**:",
        f"   - Validates model predictions on the subset where Ibn Ḥajar and al-Dhahabī reached identical appraisal verdicts ($N={n_consensus}$).",
        "",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Generated formal academic report at {report_path}")

def get_arch_desc(arm: str) -> str:
    if "Majority" in arm: return "Constant Majority"
    if "Stratified Random" in arm: return "Empirical Sampling"
    if "Name" in arm or "LR" in arm: return "Logistic Regression"
    if "GCN" in arm: return "Graph ConvNet (2-layer)"
    if "MLP" in arm: return "MLP Classifier (2-layer)"
    return "Linear / Statistical"

if __name__ == "__main__":
    main()
