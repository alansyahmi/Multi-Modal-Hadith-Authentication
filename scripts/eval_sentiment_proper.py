"""
Proper evaluation of graph-only vs multi-modal GNN,
with meaningful metrics for the bimodal grade distribution.
"""
import sys, json, re, ast
from pathlib import Path
from collections import defaultdict, Counter
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    classification_report, confusion_matrix,
)
from scipy.stats import spearmanr, pearsonr

from src.stage3.trainer import set_seed, create_masks
from src.stage3.gnn_model import build_model
from src.stage3.sentiment import lexicon_sentiment_score

# ── Load data ──
print("Loading graph...")
graph = np.load(r"output/stage2/pyg_graph_large_graded.npz")
x = graph["x"]
edge_index = graph["edge_index"]
ei = torch.from_numpy(edge_index).long() - 1

with open(r"output/stage2/node_names.json", encoding="utf-8") as f:
    id_to_name = json.load(f)

# ── Build multi-modal features ──
print("Building sentiment features...")
sanadset = pd.read_csv(r"output/stage1/sanadset_cleaned.csv", nrows=50000)
matn_by_narrator = defaultdict(list)
for _, row in sanadset.iterrows():
    matn = str(row.get("Matn", ""))
    if not matn or len(matn) < 20: continue
    try:
        chain = ast.literal_eval(row["Sanad"]) if isinstance(row["Sanad"], str) else []
    except: continue
    for name in chain:
        name = re.sub(r'[ً-ٌٍ-َّْٰ]', '', name.strip())
        if len(name) > 3: matn_by_narrator[name].append(matn)

narrator_sent = {}
for name, matns in matn_by_narrator.items():
    feats = [lexicon_sentiment_score(m) for m in matns[:10]]
    narrator_sent[name] = [np.mean([f[k] for f in feats]) for k in feats[0]]

sent_dim = len(feats[0])
expanded_x = np.hstack([x, np.zeros((x.shape[0], sent_dim), dtype=np.float32)])
for node_id_str, name in id_to_name.items():
    idx = int(node_id_str) - 1
    name_clean = re.sub(r'[ً-ٌٍ-َّْٰ]', '', name)
    if name_clean in narrator_sent:
        expanded_x[idx, x.shape[1]:] = narrator_sent[name_clean]

# ── Train and evaluate both models ──
def train_and_eval(x_data, name, num_runs=3):
    """Train model num_runs times, return avg metrics."""
    set_seed(42)

    all_preds = []
    all_true = []

    for run in range(num_runs):
        set_seed(42 + run * 100)
        data = Data(x=torch.from_numpy(x_data).float(), edge_index=ei)
        data.y = data.x[:, 0].clone()

        train_mask, val_mask, test_mask = create_masks(data.num_nodes)
        # Mask grade for val+test
        mask = ~train_mask
        data.x[mask, 0] = 0.30

        model = build_model(in_channels=data.x.size(1), model_type="gcn")
        opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        crit = torch.nn.MSELoss()

        best_val = float("inf")
        best_state = None
        patience = 0

        for epoch in range(200):
            model.train()
            opt.zero_grad()
            out = model(data.x, data.edge_index)
            loss = crit(out[train_mask], data.y[train_mask])
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_mae = torch.abs(val_out[val_mask] - data.y[val_mask]).mean().item()

            if val_mae < best_val:
                best_val = val_mae
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 30: break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index)[test_mask].cpu().numpy()
            true = data.y[test_mask].cpu().numpy()

        all_preds.append(pred)
        all_true.append(true)

    # Average metrics
    preds = np.mean(all_preds, axis=0)
    trues = all_true[0]

    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    spear = spearmanr(trues, preds).statistic
    pear = pearsonr(trues, preds).statistic

    # Binary: "accepted" (>= 0.65) vs "rejected" (< 0.65)
    true_bin = (trues >= 0.65).astype(int)
    pred_bin = (preds >= 0.65).astype(int)
    acc = (true_bin == pred_bin).mean()

    # Per-category MAE
    cat_mae = {}
    for label, thresh in [("thiqah", 0.80), ("saduq", 0.60), ("maqbul", 0.35), ("daif", 0.15), ("matruk", 0.0)]:
        mask_cat = (trues >= thresh - 0.01)
        # Refine: exact ranges
    cat_map = {0.85: "thiqah", 0.70: "saduq", 0.40: "maqbul", 0.15: "daif", 0.05: "matruk"}
    for val, label in cat_map.items():
        close = np.abs(trues - val) < 0.02
        if close.sum() > 5:
            cat_mae[label] = float(np.abs(preds[close] - trues[close]).mean())

    return {"mae": mae, "r2": r2, "spearman": spear, "pearson": pear,
            "accuracy_binary": acc, "cat_mae": cat_mae, "pred_mean": float(preds.mean()),
            "pred_std": float(preds.std()), "true_mean": float(trues.mean()), "true_std": float(trues.std())}

print("\nTraining graph-only GCN...")
graph_results = train_and_eval(x, "graph-only")

print("Training multi-modal GCN...")
multi_results = train_and_eval(expanded_x, "multi-modal")

# ── Print comparison ──
print(f"\n{'='*60}")
print(f"{'Metric':25s} {'Graph-only':>15s} {'Multi-modal':>15s}  Delta")
print(f"{'='*60}")
for metric in ["mae", "r2", "spearman", "pearson", "accuracy_binary"]:
    g = graph_results[metric]
    m = multi_results[metric]
    delta = m - g
    if metric in ("mae",):
        direction = "↑" if delta < 0 else "↓"
    else:
        direction = "↑" if delta > 0 else "↓"
    print(f"{metric:25s} {g:15.4f} {m:15.4f} {direction} {delta:+.4f}")

print(f"\nPer-category MAE:")
for cat in ["thiqah", "saduq", "maqbul", "daif", "matruk"]:
    g = graph_results["cat_mae"].get(cat, 0)
    m = multi_results["cat_mae"].get(cat, 0)
    print(f"  {cat:10s}: graph={g:.4f}  multi={m:.4f}")

print(f"\nPrediction distribution:")
print(f"  Graph-only: mean={graph_results['pred_mean']:.3f}, std={graph_results['pred_std']:.3f}")
print(f"  Multi-mod:  mean={multi_results['pred_mean']:.3f}, std={multi_results['pred_std']:.3f}")
print(f"  True:       mean={graph_results['true_mean']:.3f}, std={graph_results['true_std']:.3f}")

# Save
comparison = {
    "graph_only": {k: float(v) if not isinstance(v, dict) else {kk: float(vv) for kk, vv in v.items()}
                   for k, v in graph_results.items()},
    "multimodal": {k: float(v) if not isinstance(v, dict) else {kk: float(vv) for kk, vv in v.items()}
                   for k, v in multi_results.items()},
}
with open(r"output/stage3/comparison_v2.json", "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=2)
print(f"\nSaved: output/stage3/comparison_v2.json")
