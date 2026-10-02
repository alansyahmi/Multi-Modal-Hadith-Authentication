"""
Build sentiment-augmented graph features and train GNN for comparison.
"""
import sys, json, re, ast
from pathlib import Path
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.stage3.sentiment import lexicon_sentiment_score
from src.stage3.trainer import load_graph, train, set_seed
from src.stage3.gnn_model import build_model

# ── Build sentiment features ──
print("=" * 50)
print("BUILDING SENTIMENT FEATURES")
print("=" * 50)

# Load graph
graph_path = Path(r"output/stage2/pyg_graph_large_graded.npz")
graph = np.load(graph_path)
x = graph["x"]
edge_index = graph["edge_index"]
edge_attr = graph["edge_attr"]

# Load node names
with open(r"output/stage2/node_names.json", encoding="utf-8") as f:
    id_to_name = json.load(f)

# Load Sanadset matns
print(f"\nLoading Sanadset matns...")
sanadset = pd.read_csv(r"output/stage1/sanadset_cleaned.csv", nrows=50000)

# Build narrator→matns mapping
matn_by_narrator: dict[str, list[str]] = defaultdict(list)
for _, row in sanadset.iterrows():
    matn = str(row.get("Matn", ""))
    if not matn or len(matn) < 20:
        continue
    try:
        chain = ast.literal_eval(row["Sanad"]) if isinstance(row["Sanad"], str) else []
    except:
        continue
    for name in chain:
        name = re.sub(r'[ً-ٌٍ-َّْٰ]', '', name.strip())
        if len(name) > 3:
            matn_by_narrator[name].append(matn)

print(f"  Narrators with matn data: {len(matn_by_narrator):,}")

# Build sentiment features per narrator
print(f"\nComputing lexicon-based sentiment features...")
narrator_sent = {}
for name, matns in matn_by_narrator.items():
    feats = [lexicon_sentiment_score(m) for m in matns[:10]]  # Cap at 10 matns
    avg = {k: np.mean([f[k] for f in feats]) for k in feats[0]}
    narrator_sent[name] = list(avg.values())

sent_keys = list(feats[0].keys())
sent_dim = len(sent_keys)
print(f"  Sentiment dimension: {sent_dim}")
print(f"  Keys: {sent_keys}")

# Map to graph nodes
n_nodes = x.shape[0]
expanded_x = np.hstack([x, np.zeros((n_nodes, sent_dim), dtype=np.float32)])
matched = 0

for node_id_str, name in id_to_name.items():
    node_id = int(node_id_str)
    idx = node_id - 1
    name_clean = re.sub(r'[ً-ٌٍ-َّْٰ]', '', name)
    if name_clean in narrator_sent:
        expanded_x[idx, x.shape[1]:] = narrator_sent[name_clean]
        matched += 1

print(f"  Sentiment attached: {matched:,}/{n_nodes:,} ({matched/n_nodes*100:.1f}%)")

# ── Train GNN (graph-only baseline) ──
print(f"\n{'='*50}")
print("BASELINE: Graph-only GCN")
print(f"{'='*50}")

set_seed(42)
import torch
from torch_geometric.data import Data

ei = torch.from_numpy(edge_index).long() - 1
ea = torch.from_numpy(edge_attr).float() if edge_attr is not None else None
data = Data(x=torch.from_numpy(x).float(), edge_index=ei, edge_attr=ea)
data.y = data.x[:, 0].clone()

# Create masks
from src.stage3.trainer import create_masks
train_mask, val_mask, test_mask = create_masks(data.num_nodes)
mask = ~train_mask
data.x[mask, 0] = 0.30

model_base, results_base = train(data, model_type="gcn", num_epochs=150, patience=30, verbose=False)

# ── Train GNN (multi-modal: graph + sentiment) ──
print(f"\n{'='*50}")
print("MULTI-MODAL: Graph + Sentiment GCN")
print(f"{'='*50}")

data_sent = Data(
    x=torch.from_numpy(expanded_x).float(),
    edge_index=ei,
    edge_attr=ea,
)
data_sent.y = data_sent.x[:, 0].clone()
# Mask grade for test nodes
data_sent.x[mask, 0] = 0.30

model_sent, results_sent = train(data_sent, model_type="gcn", num_epochs=150, patience=30, verbose=False)

# ── Comparison ──
print(f"\n{'='*50}")
print("COMPARISON: Graph-only vs Multi-modal")
print(f"{'='*50}")

for metric in ["mae", "rmse", "r2", "accuracy", "f1_macro"]:
    base_val = results_base["test"][metric]
    sent_val = results_sent["test"][metric]
    delta = sent_val - base_val
    direction = "↑" if (metric in ("r2", "accuracy", "f1_macro") and delta > 0) or (metric in ("mae", "rmse") and delta < 0) else ""
    if (metric in ("r2", "accuracy", "f1_macro") and delta > 0) or (metric in ("mae", "rmse") and delta < 0):
        direction = "↑ better"
    elif abs(delta) < 0.001:
        direction = "= same"
    else:
        direction = "↓ worse"
    print(f"  {metric:12s}: base={base_val:.4f}  multi={sent_val:.4f}  {direction}")

# Save multi-modal graph
out_path = Path(r"output/stage2/pyg_graph_multimodal.npz")
np.savez(out_path, x=expanded_x, edge_index=edge_index, edge_attr=edge_attr)
print(f"\nMulti-modal graph saved: {out_path}")
print(f"  Features: {expanded_x.shape[1]} dims (8 graph + {sent_dim} sentiment)")

# Save comparison
comp = {
    "graph_only": results_base["test"],
    "multimodal": results_sent["test"],
    "sentiment_dim": sent_dim,
    "sentiment_feature_keys": sent_keys,
}
with open(r"output/stage3/comparison.json", "w", encoding="utf-8") as f:
    json.dump(comp, f, indent=2, ensure_ascii=False)
print(f"  Comparison: output/stage3/comparison.json")
