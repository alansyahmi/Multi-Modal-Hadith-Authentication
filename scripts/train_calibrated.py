"""
Train GNN with proper calibration and compare three feature sets:
  1. Graph-only (8 graph features)
  2. Graph + Lexicon (8 + 7 = 15 features)
  3. Graph + BERT-PCA (8 + 128 = 136 features)

Applies isotonic regression calibration post-training to align
predictions with the true grade distribution, then reports
per-category MAE, Spearman ρ, Pearson r, and prediction statistics.
"""
import sys, json, math, os
from pathlib import Path
from itertools import product

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from sklearn.metrics import (
    mean_absolute_error, r2_score, accuracy_score,
)
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr, pearsonr

from src.stage3.gnn_model import build_model
from src.stage3.trainer import set_seed, create_masks
from src.stage3.config import (
    LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE,
    TRAIN_RATIO, VAL_RATIO, RANDOM_SEED,
)

# ── Helper: ordinal category mapping ──
def to_category(score: float) -> str:
    if score >= 0.85: return "thiqah"
    elif score >= 0.60: return "saduq"
    elif score >= 0.35: return "maqbul"
    elif score >= 0.15: return "daif"
    else: return "matruk"

CATEGORY_VALUES = {
    "thiqah": 0.85, "saduq": 0.70, "maqbul": 0.40,
    "daif": 0.15, "matruk": 0.05,
}

# ── Training with early stopping ──
def train_model(data: Data, seed: int = 42, num_epochs: int = 200,
                patience: int = 30, verbose: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Train GCN and return (predictions, truths) on TEST set only."""
    set_seed(seed)

    train_mask, val_mask, test_mask = create_masks(data.num_nodes)
    data.y = data.x[:, 0].clone()
    # Mask grade for val+test
    mask = ~train_mask
    data.x[mask, 0] = 0.30

    model = build_model(in_channels=data.x.size(1), model_type="gcn")
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    crit = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10,
    )

    best_val = float("inf")
    best_state = None
    patience_count = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = crit(out[train_mask], data.y[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index, data.edge_attr)
            val_mae = torch.abs(val_out[val_mask] - data.y[val_mask]).mean().item()

        scheduler.step(val_mae)

        if val_mae < best_val - 1e-6:
            best_val = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index, data.edge_attr)
        return preds[test_mask].cpu(), data.y[test_mask].cpu(), preds.cpu(), data.y.cpu()


# ── Calibration: Isotonic regression ──
def calibrate_isotonic(train_preds: np.ndarray, train_true: np.ndarray,
                       val_preds: np.ndarray, val_true: np.ndarray,
                       test_preds: np.ndarray) -> np.ndarray:
    """
    Fit isotonic regression on train+val, apply to test predictions.
    This preserves ranking (monotonic) while calibrating to the true scale.
    """
    iso = IsotonicRegression(
        y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True,
    )
    combined_preds = np.concatenate([train_preds, val_preds])
    combined_true = np.concatenate([train_true, val_true])
    iso.fit(combined_preds, combined_true)
    return iso.predict(test_preds)


# ── Compute all metrics ──
def compute_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    """Compute comprehensive evaluation metrics."""
    mae = float(mean_absolute_error(true, pred))
    r2_val = float(r2_score(true, pred))
    spear = float(spearmanr(true, pred).statistic)
    pear = float(pearsonr(true, pred).statistic)

    # Binary classification: accepted (>= 0.65) vs rejected
    true_bin = (true >= 0.65).astype(int)
    pred_bin = (pred >= 0.65).astype(int)
    acc = float(accuracy_score(true_bin, pred_bin))

    # Per-category MAE
    cat_mae = {}
    cat_counts = {}
    for label, val in CATEGORY_VALUES.items():
        close = np.abs(true - val) < 0.025
        if close.sum() >= 3:
            cat_mae[label] = float(np.abs(pred[close] - true[close]).mean())
            cat_counts[label] = int(close.sum())
        else:
            cat_mae[label] = np.nan
            cat_counts[label] = 0

    # F1 macro for 5 categories
    true_cat = np.array([list(CATEGORY_VALUES.values()).index(
        min(CATEGORY_VALUES.values(), key=lambda v: abs(v - s))) for s in true])
    pred_cat = np.array([list(CATEGORY_VALUES.values()).index(
        min(CATEGORY_VALUES.values(), key=lambda v: abs(v - s))) for s in pred])
    from sklearn.metrics import f1_score
    f1 = float(f1_score(true_cat, pred_cat, average="macro"))

    return {
        "mae": mae, "r2": r2_val, "spearman": spear, "pearson": pear,
        "accuracy_binary": acc, "f1_macro": f1,
        "cat_mae": cat_mae, "cat_counts": cat_counts,
        "pred_mean": float(pred.mean()), "pred_std": float(pred.std()),
        "true_mean": float(true.mean()), "true_std": float(true.std()),
    }


# ── Main comparison ──
def evaluate_variant(x_data: np.ndarray, variant_name: str,
                     edge_index: torch.Tensor, edge_attr: torch.Tensor | None,
                     num_runs: int = 3) -> dict:
    """Train and evaluate one feature variant with calibration."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {variant_name} ({x_data.shape[1]} features)")
    print(f"{'='*60}")

    ei = edge_index.clone()
    ea = edge_attr.clone() if edge_attr is not None else None

    # ── Save original ground truth BEFORE any masking ──
    # The GNN uses data.x[:, 0] as both input feature AND target label.
    # train_model() masks val+test grade features to 0.30 in data.x.
    # We must save the original grades before that mutation.
    original_y = torch.from_numpy(x_data[:, 0].copy()).float()

    all_metrics = []
    for run in range(num_runs):
        seed = RANDOM_SEED + run * 100
        print(f"  Run {run+1}/{num_runs} (seed={seed})...")

        # Create fresh data object for each run (avoid mutation from previous run)
        data = Data(
            x=torch.from_numpy(x_data.copy()).float(),
            edge_index=ei.clone(),
            edge_attr=ea.clone() if ea is not None else None,
        )
        data.y = original_y.clone()

        set_seed(seed)
        train_mask, val_mask, test_mask = create_masks(data.num_nodes)
        # Mask grade feature for val+test nodes
        mask = ~train_mask
        data.x[mask, 0] = 0.30

        model = build_model(in_channels=data.x.size(1), model_type="gcn")
        opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        crit = nn.MSELoss()
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

        best_val = float("inf")
        best_state = None
        pc = 0
        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            opt.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr)
            loss = crit(out[train_mask], data.y[train_mask])
            loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index, data.edge_attr)
                vm = torch.abs(val_out[val_mask] - data.y[val_mask]).mean().item()
            sched.step(vm)
            if vm < best_val - 1e-6:
                best_val = vm
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                pc = 0
            else:
                pc += 1
                if pc >= PATIENCE: break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            raw_all = model(data.x, data.edge_index, data.edge_attr).cpu().numpy()
            train_preds_raw = raw_all[train_mask.numpy()]
            train_true_raw = data.y[train_mask].cpu().numpy()
            val_preds_raw = raw_all[val_mask.numpy()]
            val_true_raw = data.y[val_mask].cpu().numpy()
            test_preds_raw = raw_all[test_mask.numpy()]
            test_true_raw = data.y[test_mask].cpu().numpy()

        # Calibrate with isotonic regression
        calibrated = calibrate_isotonic(
            train_preds_raw, train_true_raw,
            val_preds_raw, val_true_raw,
            test_preds_raw,
        )

        metrics = compute_metrics(test_true_raw, calibrated)
        metrics["raw_mae"] = float(mean_absolute_error(test_true_raw, test_preds_raw))
        all_metrics.append(metrics)

    # Average across runs
    avg = {}
    for key in all_metrics[0]:
        if key in ("cat_mae", "cat_counts"):
            # Average numeric values, preserving keys
            avg[key] = {}
            for subkey in all_metrics[0][key]:
                vals = [m[key][subkey] for m in all_metrics if not (isinstance(m[key][subkey], float) and np.isnan(m[key][subkey]))]
                avg[key][subkey] = float(np.mean(vals)) if vals else np.nan
        elif isinstance(all_metrics[0][key], (int, float)):
            avg[key] = float(np.mean([m[key] for m in all_metrics]))
        else:
            avg[key] = all_metrics[0][key]

    avg["num_runs"] = num_runs
    return avg


# ── Load graph data ──
print("Loading graph...")
base = np.load(r"output/stage2/pyg_graph_large_graded.npz")
x_base = base["x"]
ei = torch.from_numpy(base["edge_index"]).long() - 1
ea = torch.from_numpy(base["edge_attr"]).float() if base.get("edge_attr") is not None else None

# Check which variants exist
variants = {
    "graph-only": (x_base, "8 graph features"),
}

# Try loading lexicon
lexicon_path = Path(r"output/stage2/pyg_graph_lexicon.npz")
if lexicon_path.exists():
    lex = np.load(lexicon_path)
    variants["lexicon"] = (lex["x"], "8 graph + 7 lexicon = 15 features")
    print(f"  Lexicon graph found: {lex['x'].shape}")
else:
    print(f"  Lexicon graph NOT found (will skip)")

# Try loading BERT
bert_path = Path(r"output/stage2/pyg_graph_bert.npz")
if bert_path.exists():
    bert = np.load(bert_path)
    variants["BERT"] = (bert["x"], "8 graph + 768 BERT = 776 features")
    print(f"  BERT graph found: {bert['x'].shape}")
else:
    print(f"  BERT graph NOT found (will skip — run build_bert_embeddings.py first)")

# ── Evaluate all variants ──
results = {}
for name, (x_data, desc) in variants.items():
    results[name] = evaluate_variant(x_data, f"{name} ({desc})", ei, ea)

# ── Print comparison table ──
print(f"\n\n{'='*70}")
print(f"{'FINAL COMPARISON':^70}")
print(f"{'All variants with isotonic regression calibration':^70}")
print(f"{'='*70}")

metric_order = ["mae", "raw_mae", "spearman", "pearson", "r2", "accuracy_binary", "f1_macro"]
metric_labels = {
    "mae": "MAE (calibrated)",
    "raw_mae": "MAE (raw)",
    "spearman": "Spearman ρ",
    "pearson": "Pearson r",
    "r2": "R²",
    "accuracy_binary": "Binary Acc",
    "f1_macro": "F1 Macro",
}

header = f"{'Metric':<24s}"
for name in variants:
    header += f" {name:>14s}"
print(header)
print("-" * 70)

for metric in metric_order:
    row = f"{metric_labels.get(metric, metric):<24s}"
    best_val = None
    best_name = None
    vals = {}
    for name in variants:
        v = results[name][metric]
        vals[name] = v
        if best_val is None:
            if metric in ("mae", "raw_mae"):
                best_val, best_name = v, name  # lower is better
            else:
                best_val, best_name = v, name  # higher is better
        else:
            if metric in ("mae", "raw_mae"):
                if v < best_val:
                    best_val, best_name = v, name
            else:
                if v > best_val:
                    best_val, best_name = v, name

    for name in variants:
        v = vals[name]
        marker = " *" if name == best_name else "  "
        row += f" {v:>12.4f}{marker}"
    print(row)

# Per-category MAE
print(f"\nPer-Category MAE (calibrated):")
cat_header = f"{'Category':<12s}"
for name in variants:
    cat_header += f" {name:>14s}"
print(cat_header)
print("-" * 50)
for cat in ["thiqah", "saduq", "maqbul", "daif", "matruk"]:
    row = f"{cat:<12s}"
    for name in variants:
        v = results[name]["cat_mae"].get(cat, float("nan"))
        cnt = results[name]["cat_counts"].get(cat, 0)
        if not np.isnan(v):
            row += f" {v:>12.4f}"
        else:
            row += f" {'N/A':>14s}"
    print(row)

# Prediction distribution
print(f"\nPrediction Distribution:")
print(f"{'Variant':<16s} {'Pred Mean':>10s} {'Pred Std':>10s} {'True Mean':>10s} {'True Std':>10s}")
print("-" * 56)
for name in variants:
    r = results[name]
    print(f"{name:<16s} {r['pred_mean']:>10.3f} {r['pred_std']:>10.3f} "
          f"{r['true_mean']:>10.3f} {r['true_std']:>10.3f}")

# ── Save results ──
out_path = Path(r"output/stage3/calibrated_comparison.json")
# Convert to serializable
serializable = {}
for name, r in results.items():
    serializable[name] = {}
    for k, v in r.items():
        if isinstance(v, dict):
            serializable[name][k] = {kk: float(vv) if not (isinstance(vv, float) and np.isnan(vv)) else None
                                     for kk, vv in v.items()}
        else:
            serializable[name][k] = float(v) if isinstance(v, (np.floating, float)) else v

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(serializable, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {out_path}")

print("\nDone.")
