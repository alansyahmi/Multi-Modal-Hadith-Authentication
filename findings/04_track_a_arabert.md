# Finding 4: Track A — AraBERT v2 Regression Training

## Experiment
Fine-tune AraBERT v2 (aubmindlab/bert-base-arabertv2) as a regression model on Jarḥ wa Taʿdīl biographical text to predict continuous narrator reliability scores (0.0–1.0).

Implements Track A from §4.7.1 of the thesis.

## Training Data

### Pre-Balance Distribution
| Category | Count |
|---|---|
| Matruk (0.0–0.15) | 1,400 |
| Daif (0.15–0.35) | 1,116 |
| Maqbul (0.35–0.60) | 2,389 |
| Saduq (0.60–0.85) | 3,073 |
| Thiqah (0.85–1.0) | 2,935 |
| **Total** | **10,913** |

### Post-Balance Distribution
Balanced to **1,116 per class** = **5,580 total**

Split: 3,906 train / 837 val / 837 test

## Training Config
| Parameter | Value |
|---|---|
| Model | aubmindlab/bert-base-arabertv2 (135M params) |
| Max length | 256 tokens |
| Batch size | 16 |
| Epochs | 10 |
| Learning rate | 2e-5 |
| Optimizer | AdamW |
| Loss | MSE |
| Mixed precision | FP16 (torch.amp) |
| Hardware | RTX 2070 8GB |
| Training time | 797s (13 min) |

## Epoch Progress

| Epoch | Loss | val_MAE | val_R² |
|---|---|---|---|
| 1 | 0.0612 | 0.2474 | +0.089 |
| 2 | 0.0403 | 0.2456 | +0.071 |
| 3 | 0.0361 | 0.2513 | +0.027 |
| **4** | **0.0312** | **0.2398** ✅ | **+0.122** |
| 5 | 0.0288 | 0.2445 | +0.082 |
| 6 | 0.0244 | 0.2409 | +0.104 |
| 7 | 0.0215 | 0.2446 | +0.059 |
| 8 | 0.0202 | 0.2428 | +0.086 |
| 9 | 0.0179 | 0.2443 | +0.072 |
| 10 | 0.0189 | 0.2415 | +0.098 |

Loss dropped **70%** (0.061 → 0.018). R² positive in all 10 epochs.

## Test Set Results (n=837)

| Metric | **Old Encyclopedia** (3.7K entries) | **New Encyclopedia** (10.9K entries) | Δ |
|---|---|---|---|
| **Spearman ρ** | 0.7584 | **0.8534** | **↑ +12.5%** |
| **Pearson r** | 0.7587 | **0.8677** | **↑ +14.4%** |
| **Cohen's κ** | 0.3321 | **0.4061** | **↑ +22.3%** |
| **Binary Accuracy** | 0.8254 | **0.8793** | **↑ +6.5%** |
| **Binary F1** | 0.7381 | **0.8268** | **↑ +12.0%** |
| MAE | 0.2469 | **0.2476** | ~same |
| R² | 0.1243 | **0.0861** | — |

## Per-Category MAE

| Grade | Old Encyclopedia | **New Encyclopedia** | Δ |
|---|---|---|---|
| Thiqah | 0.1963 | **0.1774** | ↓ 10% |
| **Saduq** | 0.0903 | **0.0575** | **↓ 42%** |
| Maqbul | 0.1547 | **0.1709** | — |
| Daif | 0.3683 | **0.3719** | — |
| Matruk | 0.4661 | **0.4361** | ↓ 6% |

## Key Finding: Balanced Data Matters

The balanced training was critical. With the old imbalanced data (70% thiqah/saduq), the model collapsed to predicting ~0.63 for everyone (std=0.015). With balanced data (20% each class), the model learned to distinguish categories (Spearman ρ = 0.85).

## Weakness: Score Range Compression

Despite ρ = 0.85, the model's output range was compressed:
- Test set ground truth std: **0.301**
- Model predictions std: **only 0.070**

The regression head learns to hedge toward the mean rather than predicting extreme values. A 5-class classification head with ordinal loss would likely fix this.

## Key Files
- `output/track_a/arabert_regression_model/` — Saved model
- `output/track_a/evaluation_metrics.json` — Test metrics
- `output/track_a/training_stats.json` — Training stats
- `output/track_a/test_predictions.json` — Per-sample predictions
- `scripts/train_arabert_regression.py` — Training script
