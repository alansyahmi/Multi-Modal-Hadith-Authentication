# Finding 12: 5-Class Classifier (Track A v2)

## Experiment
Replace AraBERT regression head with 5-class classification (thiqah/saduq/maqbul/daif/matruk) using ordinal-weighted cross-entropy loss. Forces the model to place narrators in discrete reliability buckets instead of predicting a near-constant mean.

## Training

| Parameter | Value |
|---|---|
| Model | aubmindlab/bert-base-arabertv2 |
| Classes | 5 (softmax) |
| Loss | CrossEntropyLoss with inverse-frequency weights |
| Epochs | 10 (best at epoch 7) |
| Training time | 1,309s (22 min on RTX 2070) |
| Data | 10,913 graded entries (unbalanced) |

### Class Weights
- matruk: 2.22
- daif: 2.78
- maqbul: 1.29
- saduq: 1.00
- thiqah: 1.07

## Test Set Results

| Grade | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| matruk | 0.82 | 0.83 | **0.83** | 218 |
| daif | 0.88 | 0.73 | **0.80** | 187 |
| maqbul | 0.86 | 0.86 | **0.86** | 351 |
| saduq | 0.83 | 0.87 | **0.85** | 462 |
| **thiqah** | **0.90** | **0.90** | **0.90** | 418 |
| **Overall** | **0.86** | **0.84** | **0.85** | 1,636 |

## Regression vs Classification

| Metric | Regression | **5-Class Classifier** | Δ |
|---|---|---|---|
| Cohen's κ | 0.4061 | **0.8692** | **+114%** |
| Spearman ρ | 0.8534 | **0.8778** | +3% |
| Accuracy (binary) | 0.8793 | 0.8557 (5-class) | — |
| F1 (macro) | 0.8268 (binary) | **0.8468** | — |

## Score Distribution on 262K KG Narrators

| Score | Class | Narrators | % |
|---|---|---|---|
| 0.85 | thiqah | 3,916 | 1.5% |
| 0.70 | saduq | 13,237 | 5.0% |
| 0.40 | maqbul | 2,223 | 0.8% |
| 0.15 | daif | 634 | 0.2% |
| 0.05 | matruk | 2,465 | 0.9% |
| (GNN fallback) | — | ~240K | 91.5% |

## Key Finding
The 5-class classifier achieves **κ=0.87 at narrator-level grading** but **only κ=0.04 at chain-level** when combined with weakest-link. The classifier correctly identifies weak narrators, but weakest-link makes the chain grade too strict. This validates the need for **ṭuruq corroboration** (Finding 13).

## Key Files
- `output/track_a_classifier/arabert_classifier/` — Saved model
- `output/track_a_classifier/classifier_metrics.json` — Test metrics
- `output/track_a_classifier/narrator_scores_classifier.json` — 262K scores
