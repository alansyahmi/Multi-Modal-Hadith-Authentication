# Finding 6: Chain Evaluation — Mean-Based Scoring

## Experiment
Score all 650K hadith chains by averaging narrator reliability scores, then map to grade thresholds.

## Aggregation Method
```python
chain_score = mean(narrator_scores)
```

## Grade Thresholds (from Experiment 8)
| Grade | Threshold |
|---|---|
| Sahih/Hasan | ≥ 0.85 |
| Marginally Accepted | ≥ 0.70 |
| Uncertain | ≥ 0.50 |
| Da'if/Rejected | < 0.50 |

## Results

| Metric | Value |
|---|---|
| Chains processed | **435,351** (of 650K, 159K parse errors) |
| Processing time | **78s** (5,536 chains/sec) |
| Mean score | **0.6311** |
| Std | **0.0227** |
| Score range | 0.289 – 0.716 |

### Grade Distribution (Mean-Based)

| Grade | Count | % |
|---|---|---|
| Sahih/Hasan | 0 | 0.0% |
| Marginally Accepted | 67 | 0.0% |
| **Uncertain** | **434,933** | **99.9%** |
| Da'if/Rejected | 351 | 0.1% |

## Root Cause Analysis

The mean collapses to "Uncertain" because:
1. **6.8 narrators per chain** — averaging many scores converges to the mean
2. **GNN scores concentrated in 0.55–0.70** — narrow band
3. **Central limit theorem** — the more narrators, the tighter the average

Example: a chain with 6 thiqat (0.85) and 1 da'if (0.15) → mean = 0.75 (Marginally Accepted). Classical method: da'if (weakest link).

## Comparison: Mean vs Weakest-Link

| Method | Mean | Std | Daif % |
|---|---|---|---|
| **Mean-based** | 0.631 | 0.023 | **0.1%** |
| **Weakest-link** | 0.558 | 0.071 | **15.9%** |

The mean-based approach is unsuitable for hadith authentication. See Finding 7 for the weakest-link solution.

## Key Files
- `output/evaluation_fullscale/chain_results.csv` — 435,351 chain scores
- `output/evaluation_fullscale/chain_summary.json` — Summary stats
- `scripts/evaluate_650k_chains.py` — Evaluation script
