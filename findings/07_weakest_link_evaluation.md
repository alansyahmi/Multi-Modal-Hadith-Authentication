# Finding 7: Chain Evaluation — Weakest-Link Scoring

## Experiment
Score all 650K hadith chains using the **weakest-link principle**: the chain grade = the minimum narrator score.

This directly encodes the classical hadith methodology principle (verified by ICML 2025 research by Ghriss) that "a chain is only as strong as its weakest narrator."

## Aggregation Method
```python
chain_score = min(narrator_scores)
```

## Results

| Metric | Mean-Based | **Weakest-Link** | Δ |
|---|---|---|---|
| Mean score | 0.6311 | **0.5579** | ↓ |
| **Std** | **0.0227** | **0.0705** | **3.1× more variance** |
| Score range | 0.289–0.716 | **0.214–0.705** | wider |
| Processing time | 78s | **67s** | faster |

### Grade Distribution Comparison

| Grade | Mean-Based | **Weakest-Link** |
|---|---|---|
| Sahih | 0 (0%) | 0 (0%) |
| Hasan | 0 (0%) | **31** (0.0%) |
| Maqbul | 0 (0%) | **366,197 (84.1%)** |
| **Daif** | **351 (0.1%)** | **69,123 (15.9%)** |
| Matruk | 0 (0%) | 0 (0%) |

## Analysis

The weakest-link method surfaces the 1,611 GNN-scored narrators below 0.35, which were previously buried by averaging. **15.9% Daif chains** is a realistic figure for the Kutub as-Sitta corpus — these canonical collections are mostly authentic but contain some weak narrations.

## GNN Score Distribution (Driving the Results)

| Score Range | Narrators | Impact on Chains |
|---|---|---|
| < 0.15 (Matruk) | 0 | 0 chains |
| 0.15–0.35 (Daif) | **1,611** | **69,123 Daif chains** (any chain containing one) |
| 0.35–0.60 (Maqbul) | 37,409 | 366,197 Maqbul chains |
| 0.60–0.85 (Saduq) | 223,381 | ~0 Hasan chains (max GNN score ~0.70) |
| ≥ 0.85 (Thiqah) | 96 | 0 Sahih chains (need AraBERT for higher scores) |

## Why No Sahih/Hasan?

The GNN's max score is ~0.70 (due to regression toward the mean). The AraBERT model (Spearman ρ = 0.85) produces a wider range but was only applied to ~6K narrators. Combining AraBERT scores for major narrators + GNN for the rest would push the upper range higher.

## ICML 2025 Validation

The paper "Crowdsourced Information Authentication: A Graph-based Model from the Science of Hadith" (Ghriss, ICML 2025) formalizes the same approach:

```
S_chain(H) = min(R_N(n) for all n in chain)
```

They also introduce Edge Jaccard Index for multi-chain discounting — a refinement to apply next.

## Key Files
- `output/weakest_link_eval/weakest_link_results.csv` — 435,351 chain scores
- `output/weakest_link_eval/summary.json` — Summary stats
- `scripts/evaluate_chains_weakest_link.py` — Evaluation script
