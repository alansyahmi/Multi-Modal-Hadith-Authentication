# Finding 14: Ṭuruq Corroboration (Multi-Chain Validation)

## Experiment
Implement the classical *ṭuruq* (multi-chain corroboration) principle: when the same matn appears with multiple independent chains, score each chain independently and take the best corroborated grade. This reflects the traditional *mutāba'āt* and *shawāhid* methodology.

## Method
1. Build a hash index of all unique matn texts in the 650K Sanadset
2. For each evaluation hadith, find all other Sanadset rows with the same matn hash
3. Score each chain independently (weakest-link + risk + classifier scores)
4. Final grade = max(best_chain_grade, single_chain_grade)

## Results

| Metric | Single-Chain | **Ṭuruq** | Δ |
|---|---|---|---|
| Hadiths improved | — | **21 (2.1%)** | +21 |
| Accuracy | 0.0822 | **0.0841** | +0.0019 |
| F1 Macro | 0.1038 | **0.1048** | +0.0010 |
| Cohen's κ | 0.0431 | **0.0434** | +0.0003 |

## Sample Improvements

| Book | Actual | Single → | Ṭuruq | Chains Found |
|---|---|---|---|---|
| Ṣaḥīḥ al-Bukhārī | Sahih | Matruk | **Hasan** | 4 |
| Ṣaḥīḥ Muslim | Sahih | Matruk | **Hasan** | 2 |
| Ṣaḥīḥ Muslim | Sahih | Daif | **Hasan** | 13 |
| Sunan Abī Dāwūd | Sahih | Matruk | **Hasan** | 2 |

## All Methods Comparison (Kutub as-Sitta, n=1,022)

| Method | Accuracy | F1 Macro | κ (quad) |
|---|---|---|---|
| GNN-only (mean-based) | 0.1047 | 0.1013 | 0.0218 |
| AraBERT regression + weakest-link | ~0.10 | — | ~0.02 |
| 5-class classifier + weakest-link | 0.0822 | 0.1038 | 0.0431 |
| **5-class classifier + ṭuruq** | **0.0841** | **0.1048** | **0.0434** |

## Why Accuracy Is Low Across All Methods

1. **Ṣaḥīḥ al-Bukhārī has 0% recall**: The chains in canonical "Ṣaḥīḥ" books contain narrators whose individual biographical profiles (weak memory, some errors) don't match their overall acceptance by the scholarly community. Scholars use **jumhūr (majority) consensus**, not strict weakest-link.

2. **GNN-only has higher accuracy but for the wrong reason**: It predicts "Maqbul" for almost everything (mean ~0.63), which happens to match ~55% of the evaluation set. The classifier at least TRIES to distinguish grades.

3. **The gap between narrator-level (κ=0.87) and chain-level (κ=0.04)** proves that individual narrator scoring is necessary but insufficient for chain authentication.

## Key Files
- `output/final_evaluation/turuq_evaluation.csv` — 1,022 scored hadiths
- `scripts/eval_with_turuq.py` — Evaluation script
