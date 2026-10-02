# Finding 13: Classifier vs Canonical Grades (Kutub as-Sitta)

## Experiment
Compare GNN-only vs 5-class classifier chain scores against ground-truth Kutub as-Sitta canonical grades.

## Comparison Table

| Metric | GNN-Only | 5-Class Classifier | Winner |
|---|---|---|---|
| **Exact Accuracy** | **0.1047** | 0.0822 | GNN |
| **±1 Grade Acc** | **0.5626** | 0.3796 | GNN |
| F1 Macro | 0.1013 | **0.1038** | Tie |
| Cohen's κ (quad) | 0.0218 | **0.0431** | CLS (+98%) |

## Per-Class Recall

| Grade | n | GNN Recall | Classifier Recall |
|---|---|---|---|
| Sahih | 786 | **0.000%** | 0.004% |
| Hasan | 105 | 0.543% | **0.324%** |
| Daif | 130 | 0.385% | **0.362%** |
| Matruk | 1 | 0.000% | 0.000% |

## Accuracy by Book

| Book | n | GNN | Classifier |
|---|---|---|---|
| Jāmiʿ at-Tirmiḏī | 164 | **0.2073** | 0.1463 |
| Sunan Abī Dāwūd | 172 | **0.1337** | 0.1163 |
| Sunan Ibn Mājah | 156 | **0.1731** | 0.1667 |
| Sunan al-Nasāʾī | 193 | **0.1192** | 0.0674 |
| Ṣaḥīḥ Muslim | 154 | 0.0000 | **0.0065** |
| Ṣaḥīḥ al-Bukhārī | 183 | 0.0000 | 0.0000 |

## Sample Predictions (Ṣaḥīḥ al-Bukhārī)

| Actual | GNN Grade | CLS Grade | CLS Score |
|---|---|---|---|
| Sahih | Matruk | Matruk | 0.315 |
| Sahih | Hasan | Matruk | 0.050 |
| Sahih | Daif | Matruk | 0.050 |
| Sahih | Daif | Daif | 0.400 |
| Sahih | Hasan | Hasan | 0.599 |

## The Gap: Narrator-Level vs Chain-Level

| Method | Narrator-Level | Chain-Level | Gap |
|---|---|---|---|
| GNN regression | κ ≈ 0.02 | Acc = 10.5% | None |
| AraBERT regression | Spearman = 0.85 | No improvement | — |
| **5-class classifier** | **κ = 0.87** | **Acc = 8.2%** | **Massive** |

The classifier is excellent at individual narrator grading (κ=0.87) but too strict for chains because weakest-link surfaces every weak narrator. Classical scholars accept hadith with some weak narrators IF corroborated by other chains (ṭuruq).

## Implication
The finding validates the need for multi-chain corroboration. A hadith appearing with multiple independent chains should NOT be graded by its weakest chain alone — if ANY chain has all-strong narrators, the hadith can be elevated.

## Key Files
- `output/final_evaluation/classifier_vs_canonical.csv` — 1,022 scored hadiths
- `output/final_evaluation/classifier_vs_canonical.json` — Summary stats
