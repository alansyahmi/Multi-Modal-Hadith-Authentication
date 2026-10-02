# Finding 8: Multi-Sanad Convergence Analysis (Experiment 9)

## Experiment
Find matn texts appearing with multiple different narrator chains across all 650K hadiths, and check whether independent chains for the same matn produce consistent scores.

This directly tests the ʿIlm ar-Rijāl principle of ṭuruq (multiple chains): if scores agree across chains → internal consistency. If they diverge → potential grading issues.

## Results

| Metric | Value |
|---|---|
| Hadiths scanned | 433,666 |
| Unique matns | 418,880 |
| **Matns with ≥3 different chains** | **1,958** |
| Total chain-score pairs | 11,736 |

## Convergence Statistics

| Metric | Value |
|---|---|
| Mean score (all groups) | 0.6321 |
| **Mean std within groups** | **0.0135** |
| Mean coefficient of variation | 0.0215 |

### Convergence Distribution

| Std threshold | Groups | % |
|---|---|---|
| ≤ 0.001 | 26 | 1.3% |
| ≤ 0.005 | 292 | 14.9% |
| ≤ 0.010 | 759 | **38.8%** |
| ≤ 0.020 | 1,597 | **81.6%** |
| ≤ 0.050 | 1,949 | **99.5%** |

## Interpretation

**81.6% of multi-chain matns have std ≤ 0.02.** This is strong evidence of internal consistency — the scoring model assigns similar scores to different chains transmitting the same text.

However, this consistency is partly due to the narrow GNN score range (all scores ~0.63). The convergence is real but the variance is artificially suppressed. With weakest-link scoring, more divergence would be expected between strong and weak chains.

## Top Divergent Cases

| Std | n Chains | Mean | Range |
|---|---|---|---|
| 0.0783 | 3 | 0.589 | [0.48, 0.64] |
| 0.0646 | 4 | 0.598 | [0.49, 0.64] |
| 0.0641 | 3 | 0.581 | [0.51, 0.66] |
| 0.0578 | 4 | 0.586 | [0.49, 0.63] |
| 0.0548 | 3 | 0.578 | [0.50, 0.63] |

These represent matns where different narrators in the chain produce meaningfully different scores — worth investigating manually.

## Key Files
- `output/multi_sanad_experiment/convergence.csv` — 1,958 groups
- `output/multi_sanad_experiment/notable_cases.txt` — High/low variance examples
- `output/multi_sanad_experiment/findings.json` — Summary
- `scripts/multi_sanad_experiment.py` — Experiment script
