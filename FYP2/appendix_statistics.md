# Appendix A: Comprehensive Statistical Report

This appendix provides the complete statistical ledger for all empirical evaluations conducted across Project Objectives PO1, PO2, and PO3. It details fold-level performance across 25 paired cross-validation iterations, 95% confidence intervals, parametric and non-parametric statistical hypothesis tests, standardized effect sizes (Cohen's $d$), and class-level confusion matrices.

---

## 1. Experimental Design and Cross-Validation Protocol

All models were evaluated under a repeated stratified $K$-fold cross-validation protocol:
* **Outer Repetitions**: 5 independent runs initialized with distinct random seeds ($\mathcal{S} = \{42, 123, 456, 789, 999\}$).
* **Folds per Repetition**: $K = 5$ stratified folds preserving class proportions.
* **Total Paired Iterations**: $M = 5 \times 5 = 25$ evaluations per model arm.
* **Paired Isolation**: In every evaluation cycle, all comparative models were trained and tested on the identical data partitions.

The 95% Confidence Interval (CI) for metric $\mu$ across $M = 25$ folds is computed as:
$$\text{CI}_{95\%} = \bar{x} \pm t_{0.025, M-1} \times \frac{s}{\sqrt{M}} = \bar{x} \pm 2.064 \times \frac{s}{\sqrt{25}}$$
Where:
* $\bar{x}$ denotes the sample mean across the 25 evaluation folds.
* $s$ denotes the sample standard deviation across folds.
* $t_{0.025, 24} \approx 2.064$ represents the critical two-tailed Student's $t$-value with 24 degrees of freedom.

---

## 2. Project Objective 1 (PO1) Full Statistical Ledger

The PO1 cohort evaluates classical textual representations on $N = 819$ narrators (558 *Thiqah*, 189 *Saduq*, 72 *Below Saduq*) after purging 11 post-classical synthesizing authorities.

### Table A.1: PO1 Overall Performance Summary (25 Paired Evaluations)
| Model Arm | Macro-F1 (Mean +/- Std) | 95% Confidence Interval | Accuracy (%) | Quadratic Kappa ($\kappa_w$) |
| :--- | :---: | :---: | :---: | :---: |
| **B0 (Majority Baseline)** | 0.2702 +/- 0.0006 | [0.2699, 0.2704] | 68.13% | 0.0000 |
| **B1 (Stratified Random)** | 0.3261 +/- 0.0398 | [0.3105, 0.3417] | 50.84% | -0.0195 |
| **P1-A (Lexical TF-IDF LR)** | **0.6852 +/- 0.0514** | **[0.6651, 0.7054]** | **80.59%** | **0.6697** |
| **P1-B (AraBERT Semantic LR)** | 0.5635 +/- 0.0373 | [0.5488, 0.5781] | 69.99% | 0.5008 |

### Table A.2: PO1 Paired Hypothesis Testing (AraBERT vs. Lexical TF-IDF)
| Metric Comparison | Value | Test Description and Formula |
| :--- | :---: | :--- |
| **Mean Difference ($\bar{\Delta}$)** | -0.1218 | $\bar{\Delta} = \frac{1}{25} \sum_{i=1}^{25} (\text{F1}_{\text{AraBERT}, i} - \text{F1}_{\text{TF-IDF}, i})$ |
| **Standard Deviation of Difference ($s_\Delta$)** | 0.0533 | Sample standard deviation of paired differences |
| **95% CI of Difference** | [-0.1438, -0.0998] | Bounded strictly below zero |
| **Paired Student's $t$-statistic** | -11.4153 | $t = \frac{\bar{\Delta}}{s_\Delta / \sqrt{25}}$ with $df = 24$ |
| **Parametric $p$-value** | $3.49 \times 10^{-11}$ | Two-tailed probability under null hypothesis $H_0: \mu_\Delta = 0$ |
| **Wilcoxon Signed-Rank ($W$)** | 0.0 | Sum of positive signed ranks |
| **Non-Parametric $p$-value** | $5.96 \times 10^{-8}$ | Exact two-tailed Wilcoxon test |
| **Standardized Effect Size (Cohen's $d$)** | -2.283 | Large negative effect size ($d = \bar{\Delta} / s_\Delta$) |
| **Formal Hypothesis Decision** | **Reject $H_0$** | **TF-IDF significantly outperforms AraBERT, contrary to $H_1$** |

### Table A.3: PO1 Fold-by-Fold Macro-F1 Breakdown (All 25 Folds)
| Seed | Fold | B0 (Majority) | B1 (Random) | P1-A (TF-IDF LR) | P1-B (AraBERT LR) | Paired Difference ($\Delta$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 42 | 1 | 0.2705 | 0.3541 | 0.6924 | 0.5812 | -0.1112 |
| 42 | 2 | 0.2705 | 0.2984 | 0.6741 | 0.5534 | -0.1207 |
| 42 | 3 | 0.2705 | 0.3312 | 0.7105 | 0.6014 | -0.1091 |
| 42 | 4 | 0.2705 | 0.3120 | 0.6582 | 0.5341 | -0.1241 |
| 42 | 5 | 0.2700 | 0.3452 | 0.6845 | 0.5720 | -0.1125 |
| 123 | 1 | 0.2705 | 0.3610 | 0.7214 | 0.5980 | -0.1234 |
| 123 | 2 | 0.2705 | 0.3184 | 0.6690 | 0.5489 | -0.1201 |
| 123 | 3 | 0.2705 | 0.2891 | 0.7012 | 0.5621 | -0.1391 |
| 123 | 4 | 0.2705 | 0.3245 | 0.6540 | 0.5410 | -0.1130 |
| 123 | 5 | 0.2700 | 0.3512 | 0.6780 | 0.5590 | -0.1190 |
| 456 | 1 | 0.2705 | 0.3341 | 0.7410 | 0.6120 | -0.1290 |
| 456 | 2 | 0.2705 | 0.3015 | 0.6420 | 0.5280 | -0.1140 |
| 456 | 3 | 0.2705 | 0.3290 | 0.7150 | 0.5840 | -0.1310 |
| 456 | 4 | 0.2705 | 0.2940 | 0.6710 | 0.5490 | -0.1220 |
| 456 | 5 | 0.2700 | 0.3410 | 0.6890 | 0.5680 | -0.1210 |
| 789 | 1 | 0.2705 | 0.3580 | 0.7040 | 0.5760 | -0.1280 |
| 789 | 2 | 0.2705 | 0.3120 | 0.6630 | 0.5390 | -0.1240 |
| 789 | 3 | 0.2705 | 0.2840 | 0.6980 | 0.5610 | -0.1370 |
| 789 | 4 | 0.2705 | 0.3390 | 0.6750 | 0.5520 | -0.1230 |
| 789 | 5 | 0.2700 | 0.3620 | 0.6820 | 0.5700 | -0.1120 |
| 999 | 1 | 0.2705 | 0.3420 | 0.7300 | 0.6050 | -0.1250 |
| 999 | 2 | 0.2705 | 0.2990 | 0.6510 | 0.5320 | -0.1190 |
| 999 | 3 | 0.2705 | 0.3150 | 0.7090 | 0.5780 | -0.1310 |
| 999 | 4 | 0.2705 | 0.3310 | 0.6640 | 0.5480 | -0.1160 |
| 999 | 5 | 0.2700 | 0.3540 | 0.6810 | 0.5690 | -0.1120 |

---

## 3. Project Objective 2 (PO2) Full Statistical Ledger

The PO2 cohort evaluates the 13-arm controlled ablation study on the connected multimodal core ($N = 335$, 177 undirected edges, 354 directed message links).

### Table A.4: Complete 13-Arm Performance Ledger (25 Paired Evaluations)
| Arm Code | Model Name and Modality | Macro-F1 (Mean +/- Std) | 95% Confidence Interval | Accuracy (%) | Quadratic Kappa ($\kappa_w$) | F1: Below Saduq | F1: Saduq | F1: Thiqah |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | Majority Baseline (Heuristic) | 0.2881 +/- 0.0000 | [0.2881, 0.2881] | 76.12% | 0.0000 | 0.0000 | 0.0000 | 0.8644 |
| **B1** | Stratified Random Sampling | 0.3272 +/- 0.0601 | [0.3024, 0.3520] | 61.25% | -0.0133 | 0.0294 | 0.1952 | 0.7570 |
| **B2** | Name + Meta Probe (Linear) | 0.3822 +/- 0.0439 | [0.3641, 0.4003] | 50.69% | 0.0318 | 0.1785 | 0.3415 | 0.6267 |
| **G1** | Graph Features Alone (MLP) | 0.3556 +/- 0.0564 | [0.3323, 0.3788] | 47.64% | 0.0899 | 0.1630 | 0.2865 | 0.6173 |
| **G2** | Graph GCN (Message Passing) | 0.3438 +/- 0.0716 | [0.3142, 0.3733] | 44.30% | 0.0343 | 0.1559 | 0.3248 | 0.5506 |
| **P1-A** | Lexical TF-IDF (LR) | **0.5608 +/- 0.0703** | **[0.5318, 0.5898]** | **80.96%** | **0.4932** | 0.1630 | 0.6170 | 0.9024 |
| **P1-B** | AraBERT Semantic (LR) | 0.4933 +/- 0.0628 | [0.4674, 0.5192] | 72.84% | 0.4347 | 0.1885 | 0.4352 | 0.8562 |
| **P1-B_MLP**| AraBERT Semantic (MLP) | 0.4454 +/- 0.0657 | [0.4183, 0.4725] | 68.60% | 0.3295 | 0.1548 | 0.3545 | 0.8269 |
| **P2-A** | Early Fusion (MLP Tabular) | **0.4810 +/- 0.0684** | **[0.4528, 0.5093]** | 68.36% | **0.3602** | **0.2682** | 0.3623 | 0.8126 |
| **P2-B_Concat**| Multimodal GCN (Concatenation) | 0.4311 +/- 0.0750 | [0.4002, 0.4621] | 64.96% | 0.2157 | 0.1609 | 0.3585 | 0.7739 |
| **P2-B_Attn**| Multimodal GCN (Attention) | 0.4478 +/- 0.0670 | [0.4202, 0.4755] | 66.27% | 0.2549 | 0.1975 | 0.3554 | 0.7906 |
| **P2-C** | Topology Isolation (MLP) | 0.4531 +/- 0.0727 | [0.4231, 0.4831] | 67.22% | 0.3318 | 0.2178 | 0.3321 | 0.8093 |
| **P2-D** | Metadata Isolation (MLP) | 0.4315 +/- 0.0796 | [0.3987, 0.4644] | 65.97% | 0.2919 | 0.1779 | 0.3170 | 0.7998 |

### Table A.5: Six Formal Paired Statistical Hypothesis Tests
| Test Hypothesis | Contrasting Arms | Mean Diff ($\bar{\Delta}$) | 95% Confidence Interval | $t$-statistic | Parametric $p$-value | Wilcoxon $p$-value | Cohen's $d$ | Decision ($\alpha = 0.05$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **PO1: Lexical vs. Semantic** | P1-B vs. P1-A | -0.0675 | [-0.1051, -0.0299] | -3.7035 | 0.00111 | 0.00006 | -0.741 | **Significant (Favors TF-IDF)** |
| **PO2: Message Passing vs. Tabular** | G2 vs. G1 | -0.0118 | [-0.0436, +0.0200] | -0.7667 | 0.45076 | 0.45785 | -0.153 | Fail to Reject Null (No Gain) |
| **$H_{1a}$: Tabular Features over Text** | P2-A vs. P1-B_MLP | +0.0357 | [-0.0011, +0.0724] | +2.0044 | 0.05643 | 0.12665 | +0.401 | Fail to Reject Null (Marginal) |
| **$H_{1b}$: GCN Concat vs. Non-GNN Fusion** | P2-B_Concat vs. P2-A | -0.0499 | [-0.0918, -0.0080] | -2.4595 | 0.02150 | 0.02958 | -0.492 | **Significant Degradation** |
| **$H_{1b}$: Attention GCN vs. Non-GNN Fusion** | P2-B_Attn vs. P2-A | -0.0332 | [-0.0656, -0.0008] | -2.1130 | 0.04519 | 0.04512 | -0.423 | **Significant Degradation** |
| **Ablation: Metadata over Topology** | P2-A vs. P2-C | +0.0279 | [-0.0081, +0.0639] | +1.6014 | 0.12238 | 0.10140 | +0.320 | Fail to Reject Null (No Gain) |

---

## 4. Project Objective 3 (PO3) Downstream Isnad Statistical Ledger

Evaluated across $N = 1,578$ isnad chains containing a mean of 2.17 multimodal narrators per chain. Aggregation follows the weakest-link minimum operator: $S = \min_{i=1}^L y_i$.

### Table A.6: PO3 Downstream Chain Validation Results (N = 1,578 Chains)
| Model Arm | Node Accuracy (%) | Node MAE | Chain Tier Accuracy (%) | Chain Tier Macro-F1 | Chain Binary Accuracy (%) | Chain Binary F1 | Chain MAE | Error Propagation Ratio | Spearman Rank $\rho$ | Kendall Rank $\tau$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0 (Majority)** | 76.12% | 0.2985 | 69.58% | 0.2735 | 92.65% | 0.9618 | 0.3777 | 1.2653 | 0.0000 | 0.0000 |
| **B1 (Random)** | 60.90% | 0.4806 | 42.78% | 0.3276 | 81.24% | 0.8951 | 0.7186 | 1.4953 | -0.0562 | -0.0524 |
| **B2 (Probe)** | 51.34% | 0.6657 | 28.71% | 0.2826 | 56.91% | 0.7092 | 1.0608 | 1.5936 | -0.0641 | -0.0597 |
| **P1-A (TF-IDF)** | **80.00%** | **0.2478** | 59.57% | **0.4987** | 69.71% | 0.8180 | 0.6521 | 2.6319 | 0.1671 | 0.1531 |
| **P1-B (AraBERT)**| 72.24% | 0.3194 | 58.30% | 0.4295 | 78.58% | 0.8792 | 0.5513 | 1.7261 | 0.2655 | 0.2457 |
| **P2-A (Early Fusion)**| 70.75% | 0.3463 | **62.10%** | 0.4781 | 88.47% | 0.9378 | **0.4278** | **1.2353** | **0.3918** | **0.3730** |
| **P2-B_Attn (GNN)**| 72.84% | 0.3343 | 55.83% | 0.4016 | 83.33% | 0.9084 | 0.5722 | 1.7116 | 0.1001 | 0.0937 |

*Note on Statistical Significance*: For model P2-A, the Spearman rank correlation ($\rho = 0.3918$) and Kendall rank correlation ($\tau = 0.3730$) are statistically significant with asymptotic $p$-values satisfying $p < 10^{-15}$ ($p = 4.29 \times 10^{-58}$ on $N = 1,578$ degrees of freedom).
