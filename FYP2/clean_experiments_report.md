# Empirical Benchmark Report: 13-Arm Controlled Cross-Validation Study

## 1. Overview of Experimental Protocol

- **Cohort Size:** 335 unique historical narrator identities (1 identity = 1 node).
- **Evaluation Protocol:** 5-fold Stratified Cross-Validation repeated across 5 distinct random seeds (25 total paired evaluations).
- **Target Ground Truth:** Ibn Ḥajar's canonical *Taqrīb al-Tahdhīb* 3-tier synthesis grade (*Thiqah*, *Ṣadūq*, *Below Ṣadūq*).
- **Zero Target Leakage:** Biographical text inputs are strictly restricted to earlier classical critics (2nd to 4th century AH); all 9th-century synthesis judgments and summary grades were excluded.
- **Identity Isolation:** Zero narrator overlap between train and test partitions.
- **Metric Reporting:** Mean ± Standard Deviation and 95% Confidence Intervals calculated across all 25 paired runs.

---

## 2. Benchmark Results Across All Experimental Arms

| Arm ID | Experimental Arm | Model Architecture | Macro-F1 (Mean ± SD) | 95% Confidence Interval | Accuracy (%) | Quadratic $\kappa$ | Consensus F1 ($N=180$) |
|---|---|---|---|---|---|---|---|
| **B0** | B0 (Majority Baseline) | Constant Majority | **0.2881 ± 0.0000** | [0.2881, 0.2881] | 76.12% | 0.0000 | 0.3593 |
| **B1** | B1 (Stratified Random) | Empirical Sampling | **0.3272 ± 0.0601** | [0.3024, 0.3520] | 61.25% | -0.0133 | 0.3050 |
| **B2** | B2 (Name + Meta Probe) | Logistic Regression | **0.3822 ± 0.0439** | [0.3641, 0.4003] | 50.69% | 0.0318 | 0.3436 |
| **P1-A** | P1-A (Lexical TF-IDF LR) | Logistic Regression | **0.5608 ± 0.0703** | [0.5318, 0.5898] | 80.96% | 0.4932 | 0.5284 |
| **P1-B** | P1-B (AraBERT Semantic LR) | Logistic Regression | **0.4933 ± 0.0628** | [0.4674, 0.5192] | 72.84% | 0.4347 | 0.4478 |
| **P1-B_MLP** | P1-B_MLP (AraBERT Semantic MLP) | MLP Classifier (2-layer) | **0.4454 ± 0.0657** | [0.4183, 0.4725] | 68.60% | 0.3295 | 0.4037 |
| **G1** | G1 (Graph Features MLP) | MLP Classifier (2-layer) | **0.3556 ± 0.0564** | [0.3323, 0.3788] | 47.64% | 0.0899 | 0.3030 |
| **G2** | G2 (Graph GCN) | Graph ConvNet (2-layer) | **0.3438 ± 0.0716** | [0.3142, 0.3733] | 44.30% | 0.0343 | 0.2755 |
| **P2-A** | P2-A (Early Fusion MLP) | MLP Classifier (2-layer) | **0.4810 ± 0.0684** | [0.4528, 0.5093] | 68.36% | 0.3602 | 0.4708 |
| **P2-B_Concat** | P2-B_Concat (Multimodal GCN Concat) | Graph ConvNet (2-layer) | **0.4311 ± 0.0750** | [0.4002, 0.4621] | 64.96% | 0.2157 | 0.3979 |
| **P2-B_Attn** | P2-B_Attn (Multimodal GCN Attention) | Graph ConvNet (2-layer) | **0.4478 ± 0.0670** | [0.4202, 0.4755] | 66.27% | 0.2549 | 0.4271 |
| **P2-C** | P2-C (Topology Isolation MLP) | MLP Classifier (2-layer) | **0.4531 ± 0.0727** | [0.4231, 0.4831] | 67.22% | 0.3318 | 0.4270 |
| **P2-D** | P2-D (Metadata Isolation MLP) | MLP Classifier (2-layer) | **0.4315 ± 0.0796** | [0.3987, 0.4644] | 65.97% | 0.2919 | 0.3980 |

---

## 3. Per-Class F1 Breakdown Across Tiers

| Arm ID | Experimental Arm | Below Ṣadūq ($N=20$) | Ṣadūq ($N=60$) | Thiqah ($N=255$) |
|---|---|---|---|---|
| **B0** | B0 (Majority Baseline) | 0.0000 | 0.0000 | 0.8644 |
| **B1** | B1 (Stratified Random) | 0.0294 | 0.1952 | 0.7570 |
| **B2** | B2 (Name + Meta Probe) | 0.1785 | 0.3415 | 0.6267 |
| **P1-A** | P1-A (Lexical TF-IDF LR) | 0.1630 | 0.6170 | 0.9024 |
| **P1-B** | P1-B (AraBERT Semantic LR) | 0.1885 | 0.4352 | 0.8562 |
| **P1-B_MLP** | P1-B_MLP (AraBERT Semantic MLP) | 0.1548 | 0.3545 | 0.8269 |
| **G1** | G1 (Graph Features MLP) | 0.1630 | 0.2865 | 0.6173 |
| **G2** | G2 (Graph GCN) | 0.1559 | 0.3248 | 0.5506 |
| **P2-A** | P2-A (Early Fusion MLP) | 0.2682 | 0.3623 | 0.8126 |
| **P2-B_Concat** | P2-B_Concat (Multimodal GCN Concat) | 0.1609 | 0.3585 | 0.7739 |
| **P2-B_Attn** | P2-B_Attn (Multimodal GCN Attention) | 0.1975 | 0.3554 | 0.7906 |
| **P2-C** | P2-C (Topology Isolation MLP) | 0.2178 | 0.3321 | 0.8093 |
| **P2-D** | P2-D (Metadata Isolation MLP) | 0.1779 | 0.3170 | 0.7998 |

---

## 4. Formal Paired Hypothesis Testing ($df = 24$)

| Research Hypothesis / Comparison | Arm A vs. Arm B | Mean Difference $\bar{\Delta}$ | 95% CI of Difference | Paired $t$-statistic | $p$-value | Decision |
|---|---|---|---|---|---|---|
| **PO1: AraBERT vs TF-IDF Baseline** | P1-B vs. P1-A | **-0.0675** | [-0.1051, -0.0299] | $t = -3.7035$ | $p = 0.00111$ | Significant Difference (p < 0.05) |
| **PO2: Message Passing vs Tabular Features Alone** | G2 vs. G1 | **-0.0118** | [-0.0436, +0.0200] | $t = -0.7667$ | $p = 0.45076$ | Fail to Reject Null (p >= 0.05) |
| **H1a: Does Tabular/Network Features Improve Text?** | P2-A vs. P1-B_MLP | **+0.0357** | [-0.0011, +0.0724] | $t = +2.0044$ | $p = 0.05643$ | Fail to Reject Null (p >= 0.05) |
| **H1b: Does GCN Message Passing Improve Beyond Non-GNN Fusion?** | P2-B_Concat vs. P2-A | **-0.0499** | [-0.0918, -0.0080] | $t = -2.4595$ | $p = 0.02150$ | Significant Difference (p < 0.05) |
| **H1b: Does Attention GCN Improve Beyond Non-GNN Fusion?** | P2-B_Attn vs. P2-A | **-0.0332** | [-0.0656, -0.0008] | $t = -2.1130$ | $p = 0.04519$ | Significant Difference (p < 0.05) |
| **Ablation: Contribution of Metadata over Pure Topology** | P2-A vs. P2-C | **+0.0279** | [-0.0081, +0.0639] | $t = +1.6014$ | $p = 0.12238$ | Fail to Reject Null (p >= 0.05) |

---

## 5. Scientific Findings and Examiner Synthesis

The empirical findings from this 25-fold controlled study provide clear answers to the project objectives:

1. **Lexical vs. Contextual Semantic NLP (PO1)**:
   - Compares earlier critic text representations without target contamination.
   - Eliminates artificial keyword masking artifacts.

2. **Contribution of Network Information Beyond Text ($H_{1a}$)**:
   - Directly tests whether augmenting biographical texts with network topology and biographical metadata yields statistically significant gains.

3. **Graph Neural Network Message Passing vs. Non-GNN Tabular Fusion ($H_{1b}$)**:
   - Isolates whether message passing across isnād edges adds measurable predictive value over combining node features directly.

4. **Dual-Synthesizer Consensus Alignment**:
   - Validates model predictions on the subset where Ibn Ḥajar and al-Dhahabī reached identical appraisal verdicts ($N=180$).
