# Empirical Benchmark Report: 10-Arm Controlled Cross-Validation Study

## 1. Overview of Experimental Protocol

- **Cohort Size:** 335 unique historical narrator identities (1 identity = 1 node).
- **Evaluation Protocol:** 5-fold Stratified Cross-Validation repeated across 5 distinct random seeds (25 total paired evaluations).
- **Target Ground Truth:** Ibn Ḥajar's canonical *Taqrīb al-Tahdhīb* 3-tier synthesis grade (*Thiqah*, *Ṣadūq*, *Below Ṣadūq*).
- **Zero Target Leakage:** Biographical text inputs are strictly restricted to earlier classical critics (2nd to 4th century AH); all 9th-century synthesis judgments and summary grades were excluded.
- **Identity Isolation:** Zero narrator overlap between train and test partitions.
- **Metric Reporting:** Mean ± Standard Deviation and 95% Confidence Intervals calculated across all 25 paired runs.

---

## 2. Benchmark Results Across All Experimental Arms

| Arm ID | Experimental Arm | Model Architecture | Macro-F1 (Mean ± SD) | 95% Confidence Interval | Accuracy (%) | Quadratic $\kappa$ | Consensus F1 ($N=181$) |
|---|---|---|---|---|---|---|---|
| **B0** | B0 (Majority Baseline) | Constant Majority | **0.2900 ± 0.0016** | [0.2894, 0.2907] | 77.01% | 0.0000 | 0.3717 |
| **B1** | B1 (Stratified Random) | Empirical Sampling | **0.3383 ± 0.0568** | [0.3148, 0.3617] | 62.57% | 0.0285 | 0.3484 |
| **B2** | B2 (Name + Meta Probe) | Logistic Regression | **0.3806 ± 0.0618** | [0.3551, 0.4061] | 52.18% | 0.0248 | 0.3216 |
| **P1-A** | P1-A (Lexical TF-IDF LR) | Logistic Regression | **0.5246 ± 0.0278** | [0.5131, 0.5360] | 83.82% | 0.5396 | 0.5113 |
| **P1-B** | P1-B (AraBERT Semantic LR) | Logistic Regression | **0.4958 ± 0.0800** | [0.4628, 0.5288] | 74.33% | 0.4614 | 0.4580 |
| **P1-B_MLP** | P1-B_MLP (AraBERT Semantic MLP) | MLP Classifier (2-layer) | **0.4511 ± 0.0782** | [0.4188, 0.4833] | 70.21% | 0.3992 | 0.4335 |
| **G1** | G1 (Graph Features MLP) | MLP Classifier (2-layer) | **0.3628 ± 0.0740** | [0.3323, 0.3934] | 49.67% | 0.0698 | 0.3337 |
| **G2** | G2 (Graph GCN) | Graph ConvNet (2-layer) | **0.3536 ± 0.0476** | [0.3339, 0.3732] | 45.01% | 0.0583 | 0.2877 |
| **P2-A** | P2-A (Early Fusion MLP) | MLP Classifier (2-layer) | **0.4763 ± 0.0898** | [0.4392, 0.5133] | 68.12% | 0.3305 | 0.4571 |
| **P2-B_Concat** | P2-B_Concat (Multimodal GCN Concat) | Graph ConvNet (2-layer) | **0.4439 ± 0.0964** | [0.4041, 0.4837] | 63.22% | 0.1999 | 0.4180 |
| **P2-B_Attn** | P2-B_Attn (Multimodal GCN Attention) | Graph ConvNet (2-layer) | **0.4723 ± 0.0838** | [0.4377, 0.5069] | 69.13% | 0.2760 | 0.4464 |
| **P2-C** | P2-C (Topology Isolation MLP) | MLP Classifier (2-layer) | **0.4414 ± 0.0951** | [0.4021, 0.4806] | 69.73% | 0.3248 | 0.4559 |
| **P2-D** | P2-D (Metadata Isolation MLP) | MLP Classifier (2-layer) | **0.4966 ± 0.0640** | [0.4702, 0.5230] | 70.45% | 0.3449 | 0.4906 |

---

## 3. Per-Class F1 Breakdown Across Tiers

| Arm ID | Experimental Arm | Below Ṣadūq ($N=13$) | Ṣadūq ($N=64$) | Thiqah ($N=258$) |
|---|---|---|---|---|
| **B0** | B0 (Majority Baseline) | 0.0000 | 0.0000 | 0.8701 |
| **B1** | B1 (Stratified Random) | 0.0633 | 0.1860 | 0.7655 |
| **B2** | B2 (Name + Meta Probe) | 0.1901 | 0.3112 | 0.6406 |
| **P1-A** | P1-A (Lexical TF-IDF LR) | 0.0000 | 0.6633 | 0.9104 |
| **P1-B** | P1-B (AraBERT Semantic LR) | 0.1660 | 0.4648 | 0.8566 |
| **P1-B_MLP** | P1-B_MLP (AraBERT Semantic MLP) | 0.1658 | 0.3564 | 0.8310 |
| **G1** | G1 (Graph Features MLP) | 0.1475 | 0.3170 | 0.6240 |
| **G2** | G2 (Graph GCN) | 0.1508 | 0.3644 | 0.5456 |
| **P2-A** | P2-A (Early Fusion MLP) | 0.2216 | 0.4058 | 0.8014 |
| **P2-B_Concat** | P2-B_Concat (Multimodal GCN Concat) | 0.2047 | 0.3853 | 0.7417 |
| **P2-B_Attn** | P2-B_Attn (Multimodal GCN Attention) | 0.2077 | 0.4098 | 0.7994 |
| **P2-C** | P2-C (Topology Isolation MLP) | 0.1271 | 0.3724 | 0.8247 |
| **P2-D** | P2-D (Metadata Isolation MLP) | 0.2736 | 0.3982 | 0.8180 |

---

## 4. Formal Paired Hypothesis Testing ($df = 24$)

| Research Hypothesis / Comparison | Arm A vs. Arm B | Mean Difference $\bar{\Delta}$ | 95% CI of Difference | Paired $t$-statistic | $p$-value | Decision |
|---|---|---|---|---|---|---|
| **PO1: AraBERT vs TF-IDF Baseline** | P1-B vs. P1-A | **-0.0288** | [-0.0634, +0.0058] | $t = -1.7166$ | $p = 0.09893$ | Fail to Reject Null (p >= 0.05) |
| **PO2: Message Passing vs Tabular Features Alone** | G2 vs. G1 | **-0.0093** | [-0.0310, +0.0125] | $t = -0.8793$ | $p = 0.38796$ | Fail to Reject Null (p >= 0.05) |
| **H1a: Does Tabular/Network Features Improve Text?** | P2-A vs. P1-B_MLP | **+0.0252** | [-0.0170, +0.0674] | $t = +1.2328$ | $p = 0.22960$ | Fail to Reject Null (p >= 0.05) |
| **H1b: Does GCN Message Passing Improve Beyond Non-GNN Fusion?** | P2-B_Concat vs. P2-A | **-0.0324** | [-0.0735, +0.0088] | $t = -1.6235$ | $p = 0.11754$ | Fail to Reject Null (p >= 0.05) |
| **H1b: Does Attention GCN Improve Beyond Non-GNN Fusion?** | P2-B_Attn vs. P2-A | **-0.0040** | [-0.0424, +0.0345] | $t = -0.2134$ | $p = 0.83284$ | Fail to Reject Null (p >= 0.05) |
| **Ablation: Contribution of Metadata over Pure Topology** | P2-A vs. P2-C | **+0.0349** | [-0.0092, +0.0790] | $t = +1.6319$ | $p = 0.11575$ | Fail to Reject Null (p >= 0.05) |

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
   - Validates model predictions on the subset where Ibn Ḥajar and al-Dhahabī reached identical appraisal verdicts ($N=181$).
