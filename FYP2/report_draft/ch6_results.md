# Chapter 6: Results

## 6.1 PO1 Large-Scale Biographical Text Benchmark Results (N = 844)

The primary goal of Project Objective 1 (PO1) is to determine how accurately machine learning models can predict canonical narrator reliability tiers from classical earlier-critic biographical statements under strict zero-leakage conditions. Table 6.1 presents the evaluation results across all 844 historical scholars in the Hadith-Viz corpus possessing verified *Taqrīb* grades and earlier-critic commentary. Evaluations were executed across 25 paired runs (5-fold Stratified Cross-Validation repeated across 5 random seeds).

### Table 6.1: PO1 Large-Scale Biographical Text Benchmark Results (N = 844)

| Model ID | Model Architecture | Macro-F1 (Mean ± SD) | 95% Confidence Interval | Accuracy (%) | Quadratic Weighted $\kappa$ |
|---|---|---|---|---|---|
| **B0** | Majority Baseline | **0.2713 ± 0.0002** | [0.2712, 0.2713] | 68.60% | 0.0000 |
| **B1** | Stratified Random | **0.3400 ± 0.0327** | [0.3265, 0.3535] | 53.51% | 0.0158 |
| **P1-A** | Lexical TF-IDF LR | **0.7089 ± 0.0422** | [0.6915, 0.7263] | **81.06%** | **0.6498** |
| **P1-B** | AraBERT Semantic LR | **0.5902 ± 0.0412** | [0.5732, 0.6072] | 70.78% | 0.4412 |

Several key findings emerge from the large-scale biographical benchmark:
* **High Predictive Utility of Classical Evaluative Text:** Both machine learning models substantially outperform the majority baseline (B0: 0.2713 Macro-F1) and stratified chance (B1: 0.3400 Macro-F1). This demonstrates that 2nd to 4th-century classical critic statements provide strong, consistent signals that reliably predict Ibn Ḥajar's 9th-century synthesis verdicts.
* **Superiority of Regularized Lexical Representations:** The lexical TF-IDF model (P1-A) achieved a superior Macro-F1 score of **0.7089 ± 0.0422** and an overall accuracy of **81.06%**, outperforming the pretrained AraBERT semantic classifier (P1-B: **0.5902 ± 0.0412**).
* **Substantial Inter-Rater Agreement:** P1-A achieved a Quadratic Weighted Kappa of **$\kappa = 0.6498$**, reflecting substantial ordinal agreement with Ibn Ḥajar's canonical ratings. By contrast, AraBERT achieved moderate agreement ($\kappa = 0.4412$).
* **Domain Adaptation Friction:** The performance margin between TF-IDF and AraBERT indicates domain friction. AraBERT was pretrained primarily on Modern Standard Arabic news articles and Wikipedia entries. Classical Arabic biographical texts contain concise technical formulaic descriptors (such as *laysa bi-qawī* or *kāna yukhṭiʾ*) that provide direct discriminative signals to regularized lexical n-gram models, whereas deep contextual transformers dilute these precise lexical signals across general sentence embeddings.

---

## 6.2 PO2 Controlled 10-Arm Multimodal Graph Ablation Results (N = 335)

Project Objective 2 (PO2) evaluates whether modeling the transmission network as a graph structure provides measurable predictive uplift beyond biographical text. Table 6.2 reports the full 10-arm benchmark executed across 25 paired cross-validation folds on the condensed multimodal cohort ($N = 335$ nodes, 354 undirected transmission edges).

### Table 6.2: Benchmark Results Across All 10 Experimental Arms (N = 335, 25 Paired Folds)

| Arm ID | Model Identifier | Model Architecture | Macro-F1 (Mean ± SD) | 95% Confidence Interval | Accuracy (%) | Quadratic $\kappa$ | Consensus F1 ($N=181$) |
|---|---|---|---|---|---|---|---|
| **B0** | Majority Baseline | Constant Majority | **0.2900 ± 0.0016** | [0.2894, 0.2907] | 77.01% | 0.0000 | 0.3717 |
| **B1** | Stratified Random | Empirical Sampling | **0.3383 ± 0.0568** | [0.3148, 0.3617] | 62.57% | 0.0285 | 0.3484 |
| **B2** | Name + Meta Probe | Logistic Regression | **0.3806 ± 0.0618** | [0.3551, 0.4061] | 52.18% | 0.0248 | 0.3216 |
| **P1-A** | Lexical TF-IDF LR | Logistic Regression | **0.5246 ± 0.0278** | [0.5131, 0.5360] | **83.82%** | **0.5396** | **0.5113** |
| **P1-B** | AraBERT Semantic LR | Logistic Regression | **0.4958 ± 0.0800** | [0.4628, 0.5288] | 74.33% | 0.4614 | 0.4580 |
| **P1-B_MLP** | AraBERT Semantic MLP | 2-layer MLP (Hidden: 64) | **0.4511 ± 0.0782** | [0.4188, 0.4833] | 70.21% | 0.3992 | 0.4335 |
| **G1** | Graph Features MLP | 2-layer MLP (Hidden: 32) | **0.3628 ± 0.0740** | [0.3323, 0.3934] | 49.67% | 0.0698 | 0.3337 |
| **G2** | Graph GCN | 2-layer GCN (Hidden: 32) | **0.3536 ± 0.0476** | [0.3339, 0.3732] | 45.01% | 0.0583 | 0.2877 |
| **P2-A** | Early Fusion MLP | 2-layer MLP (Hidden: 64) | **0.4763 ± 0.0898** | [0.4392, 0.5133] | 68.12% | 0.3305 | 0.4571 |
| **P2-B_Concat** | Multimodal GCN Concat | 2-layer GCN (Hidden: 64) | **0.4439 ± 0.0964** | [0.4041, 0.4837] | 63.22% | 0.1999 | 0.4180 |
| **P2-B_Attn** | Multimodal GCN Attention | Cross-Modal GNN Attention | **0.4723 ± 0.0838** | [0.4377, 0.5069] | 69.13% | 0.2760 | 0.4464 |
| **P2-C** | Topology Isolation MLP | 2-layer MLP (Hidden: 16) | **0.4414 ± 0.0951** | [0.4021, 0.4806] | 69.73% | 0.3248 | 0.4559 |
| **P2-D** | Metadata Isolation MLP | 2-layer MLP (Hidden: 16) | **0.4966 ± 0.0640** | [0.4702, 0.5230] | 70.45% | 0.3449 | 0.4906 |

---

## 6.3 Per-Class Performance and Confusion Matrix Dynamics

Table 6.3 reports the granular per-class F1-scores across all three reliability tiers (*Below Ṣadūq*, *Ṣadūq*, and *Thiqah*).

### Table 6.3: Per-Class F1 Breakdown Across Tiers on the Condensed Cohort (N = 335)

| Arm ID | Model Identifier | Below Ṣadūq ($N=13$) | Ṣadūq ($N=64$) | Thiqah ($N=258$) |
|---|---|---|---|---|
| **B0** | Majority Baseline | 0.0000 | 0.0000 | 0.8701 |
| **B1** | Stratified Random | 0.0633 | 0.1860 | 0.7655 |
| **B2** | Name + Meta Probe | 0.1901 | 0.3112 | 0.6406 |
| **P1-A** | Lexical TF-IDF LR | 0.0000 | **0.6633** | **0.9104** |
| **P1-B** | AraBERT Semantic LR | 0.1660 | 0.4648 | 0.8566 |
| **P1-B_MLP** | AraBERT Semantic MLP | 0.1658 | 0.3564 | 0.8310 |
| **G1** | Graph Features MLP | 0.1475 | 0.3170 | 0.6240 |
| **G2** | Graph GCN | 0.1508 | 0.3644 | 0.5456 |
| **P2-A** | Early Fusion MLP | 0.2216 | 0.4058 | 0.8014 |
| **P2-B_Concat** | Multimodal GCN Concat | 0.2047 | 0.3853 | 0.7417 |
| **P2-B_Attn** | Multimodal GCN Attention | 0.2077 | 0.4098 | 0.7994 |
| **P2-C** | Topology Isolation MLP | 0.1271 | 0.3724 | 0.8247 |
| **P2-D** | Metadata Isolation MLP | **0.2736** | 0.3982 | 0.8180 |

The per-class performance data illustrates critical model behaviors:
* **The Minority Class Challenge:** On the condensed cohort ($N=335$), the minority class (*Below Ṣadūq*) contains only 13 historical narrators (3.9% of the cohort). The lexical model (P1-A) failed to classify this minority class ($F1 = 0.0000$) due to heavy regularized penalization, but achieved the highest overall accuracy (83.82%) and *Ṣadūq* discrimination ($F1 = 0.6633$).
* **Neural Recall of Minority Narrators:** Neural network models utilizing inverse-frequency loss weighting (P2-A, P2-D, and P2-B_Attn) successfully recalled minority unreliable narrators, achieving minority F1-scores between 0.2047 and 0.2736.
* **Bounded Ordinal Confusion:** Inspection of out-of-fold confusion matrices indicates that errors were strictly bounded within adjacent classes. Trustworthy narrators (*Thiqah*) were occasionally misclassified as *Ṣadūq*, but were never misclassified as *Below Ṣadūq*. Similarly, *Below Ṣadūq* narrators were never misclassified as *Thiqah*.

---

## 6.4 Formal Statistical Hypothesis Testing Decisions

Table 6.4 details the formal pairwise hypothesis testing results across all 25 paired cross-validation folds ($df = 24$, $\alpha = 0.05$).

### Table 6.4: Paired Two-Tailed Hypothesis Testing Results Across 25 Paired Folds (df = 24)

| Research Hypothesis | Paired Comparison (Arm A vs. Arm B) | Mean Difference $\bar{\Delta}$ | 95% Confidence Interval | Paired $t$-statistic | $p$-value | Empirical Decision |
|---|---|---|---|---|---|---|
| **$H_1$: Semantic vs Lexical (N=335)** | P1-B (AraBERT) vs P1-A (TF-IDF) | **-0.0288** | [-0.0634, +0.0058] | $t = -1.7166$ | $p = 0.0989$ | **Fail to Reject $H_0$** |
| **$H_{2a}$: Message Passing Benefit** | G2 (Graph GCN) vs G1 (Features MLP) | **-0.0093** | [-0.0310, +0.0125] | $t = -0.8793$ | $p = 0.3880$ | **Fail to Reject $H_0$** |
| **$H_{2b}$: Multimodal Attention Benefit** | P2-B_Attn vs P2-A (Early Fusion MLP) | **-0.0040** | [-0.0427, +0.0347] | $t = -0.2134$ | $p = 0.8330$ | **Fail to Reject $H_0$** |
| **Control: Feature Augmentation** | P2-A (Fusion MLP) vs P1-B_MLP (Text MLP) | **+0.0252** | [-0.0170, +0.0674] | $t = +1.2328$ | $p = 0.2296$ | **Fail to Reject $H_0$** |
| **$H_1$ on Full Corpus (N=844)** | P1-B (AraBERT) vs P1-A (TF-IDF) | **-0.1187** | [-0.1369, -0.1005] | $t = -13.422$ | $p = 0.0000$ | **Reject $H_0$ (TF-IDF leads)** |

The hypothesis testing decisions resolve the core research questions:
1. **Resolution of RQ1 ($H_1$):** Contextual AraBERT does not outperform regularized lexical TF-IDF. On the condensed cohort ($N=335$), the difference is not statistically significant ($p = 0.0989$). On the large-scale corpus ($N=844$), TF-IDF significantly outperforms AraBERT ($p < 0.001$).
2. **Resolution of RQ2 ($H_{2a}$ and $H_{2b}$):** Graph neural network message passing provides no statistically significant benefit over non-relational baselines. GCN with transmission edges (G2) performs worse on average than an MLP operating on node features alone (G1, $p = 0.388$). Fusing text with graph topology via GNN attention (P2-B_Attn) yields no gain over non-graph early concatenation (P2-A, $p = 0.833$).

---

## 6.5 PO3 Downstream Isnād Chain Validation Results (N = 1,578 Chains)

Project Objective 3 (PO3) evaluates the practical utility of narrator-level predictions by propagating out-of-fold scores through 1,578 verified transmission chains from SanadSet containing $\ge 2$ cohort narrators (mean cohort depth: 2.17 narrators per chain). Table 6.5 details the chain-level results under weakest-link aggregation:

$$S_{\text{chain}} = \min_{i \in \text{chain}} y_i$$

### Table 6.5: Downstream Chain Validation Results Across 1,578 Historical Chains

| Arm ID | Model Identifier | Chain Tier Acc (%) | Chain Binary F1 | Chain MAE | Node MAE | Error Propagation Ratio | Spearman Rank $\rho$ |
|---|---|---|---|---|---|---|---|
| **B0** | Majority Baseline | 70.98% | 0.9796 | 0.3302 | 0.2687 | 1.2289 | 0.0000 |
| **B1** | Stratified Random | 39.80% | 0.9506 | 0.6318 | 0.4358 | 1.4497 | 0.0696 |
| **B2** | Name + Meta Probe | 27.88% | 0.7589 | 1.0279 | 0.6090 | 1.6879 | -0.0982 |
| **P1-A** | Lexical TF-IDF LR | **71.86%** | **0.9790** | **0.2902** | **0.1761** | 1.6480 | **0.5317** |
| **P1-B** | AraBERT Semantic LR | 56.78% | 0.9094 | 0.5089 | 0.2806 | 1.8135 | 0.3558 |
| **P2-A** | Early Fusion MLP | 64.51% | 0.9776 | 0.3663 | 0.3134 | **1.1686** | 0.3786 |
| **P2-B_Attn** | Multimodal GCN Attention | 63.88% | 0.9825 | 0.3720 | 0.3045 | 1.2217 | 0.2887 |

Key conclusions from the downstream chain evaluation include:
* **Strong Rank Preservation:** The lexical TF-IDF model achieved a statistically significant rank correlation of **$\rho = 0.5317$ ($p < 10^{-15}$)** with ground-truth chain tiers.
* **High Binary Chain Acceptance Accuracy:** All high-performing models achieved chain binary F1 scores exceeding **0.97** for distinguishing acceptable (*Maqbūl*) from rejected (*Mardūd*) narrations.
* **Error Dampening via Bounded Mistakes:** The Error Propagation Ratio remained tightly bounded (1.1686 to 1.6480). Because individual narrator prediction mistakes occurred exclusively between adjacent categories, downstream weakest-link aggregation prevented catastrophic failure cascades across entire transmission paths.
