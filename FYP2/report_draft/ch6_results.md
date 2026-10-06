# Chapter 6: Results

## 6.1 PO1 Large-Scale Biographical Text Benchmark Results (N = 819)

The primary goal of Project Objective 1 (PO1) is to determine how accurately machine learning models can predict canonical narrator reliability tiers from classical earlier critic biographical statements under strict zero-leakage conditions. Table 6.1 presents the evaluation results across all 819 historical scholars in the canonical Hadith-Viz corpus possessing verified *Taqrīb* grades and earlier critic commentary dated strictly to the 2nd to 4th centuries AH. All statements from al-Mizzī (d. 742 AH) and ten subsequent post-classical authorities are completely purged. Evaluations were executed across 25 paired runs (5-fold Stratified Cross-Validation repeated across 5 random seeds).

### Table 6.1: PO1 Large-Scale Biographical Text Benchmark Results (N = 819)

| Model ID | Model Architecture | Macro-F1 (Mean ± SD) | 95% Confidence Interval | Accuracy (%) | Quadratic Weighted $\kappa$ |
|---|---|---|---|---|---|
| **B0** | Majority Baseline | **0.2702 ± 0.0006** | [0.2699, 0.2704] | 68.13% | 0.0000 |
| **B1** | Stratified Random | **0.3261 ± 0.0398** | [0.3097, 0.3426] | 50.84% | -0.0195 |
| **P1-A** | Lexical TF-IDF LR | **0.6852 ± 0.0514** | [0.6640, 0.7064] | **80.59%** | **0.6697** |
| **P1-B** | AraBERT Semantic LR | **0.5635 ± 0.0373** | [0.5481, 0.5789] | 69.99% | 0.5008 |

Several key findings emerge from the large-scale biographical benchmark:
* **High Predictive Utility of Classical Evaluative Text:** Both machine learning models substantially outperform the majority baseline (B0: 0.2702 Macro-F1) and stratified chance (B1: 0.3261 Macro-F1). This demonstrates that 2nd to 4th century classical critic statements provide strong, consistent signals that reliably predict Ibn Ḥajar's 9th-century synthesis verdicts even after completely purging all later synthesizing summaries.
* **Superiority of Regularized Lexical Representations:** The lexical TF-IDF model (P1-A) achieved a superior Macro-F1 score of **0.6852 ± 0.0514** and an overall accuracy of **80.59%**, significantly outperforming the pretrained AraBERT semantic classifier (P1-B: **0.5635 ± 0.0373**).
* **Substantial Inter-Rater Agreement:** P1-A achieved a Quadratic Weighted Kappa of **$\kappa = 0.6697$**, reflecting substantial ordinal agreement with Ibn Ḥajar's canonical ratings. By contrast, AraBERT achieved moderate agreement ($\kappa = 0.5008$).
* **Correction of Hypothesis Direction:** In the formal paired comparison, AraBERT underperformed TF-IDF by $\bar{\Delta} = -0.1218$ ($t = -11.4153, p = 3.49 \times 10^{-11}$). The null hypothesis is rejected; the empirical evidence significantly favors the lexical baseline over AraBERT, contrary to the original alternative hypothesis ($H_1$). Classical Arabic jarḥ wa-taʿdīl terminology (`ثقة`, `ضعيف`) provides a dense, highly specialized vocabulary that linear n-gram models exploit with higher precision than general-domain transformer embeddings.

---

## 6.2 PO2 Controlled 13-Arm Multimodal Graph Ablation Results (N = 335)

Project Objective 2 (PO2) evaluates whether modeling the transmission network as a graph structure provides measurable predictive uplift beyond biographical text. Table 6.2 reports the full 13-arm benchmark executed across 25 paired cross-validation folds on the condensed multimodal cohort ($N = 335$ nodes, 177 unique undirected relationships representing 354 directed message edges in PyG).

### Table 6.2: Benchmark Results Across All 13 Experimental Arms (N = 335, 25 Paired Folds)

| Arm ID | Model Identifier | Model Architecture | Macro-F1 (Mean ± SD) | 95% Confidence Interval | Accuracy (%) | Quadratic $\kappa$ | Consensus F1 ($N=180$) |
|---|---|---|---|---|---|---|---|
| **B0** | Majority Baseline | Constant Majority | **0.2881 ± 0.0000** | [0.2881, 0.2881] | 76.12% | 0.0000 | 0.3593 |
| **B1** | Stratified Random | Empirical Sampling | **0.3272 ± 0.0601** | [0.3024, 0.3520] | 61.25% | -0.0133 | 0.3050 |
| **B2** | Name + Meta Probe | Logistic Regression | **0.3822 ± 0.0439** | [0.3641, 0.4003] | 50.69% | 0.0318 | 0.3436 |
| **P1-A** | Lexical TF-IDF LR | Logistic Regression | **0.5608 ± 0.0703** | [0.5318, 0.5898] | **80.96%** | **0.4932** | **0.5284** |
| **P1-B** | AraBERT Semantic LR | Logistic Regression | **0.4933 ± 0.0628** | [0.4674, 0.5192] | 72.84% | 0.4347 | 0.4478 |
| **P1-B_MLP** | AraBERT Semantic MLP | 2-layer MLP (Hidden: 64) | **0.4454 ± 0.0657** | [0.4183, 0.4725] | 68.60% | 0.3295 | 0.4037 |
| **G1** | Graph Features MLP | 2-layer MLP (Hidden: 64) | **0.3556 ± 0.0564** | [0.3323, 0.3788] | 47.64% | 0.0899 | 0.3030 |
| **G2** | Graph GCN | 2-layer GCN (Hidden: 64) | **0.3438 ± 0.0716** | [0.3142, 0.3733] | 44.30% | 0.0343 | 0.2755 |
| **P2-A** | Early Fusion MLP | 2-layer MLP (Hidden: 64) | **0.4810 ± 0.0684** | [0.4528, 0.5093] | 68.36% | 0.3602 | 0.4708 |
| **P2-B_Concat** | Multimodal GCN Concat | 2-layer GCN (Hidden: 64) | **0.4311 ± 0.0750** | [0.4002, 0.4621] | 64.96% | 0.2157 | 0.3979 |
| **P2-B_Attn** | Multimodal GCN Attention | Cross-Modal GNN Attention | **0.4478 ± 0.0670** | [0.4202, 0.4755] | 66.27% | 0.2549 | 0.4271 |
| **P2-C** | Topology Isolation MLP | 2-layer MLP (Hidden: 64) | **0.4531 ± 0.0727** | [0.4231, 0.4831] | 67.22% | 0.3318 | 0.4270 |
| **P2-D** | Metadata Isolation MLP | 2-layer MLP (Hidden: 64) | **0.4315 ± 0.0796** | [0.3987, 0.4644] | 65.97% | 0.2919 | 0.3980 |

---

## 6.3 Per-Class Performance and Confusion Matrix Dynamics

Table 6.3 reports the granular per-class F1-scores across all three reliability tiers (*Below Ṣadūq*, *Ṣadūq*, and *Thiqah*).

### Table 6.3: Per-Class F1 Breakdown Across Tiers on the Condensed Cohort (N = 335)

| Arm ID | Model Identifier | Below Ṣadūq ($N=20$) | Ṣadūq ($N=60$) | Thiqah ($N=255$) |
|---|---|---|---|---|
| **B0** | Majority Baseline | 0.0000 | 0.0000 | 0.8644 |
| **B1** | Stratified Random | 0.0294 | 0.1952 | 0.7570 |
| **B2** | Name + Meta Probe | 0.1785 | 0.3415 | 0.6267 |
| **P1-A** | Lexical TF-IDF LR | 0.1630 | **0.6170** | **0.9024** |
| **P1-B** | AraBERT Semantic LR | 0.1885 | 0.4352 | 0.8562 |
| **P1-B_MLP** | AraBERT Semantic MLP | 0.1548 | 0.3545 | 0.8269 |
| **G1** | Graph Features MLP | 0.1630 | 0.2865 | 0.6173 |
| **G2** | Graph GCN | 0.1559 | 0.3248 | 0.5506 |
| **P2-A** | Early Fusion MLP | **0.2682** | 0.3623 | 0.8126 |
| **P2-B_Concat** | Multimodal GCN Concat | 0.1609 | 0.3585 | 0.7739 |
| **P2-B_Attn** | Multimodal GCN Attention | 0.1975 | 0.3554 | 0.7906 |
| **P2-C** | Topology Isolation MLP | 0.2178 | 0.3321 | 0.8093 |
| **P2-D** | Metadata Isolation MLP | 0.1779 | 0.3170 | 0.7998 |

The per-class performance data illustrates critical model behaviors:
* **The Minority Class Challenge:** On the condensed cohort ($N=335$), the minority class (*Below Ṣadūq*) contains 20 historical narrators (5.97% of the cohort). The lexical model (P1-A) achieved the highest overall accuracy (80.96%) and *Ṣadūq* discrimination ($F1 = 0.6170$).
* **Neural Recall of Minority Narrators:** Neural network models utilizing inverse-frequency loss weighting (especially Early Fusion MLP P2-A) achieved the highest minority recall, reaching a Below Ṣadūq F1-score of **0.2682**.
* **Bounded Ordinal Confusion:** Inspection of out-of-fold confusion matrices indicates that errors were strictly bounded within adjacent classes. Trustworthy narrators (*Thiqah*) were occasionally misclassified as *Ṣadūq*, but rarely misclassified as *Below Ṣadūq*.

---

## 6.4 Formal Statistical Hypothesis Testing Decisions

Table 6.4 details the formal pairwise hypothesis testing results across all 25 paired cross-validation folds ($df = 24$, $\alpha = 0.05$).

### Table 6.4: Paired Two-Tailed Hypothesis Testing Results Across 25 Paired Folds (df = 24)

| Research Hypothesis | Paired Comparison (Arm A vs. Arm B) | Mean Difference $\bar{\Delta}$ | 95% Confidence Interval | Paired $t$-statistic | $p$-value | Empirical Decision |
|---|---|---|---|---|---|---|
| **$H_1$: Semantic vs Lexical (N=819)** | P1-B (AraBERT) vs P1-A (TF-IDF) | **-0.1218** | [-0.1438, -0.0998] | $t = -11.4153$ | $p < 0.0001$ | **Reject $H_0$ (TF-IDF leads)** |
| **$H_1$: Semantic vs Lexical (N=335)** | P1-B (AraBERT) vs P1-A (TF-IDF) | **-0.0675** | [-0.1051, -0.0299] | $t = -3.7035$ | $p = 0.00111$ | **Reject $H_0$ (TF-IDF leads)** |
| **$H_{0a}$: Network Features over Text** | P2-A (Early Fusion) vs P1-B_MLP (Text) | **+0.0357** | [-0.0011, +0.0724] | $t = +2.0044$ | $p = 0.05643$ | **Fail to Reject $H_0$** |
| **$H_{0b}$: Message Passing vs Early Fusion** | P2-B_Concat vs P2-A (Early Fusion) | **-0.0499** | [-0.0918, -0.0080] | $t = -2.4595$ | $p = 0.02150$ | **Fail to Reject $H_0$ (GNN degrades)** |
| **$H_{0b}$: Attention GCN vs Early Fusion** | P2-B_Attn vs P2-A (Early Fusion) | **-0.0332** | [-0.0656, -0.0008] | $t = -2.1130$ | $p = 0.04519$ | **Fail to Reject $H_0$ (GNN degrades)** |
| **GCN vs Graph MLP Alone** | G2 (Graph GCN) vs G1 (Features MLP) | **-0.0118** | [-0.0436, +0.0200] | $t = -0.7667$ | $p = 0.45076$ | **Fail to Reject $H_0$** |

The hypothesis testing decisions resolve the core research questions:
1. **Resolution of RQ1 ($H_1$):** Pretrained AraBERT does not outperform regularized lexical TF-IDF. Across both the large-scale benchmark ($N=819$, $p < 10^{-10}$) and the ablation cohort ($N=335$, $p = 0.00111$), lexical TF-IDF demonstrates statistically significant superiority.
2. **Resolution of RQ2 ($H_{0a}$ and $H_{0b}$):** Graph neural network message passing provides no statistically significant benefit over non-relational baselines. GCN with transmission edges (G2) performs worse on average than an MLP operating on node features alone (G1, $p = 0.451$). Fusing text with graph topology via GNN convolution or attention (P2-B_Concat and P2-B_Attn) significantly underperforms non-graph tabular concatenation (P2-A, $p = 0.0215$ and $p = 0.0452$).

---

## 6.5 PO3 Downstream Isnād Chain Validation Results (N = 1,578 Chains)

Project Objective 3 (PO3) evaluates the practical utility of narrator-level predictions by propagating out-of-fold scores through 1,578 verified transmission chains from SanadSet containing $\ge 2$ cohort narrators (mean cohort depth: 2.17 narrators per chain). Table 6.5 details the chain-level results under weakest-link aggregation:

$$S_{\text{chain}} = \min_{i \in \text{chain}} y_i$$

### Table 6.5: Downstream Chain Validation Results Across 1,578 Historical Chains

| Arm ID | Model Identifier | Chain Tier Acc (%) | Chain Binary F1 | Chain MAE | Node MAE | Error Propagation Ratio | Spearman Rank $\rho$ |
|---|---|---|---|---|---|---|---|
| **B0** | Majority Baseline | 69.58% | 0.9618 | 0.3777 | 0.2985 | 1.2653 | 0.0000 |
| **B1** | Stratified Random | 42.78% | 0.8951 | 0.7186 | 0.4806 | 1.4953 | -0.0562 |
| **B2** | Name + Meta Probe | 28.71% | 0.7092 | 1.0608 | 0.6657 | 1.5936 | -0.0641 |
| **P1-A** | Lexical TF-IDF LR | 59.57% | 0.8180 | 0.6521 | **0.2478** | 2.6319 | 0.1671 |
| **P1-B** | AraBERT Semantic LR | 58.30% | 0.8792 | 0.5513 | 0.3194 | 1.7261 | 0.2655 |
| **P2-A** | Early Fusion MLP | **62.10%** | **0.9378** | **0.4278** | 0.3463 | **1.2353** | **0.3918** |
| **P2-B_Attn** | Multimodal GCN Attention | 55.83% | 0.9084 | 0.5722 | 0.3343 | 1.7116 | 0.1001 |

Key conclusions from the downstream chain evaluation include:
* **The Canonical Selection Floor:** In canonical collections (*Kutub al-Sittah*), 92.6% of verified chains are acceptable by default (Tier 1 or Tier 2). As a result, the naive majority baseline (B0) achieves 69.58% exact tier accuracy and 0.9618 binary F1 despite possessing zero ranking capability ($\rho = 0.0000$). High binary F1 must therefore be interpreted strictly as an artifact of collection curation rather than model brilliance.
* **Preservation of Ordinal Rank:** The Early Fusion MLP (P2-A) achieved the highest chain tier accuracy among machine learning models (**62.10%**), the lowest chain MAE (**0.4278**), and the highest ordinal rank correlation (**$\rho = 0.3918$, $p < 10^{-15}$**), confirming genuine predictive discernment across transmission paths.
* **Error Dampening via Bounded Mistakes:** The Error Propagation Ratio for P2-A remained tightly bounded at 1.2353. Because narrator prediction errors occurred almost exclusively between adjacent ordinal tiers, downstream weakest-link aggregation prevented catastrophic failure cascades.
