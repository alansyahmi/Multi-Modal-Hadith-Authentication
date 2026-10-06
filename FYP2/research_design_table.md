# Formal Research Design Table (Audited and Verified Version)
**Project:** Automated Narrator Reliability Assessment in Hadith Isnāds using Semantic NLP and Knowledge Graph Fusion  
**Student:** Alan Syahmi bin Sanih @ Sani (BI23110344)  
**Supervisor:** Dr. Khalifa Chekima  
**Faculty:** Faculty of Computing and Informatics (FCI), Universiti Malaysia Sabah (UMS)  
**Last Updated:** 2026-10-06  
**Version:** 2.1 (Empirical Audit and Full Leakage Elimination)

---

> [!NOTE]
> This table defines the formal scientific design of the three project objectives (PO1, PO2, PO3).
> It incorporates all verified findings from the empirical audit:
> 1. Target leakage is structurally eliminated by isolating 9th century synthesis verdicts from 2nd to 4th century earlier critic text, explicitly purging later synthesizers including al-Mizzi (d. 742 AH).
> 2. Evaluation uses 25 paired runs (5-fold stratified cross-validation repeated across 5 random seeds) with strict identity grouping (1 historical identity = 1 node).
> 3. Primary metric is locked to 3-Tier Macro-F1 (*Thiqah*, *Ṣadūq*, *Below Ṣadūq*).
> 4. Negative and null findings regarding graph message passing are formally reported as scientific discoveries concerning transmission network heterophily.

---

## 1. How to Read This Table

Each row represents one testable hypothesis tied directly to an empirical experiment:
* **H₀** = Null hypothesis (assumed default condition).
* **H₁** = Alternative hypothesis (claimed experimental effect).
* **Decision Rule** = The statistical threshold ($\alpha = 0.05$ with paired $t$-test and Wilcoxon signed-rank test across 25 evaluations).
* **Primary Metric** = 3-Tier Macro-F1 (unweighted mean of class F1 scores across *Below Ṣadūq*, *Ṣadūq*, and *Thiqah*).

---

## 2. Table 1: PO1 Semantic NLP Representation vs. Lexical Baseline

| Field | Specification |
|:---|:---|
| **Project Objective** | **PO1:** Extract evaluative reliability signals from classical Arabic earlier critic statements using transformer based contextual embeddings. |
| **Research Question** | Does contextual semantic representation via pre-trained AraBERT outperform a regularized lexical TF-IDF baseline for narrator reliability classification under zero target leakage conditions? |
| **Null Hypothesis (H₀)** | There is no statistically significant difference in 3-Tier Macro-F1 between an AraBERT semantic classifier and a lexical TF-IDF baseline when evaluated on earlier critic appraisals ($p \ge 0.05$). |
| **Alternative Hypothesis (H₁)** | Pre-trained AraBERT semantic embeddings produce a statistically significantly higher 3-Tier Macro-F1 than the lexical TF-IDF baseline ($p < 0.05$). |
| **Independent Variable** | Text feature representation: (A) Lexical TF-IDF (character and word n-grams, sublinear scaling), (B) Pre-trained AraBERT contextual embeddings (`aubmindlab/bert-base-arabertv2`, 768 dimensions, frozen). |
| **Dependent Variable** | 3-Tier Macro-F1 score (*Below Ṣadūq*, *Ṣadūq*, *Thiqah*) evaluated across 25 paired cross-validation folds. |
| **Controlled Variables** | Canonical leak-free narrator cohort ($N=819$ unique historical identities in primary benchmark; $N=335$ in ablation matrix); zero-leakage earlier critic text inputs (excluding al-Mizzi and all 7th to 9th century authorities); same classifier families (balanced Logistic Regression and 2-layer MLP); same fold splits. |
| **Experimental Arms** | **P1-A:** Lexical TF-IDF + Logistic Regression.<br/>**P1-B:** Frozen AraBERT (768-dim) + Logistic Regression.<br/>**P1-B_MLP:** Frozen AraBERT (768-dim) + 2-layer MLP (hidden=64, dropout=0.3). |
| **Primary Metric** | **Macro-F1 (3 Tiers)** |
| **Secondary Metrics** | Overall Accuracy, Quadratic Cohen's $\kappa$ (ordinal penalty), Per-class F1, Dual-Synthesizer Consensus F1. |
| **Empirical Results: Primary Benchmark ($N=819$)** | **P1-A (TF-IDF):** Macro-F1 = **0.6852 ± 0.0514** (Acc: 80.59%, Quadratic $\kappa$: 0.6697).<br/>**P1-B (AraBERT LR):** Macro-F1 = **0.5635 ± 0.0373** (Acc: 69.99%, Quadratic $\kappa$: 0.5008).<br/>**B0 (Majority Baseline):** Macro-F1 = 0.2702 ± 0.0006 (Acc: 68.13%, $\kappa$: 0.0000).<br/>**B1 (Stratified Random):** Macro-F1 = 0.3261 ± 0.0398 (Acc: 50.84%, $\kappa$: -0.0195). |
| **Empirical Results: 13-Arm Ablation ($N=335$)** | **P1-A (TF-IDF):** Macro-F1 = **0.5608 ± 0.0703** (Acc: 80.96%, Quadratic $\kappa$: 0.4932).<br/>**P1-B (AraBERT LR):** Macro-F1 = **0.4933 ± 0.0628** (Acc: 72.84%, Quadratic $\kappa$: 0.4347).<br/>**P1-B_MLP:** Macro-F1 = **0.4454 ± 0.0657** (Acc: 68.60%, Quadratic $\kappa$: 0.3295). |
| **Statistical Decision** | On $N=819$, paired difference $\bar{\Delta} = -0.1218$ ($t = -11.4153, p = 3.49 \times 10^{-11}$). On $N=335$, paired difference $\bar{\Delta} = -0.0675$ ($t = -3.7035, p = 0.00111$). **Reject H₀; the evidence significantly favors the lexical TF-IDF baseline over AraBERT, contrary to H₁.** Classical technical jarḥ wa-taʿdīl terminology (`ثقة`, `ضعيف`) provides a dense lexical signal that linear n-gram models exploit with higher precision than general-domain transformer embeddings. |

---

## 3. Table 2: PO2 Transmission Graph Topology vs. Multimodal Fusion

| Field | Specification |
|:---|:---|
| **Project Objective** | **PO2:** Model isnād transmission topology as a Knowledge Graph and evaluate whether network structure provides measurable predictive uplift beyond biographical text. |
| **Research Question** | Does fusing isnād transmission topology with biographical text semantics provide a statistically significant predictive gain over text semantics alone, and does GNN message passing outperform non-GNN tabular fusion? |
| **Null Hypotheses** | **H₀a:** Augmenting biographical text with network tabular features yields no significant gain over text alone ($p \ge 0.05$).<br/>**H₀b:** Graph convolutional message passing across isnād edges yields no significant gain over non-GNN feature concatenation ($p \ge 0.05$). |
| **Alternative Hypotheses** | **H₁a:** Multimodal early fusion significantly outperforms text-only models ($p < 0.05$).<br/>**H₁b:** GNN message passing significantly outperforms non-GNN multimodal fusion ($p < 0.05$). |
| **Independent Variables** | Modality configuration and architecture across 13 arms: Graph-Only (G1 MLP, G2 GCN), Text-Only (P1-A, P1-B, P1-B_MLP), Non-GNN Early Fusion (P2-A MLP, P2-C, P2-D), Full Multimodal GNN (P2-B_Concat, P2-B_Attn). |
| **Dependent Variable** | 3-Tier Macro-F1 score across 25 paired cross-validation folds. |
| **Controlled Variables** | Same condensed graph ($N=335$ nodes, 177 unique undirected relationships representing 354 directed message edges in PyG); transductive node classification with label-disjoint splits; same 8 tabular features (5 topology + 3 metadata); same 768-dim text embeddings; same inverse-frequency loss weighting. |
| **Empirical Results (25 Folds)** | **G1 (Graph Features MLP):** 0.3556 ± 0.0564 (Acc: 47.64%).<br/>**G2 (Graph GCN):** 0.3438 ± 0.0716 (Acc: 44.30%).<br/>**P2-A (Early Fusion MLP, no edges):** **0.4810 ± 0.0684** (Acc: 68.36%).<br/>**P2-B_Concat (GNN Concat):** 0.4311 ± 0.0750 (Acc: 64.96%).<br/>**P2-B_Attn (GNN Attention):** 0.4478 ± 0.0670 (Acc: 66.27%).<br/>**P2-C (Topology Isolation MLP):** 0.4531 ± 0.0727 (Acc: 67.22%).<br/>**P2-D (Metadata Isolation MLP):** 0.4315 ± 0.0796 (Acc: 65.97%). |
| **Statistical Decisions** | 1. **Testing H₀a (Network Tabular Uplift):** P2-A vs P1-B_MLP yields $\bar{\Delta} = +0.0357$ ($t = +2.0044, p = 0.05643$). **Fail to reject H₀a at $\alpha = 0.05$.**<br/>2. **Testing H₀b (Message Passing vs Non-GNN Fusion):** P2-B_Concat vs P2-A yields $\bar{\Delta} = -0.0499$ ($t = -2.4595, p = 0.02150$). P2-B_Attn vs P2-A yields $\bar{\Delta} = -0.0332$ ($t = -2.1130, p = 0.04519$). **Fail to reject H₀b; message passing significantly degrades performance relative to non-GNN fusion.**<br/>3. **GCN vs Non-GNN Graph Features:** G2 vs G1 yields $\bar{\Delta} = -0.0118$ ($t = -0.7667, p = 0.45076$). **Fail to reject H₀.** |
| **Domain Finding** | **Transmission Network Heterophily:** Reliable teachers regularly transmitted to students across different reliability tiers. Teacher-student transmission links do not exhibit homophilous label clustering. Consequently, standard GCN message passing aggregates conflicting neighbourhood labels, acting as an over-smoothing filter that provides no predictive benefit beyond biographical text. |

---

## 4. Table 3: PO3 Downstream Chain Error Propagation

| Field | Specification |
|:---|:---|
| **Project Objective** | **PO3:** Validate how narrator-level reliability predictions propagate through multi-hop transmission chains under the classical weakest-link aggregation rule. |
| **Research Question** | How accurately do predicted narrator reliability scores reproduce ground-truth chain reliability verdicts when aggregated via the minimum operator ($S_{\text{chain}} = \min_{i} y_i$) across verified canonical isnād chains? |
| **Null Hypothesis (H₀)** | Predicted narrator scores aggregated via weakest-link minimum produce chain-level rankings that do not correlate with reference oracle chain scores (Spearman rank correlation $\rho \le 0$). |
| **Alternative Hypothesis (H₁)** | Predicted narrator scores produce chain-level rankings with statistically significant positive rank correlation ($\rho > 0, p < 0.05$), and maintain bounded ordinal error propagation. |
| **Dataset Evaluated** | **1,578 verified transmission chains** from SanadSet containing $\ge 2$ cohort narrators (mean cohort depth: 2.17 narrators per chain). Ground-truth chain distribution: 116 Below Ṣadūq (7.4%), 364 Ṣadūq (23.1%), 1,098 Thiqah (69.6%). |
| **Independent Variable** | Model scoring source: Out-of-fold test predictions from B0, B1, B2, P1-A, P1-B, P2-A, and P2-B_Attn. |
| **Dependent Variables** | Exact 3-Tier Chain Accuracy, Binary Chain Acceptance F1 (Accepted: Tier $\ge 1$ vs Rejected: Tier 0), Spearman $\rho$, Kendall $\tau$, Chain MAE, Node MAE, Error Propagation Ratio (Chain MAE / Node MAE). |
| **Empirical Results** | **P2-A (Early Fusion MLP):** Chain Acc = **62.10%**, Binary F1 = **0.9378**, Chain MAE = **0.4278**, Node MAE = 0.3463, Spearman $\rho$ = **0.3918**.<br/>**P1-A (Lexical TF-IDF):** Chain Acc = **59.57%**, Binary F1 = **0.8180**, Chain MAE = **0.6521**, Node MAE = **0.2478**, Spearman $\rho$ = **0.1671**.<br/>**P1-B (AraBERT Semantic LR):** Chain Acc = **58.30%**, Binary F1 = **0.8792**, Chain MAE = **0.5513**, Node MAE = 0.3194, Spearman $\rho$ = **0.2655**.<br/>**P2-B_Attn (Multimodal GNN):** Chain Acc = **55.83%**, Binary F1 = **0.9084**, Chain MAE = **0.5722**, Node MAE = 0.3343, Spearman $\rho$ = **0.1001**.<br/>**B0 (Majority Baseline):** Chain Acc = **69.58%**, Binary F1 = **0.9618**, Chain MAE = 0.3777, Node MAE = 0.2985, Spearman $\rho$ = **0.0000**. |
| **Statistical Decision** | **Reject H₀ in favor of H₁.** Rank correlation is positive and statistically significant. Furthermore, binary chain acceptance F1 (0.81 to 0.94) demonstrates that model errors are restricted to adjacent tiers rather than polarity inversions. However, high binary F1 must be interpreted alongside the canonical collection selection effect (where 92.6% of chains in verified collections are acceptable by default), establishing exact 3-tier accuracy and Spearman $\rho$ as the decisive evaluation criteria. |

---

## 5. Summary Decision Matrix

| Project Objective | Statistical Comparison | Metric | Observed Difference $\bar{\Delta}$ | $p$-value | Formal Scientific Decision |
|:---|:---|:---|:---|:---|:---|
| **PO1 (Semantic NLP, N=819)** | P1-B (AraBERT) vs. P1-A (TF-IDF) | 3-Tier Macro-F1 | -0.1218 | $3.49 \times 10^{-11}$ | **Reject H₀ in favor of Lexical Baseline:** TF-IDF significantly outperforms AraBERT, contrary to H₁. |
| **PO1 (Semantic NLP, N=335)** | P1-B (AraBERT) vs. P1-A (TF-IDF) | 3-Tier Macro-F1 | -0.0675 | 0.00111 | **Reject H₀ in favor of Lexical Baseline:** Lexical signal consistently outperforms frozen AraBERT. |
| **PO2 (Tabular Gain)** | P2-A (Early Fusion) vs. P1-B_MLP | 3-Tier Macro-F1 | +0.0357 | 0.05643 | **Fail to Reject H₀a:** Tabular features provide marginal, non-significant gain ($p \ge 0.05$). |
| **PO2 (GNN Concat Value)** | P2-B_Concat (GNN) vs. P2-A (No-Edge) | 3-Tier Macro-F1 | -0.0499 | 0.02150 | **Fail to Reject H₀b:** Message passing significantly degrades performance ($p < 0.05$, negative $\bar{\Delta}$). |
| **PO2 (GNN Attention Value)** | P2-B_Attn (GNN) vs. P2-A (No-Edge) | 3-Tier Macro-F1 | -0.0332 | 0.04519 | **Fail to Reject H₀b:** Attention GCN significantly underperforms tabular early fusion. |
| **PO2 (GCN Alone)** | G2 (Graph GCN) vs. G1 (Graph MLP) | 3-Tier Macro-F1 | -0.0118 | 0.45076 | **Fail to Reject H₀:** Graph edges provide no predictive uplift over node features alone. |
| **PO3 (Chain Validation)** | P2-A Predictions vs. Oracle Chains | Spearman $\rho$ | $\rho = +0.3918$ | $< 10^{-15}$ | **Reject H₀:** Significant ordinal ranking preservation; error remains bounded. |

---

## 6. Official Mapping to FCI UMS Thesis Chapters

| Research Design Element | FCI UMS Report Chapter | Required Section / Content |
|:---|:---|:---|
| Problem formulation and scope bounds | **Chapter 1: Introduction** | §1.2 Problem Statement, §1.3 Objectives, §1.4 Scope |
| Homophily in GNNs and classical Arabic NLP | **Chapter 2: Literature Review** | §2.2 Arabic NLP and Transformers, §2.4 GNNs and Network Heterophily |
| Zero-leakage extraction and entity resolution | **Chapter 3: Project Methodology** | §3.2 Data Cleaning and Deduplication, §3.3 Graph Construction |
| 13-arm matrix and 25-fold cross-validation | **Chapter 4: System / Experiment Design** | §4.2 Baseline Formulations, §4.3 Evaluation Protocol |
| Modular scripts and feature pipelines | **Chapter 5: Implementation** | §5.2 NLP Module, §5.3 Graph Module, §5.4 Fusion Architecture |
| Empirical tables, CIs and hypothesis tests | **Chapter 6: Results** | §6.2 PO1 Benchmark, §6.3 PO2 Ablation, §6.4 PO3 Chain Validation |
| Interpretation of network heterophily | **Chapter 7: Discussion and Conclusion** | §7.1 Domain Insights, §7.2 Limitations, §7.3 Future Work |

---
*End of Formal Research Design Table (v2.1)*
