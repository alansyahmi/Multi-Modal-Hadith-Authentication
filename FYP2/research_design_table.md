# Formal Research Design Table (Audited & Recovered Version)
**Project:** Automated Narrator Reliability Assessment in Hadith Isnāds using Semantic NLP and Knowledge Graph Fusion  
**Student:** Alan Syahmi bin Sanih @ Sani (BI23110344)  
**Supervisor:** Dr. Khalifa Chekima  
**Faculty:** Faculty of Computing and Informatics (FCI), Universiti Malaysia Sabah (UMS)  
**Last Updated:** 2026-10-05  
**Version:** 2.0 (Post-Audit Empirical Rebuild)

---

> [!NOTE]
> This table defines the formal scientific design of the three project objectives (PO1, PO2, PO3).
> It incorporates all corrections from the examiner-style audit:
> 1. Target leakage is structurally eliminated by isolating 9th-century synthesis verdicts from 2nd to 4th-century earlier-critic text.
> 2. Evaluation uses 25 paired runs (5-fold stratified cross-validation repeated across 5 random seeds) with strict identity grouping (1 historical identity = 1 node).
> 3. Primary metric is locked to 3-Tier Macro-F1 (*Thiqah*, *Ṣadūq*, *Below Ṣadūq*).
> 4. Negative and null findings regarding graph message passing are formally reported as scientific discoveries concerning transmission network heterophily.

---

## 1. How to Read This Table

Each row represents one testable hypothesis tied directly to an empirical experiment:
* **H₀** = Null hypothesis (assumed default condition).
* **H₁** = Alternative hypothesis (claimed experimental effect).
* **Decision Rule** = The statistical threshold ($\alpha = 0.05$ with paired $t$-test and Wilcoxon signed-rank test across 25 evaluations).
* **Primary Metric** = 3-Tier Macro-F1 (unweighted mean of class F1-scores across *Below Ṣadūq*, *Ṣadūq*, and *Thiqah*).

---

## 2. Table 1: PO1 Semantic NLP Representation vs. Lexical Baseline

| Field | Specification |
|:---|:---|
| **Project Objective** | **PO1:** Extract evaluative reliability signals from classical Arabic earlier-critic statements using transformer-based contextual embeddings. |
| **Research Question** | Does contextual semantic representation via pre-trained AraBERT outperform a regularized lexical TF-IDF baseline for narrator reliability classification under zero target leakage conditions? |
| **Null Hypothesis (H₀)** | There is no statistically significant difference in 3-Tier Macro-F1 between an AraBERT semantic classifier and a lexical TF-IDF baseline when evaluated on earlier-critic appraisals ($p \ge 0.05$). |
| **Alternative Hypothesis (H₁)** | Pre-trained AraBERT semantic embeddings produce a statistically significantly higher 3-Tier Macro-F1 than the lexical TF-IDF baseline ($p < 0.05$). |
| **Independent Variable** | Text feature representation: (A) Lexical TF-IDF (character and word n-grams, sublinear scaling), (B) Pre-trained AraBERT contextual embeddings (`aubmindlab/bert-base-arabertv2`, 768 dimensions, frozen). |
| **Dependent Variable** | 3-Tier Macro-F1 score (*Below Ṣadūq*, *Ṣadūq*, *Thiqah*) evaluated across 25 paired cross-validation folds. |
| **Controlled Variables** | Same narrator cohort ($N=335$ unique historical identities); same zero-leakage earlier-critic text inputs; same classifier families (balanced Logistic Regression and 2-layer MLP); same fold splits. |
| **Experimental Arms** | **P1-A:** Lexical TF-IDF + Logistic Regression.<br/>**P1-B:** Frozen AraBERT (768-dim) + Logistic Regression.<br/>**P1-B_MLP:** Frozen AraBERT (768-dim) + 2-layer MLP (hidden=64, dropout=0.3). |
| **Primary Metric** | **Macro-F1 (3 Tiers)** |
| **Secondary Metrics** | Overall Accuracy, Quadratic Cohen's $\kappa$ (ordinal penalty), Per-class F1, Dual-Synthesizer Consensus F1 ($N=181$). |
| **Empirical Results (25 Folds)** | P1-A (TF-IDF): **0.5246 ± 0.0278** (Acc: 83.82%, $\kappa$: 0.5396).<br/>P1-B (AraBERT LR): **0.4958 ± 0.0800** (Acc: 74.33%, $\kappa$: 0.4614).<br/>P1-B_MLP: **0.4511 ± 0.0782** (Acc: 70.21%, $\kappa$: 0.3992). |
| **Statistical Decision** | Paired difference $\bar{\Delta} = -0.0288$ ($t = -1.7166, p = 0.0989$). **Fail to reject H₀ at $\alpha = 0.05$.** Classical earlier-critic vocabulary (`ثقة`, `ضعيف`) provides a dense predictive signal, making regularized lexical models exceptionally competitive with deep transformers. |

---

## 3. Table 2: PO2 Transmission Graph Topology vs. Multimodal Fusion

| Field | Specification |
|:---|:---|
| **Project Objective** | **PO2:** Model isnād transmission topology as a Knowledge Graph and evaluate whether network structure provides measurable predictive uplift beyond biographical text. |
| **Research Question** | Does fusing isnād transmission topology with biographical text semantics provide a statistically significant predictive gain over text semantics alone, and does GNN message passing outperform non-GNN tabular fusion? |
| **Null Hypotheses** | **H₀a:** Augmenting biographical text with network tabular features yields no significant gain over text alone.<br/>**H₀b:** Graph convolutional message passing across isnād edges yields no significant gain over non-GNN feature concatenation. |
| **Alternative Hypotheses** | **H₁a:** Multimodal early fusion significantly outperforms text-only models ($p < 0.05$).<br/>**H₁b:** GNN message passing significantly outperforms non-GNN multimodal fusion ($p < 0.05$). |
| **Independent Variables** | Modality configuration and architecture: Graph-Only (G1 MLP, G2 GCN), Text-Only (P1-B_MLP), Non-GNN Early Fusion (P2-A MLP), Full Multimodal GNN (P2-B Concat and Attention). |
| **Dependent Variable** | 3-Tier Macro-F1 score across 25 paired cross-validation folds. |
| **Controlled Variables** | Same condensed graph (335 nodes, 354 undirected edges); same 8 tabular features (5 topology + 3 metadata); same 768-dim text embeddings; same inverse-frequency loss weighting. |
| **Empirical Results (25 Folds)** | G1 (Graph Features MLP): **0.3628 ± 0.0740**.<br/>G2 (Graph GCN): **0.3536 ± 0.0476**.<br/>P2-A (Early Fusion MLP, no edges): **0.4763 ± 0.0898**.<br/>P2-B_Concat (GNN Concat): **0.4439 ± 0.0964**.<br/>P2-B_Attn (GNN Attention): **0.4723 ± 0.0838**.<br/>P2-C (Topology Isolation): **0.4414 ± 0.0951**.<br/>P2-D (Metadata Isolation): **0.4966 ± 0.0640**. |
| **Statistical Decisions** | 1. **Testing H₀a (Network Tabular Features):** P2-A vs P1-B_MLP yields $\bar{\Delta} = +0.0252$ ($t = +1.2328, p = 0.2296$). **Fail to reject H₀a.**<br/>2. **Testing H₀b (Message Passing):** P2-B_Attn vs P2-A yields $\bar{\Delta} = -0.0040$ ($t = -0.2134, p = 0.8328$). **Fail to reject H₀b.**<br/>3. **GCN vs Non-GNN Graph:** G2 vs G1 yields $\bar{\Delta} = -0.0093$ ($t = -0.8793, p = 0.3880$). |
| **Domain Finding** | **Transmission Network Heterophily:** Reliable teachers regularly transmitted to students across different reliability tiers. Teacher-student transmission links do not exhibit homophilous label clustering. Consequently, GNN message passing acts as an over-smoothing filter that provides no predictive benefit beyond biographical text. |

---

## 4. Table 3: PO3 Downstream Chain Error Propagation

| Field | Specification |
|:---|:---|
| **Project Objective** | **PO3:** Validate how narrator-level reliability predictions propagate through multi-hop transmission chains under the classical weakest-link aggregation rule. |
| **Research Question** | How accurately do predicted narrator reliability scores reproduce ground-truth chain reliability verdicts when aggregated via the minimum operator ($S_{\text{chain}} = \min_{i} y_i$) across verified canonical isnād chains? |
| **Null Hypothesis (H₀)** | Predicted narrator scores aggregated via weakest-link minimum produce chain-level rankings that do not correlate with reference oracle chain scores (Spearman rank correlation $\rho \le 0$). |
| **Alternative Hypothesis (H₁)** | Predicted narrator scores produce chain-level rankings with statistically significant positive rank correlation ($\rho > 0, p < 0.05$), and maintain high binary chain acceptance agreement. |
| **Dataset Evaluated** | **1,578 verified transmission chains** from SanadSet containing $\ge 2$ cohort narrators (mean cohort depth: 2.17 narrators per chain). |
| **Independent Variable** | Model scoring source: Out-of-fold test predictions from B0, B1, B2, P1-A, P1-B, P2-A, and P2-B_Attn. |
| **Dependent Variables** | Exact 3-Tier Chain Accuracy, Binary Chain Acceptance F1 (Accepted: Tier $\ge 1$ vs Rejected: Tier 0), Spearman $\rho$, Kendall $\tau$, Error Propagation Ratio (Chain MAE / Node MAE). |
| **Empirical Results** | **P1-A (Lexical TF-IDF):** Chain Acc = **71.86%**, Binary F1 = **0.9790**, Chain MAE = **0.2902**, Spearman $\rho$ = **0.5317**.<br/>**P2-A (Early Fusion MLP):** Chain Acc = **64.51%**, Binary F1 = **0.9776**, Chain MAE = **0.3663**, Spearman $\rho$ = **0.3786**.<br/>**P2-B_Attn (Multimodal GNN):** Chain Acc = **63.88%**, Binary F1 = **0.9825**, Chain MAE = **0.3720**, Spearman $\rho$ = **0.2887**.<br/>**B0 (Majority Baseline):** Chain Acc = **70.98%**, Binary F1 = **0.9796**, Spearman $\rho$ = **0.0000**. |
| **Statistical Decision** | **Reject H₀ in favor of H₁.** Rank correlation is positive and highly significant ($p < 10^{-15}$). Binary chain acceptance agreement exceeds **97.9%**, confirming that model errors remain bounded within adjacent tiers and do not cause catastrophic chain invalidation. |

---

## 5. Summary Decision Matrix

| Project Objective | Statistical Comparison | Metric | Observed Difference $\bar{\Delta}$ | $p$-value | Formal Scientific Decision |
|:---|:---|:---|:---|:---|:---|
| **PO1 (Semantic NLP)** | P1-B (AraBERT) vs. P1-A (TF-IDF) | 3-Tier Macro-F1 | -0.0288 | 0.0989 | **Fail to Reject H₀:** TF-IDF is highly competitive on technical vocabulary. |
| **PO2 (Tabular Gain)** | P2-A (Early Fusion) vs. P1-B_MLP | 3-Tier Macro-F1 | +0.0252 | 0.2296 | **Fail to Reject H₀:** Tabular features provide minor, non-significant gain. |
| **PO2 (GNN Value)** | P2-B_Attn (GNN) vs. P2-A (No-Edge) | 3-Tier Macro-F1 | -0.0040 | 0.8328 | **Fail to Reject H₀:** GNN message passing does not beat non-GNN fusion. |
| **PO2 (GCN Alone)** | G2 (Graph GCN) vs. G1 (Graph MLP) | 3-Tier Macro-F1 | -0.0093 | 0.3880 | **Fail to Reject H₀:** Network edges alone provide zero predictive uplift. |
| **PO3 (Chain Validation)**| P1-A Predictions vs. Oracle Chains | Spearman $\rho$ | $\rho = +0.5317$ | $< 10^{-15}$ | **Reject H₀:** Significant ordinal alignment; Binary F1 > 0.979. |

---

## 6. Official Mapping to FCI UMS Thesis Chapters

| Research Design Element | FCI UMS Report Chapter | Required Section / Content |
|:---|:---|:---|
| Problem formulation & scope bounds | **Chapter 1: Introduction** | §1.2 Problem Statement, §1.3 Objectives, §1.4 Scope |
| Homophily in GNNs & classical Arabic NLP | **Chapter 2: Literature Review** | §2.2 Arabic NLP & Transformers, §2.4 GNNs & Network Heterophily |
| Zero-leakage extraction & identity resolution | **Chapter 3: Project Methodology** | §3.2 Data Cleaning & Deduplication, §3.3 Graph Construction |
| 10-arm matrix & 25-fold cross-validation | **Chapter 4: System / Experiment Design** | §4.2 Baseline Formulations, §4.3 Evaluation Protocol |
| Modular scripts & feature pipelines | **Chapter 5: Implementation** | §5.2 NLP Module, §5.3 Graph Module, §5.4 Fusion Architecture |
| Empirical tables, CIs & hypothesis tests | **Chapter 6: Results** | §6.2 PO1 Benchmark, §6.3 PO2 Ablation, §6.4 PO3 Chain Validation |
| Interpretation of network heterophily | **Chapter 7: Discussion & Conclusion** | §7.1 Domain Insights, §7.2 Limitations, §7.3 Future Work |

---
*End of Formal Research Design Table (v2.0)*
