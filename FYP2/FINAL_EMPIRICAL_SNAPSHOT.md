# Final Empirical Snapshot and Canonical Research Freeze

* **Project Title**: A Multi-Modal Framework for Quantifying Narrator Reliability in *ʿIlm ar-Rijal* Using Semantic Sentiment Analysis and Graph Neural Networks
* **Candidate**: Alan Syahmi bin Sanih @ Sani (Matric: BI23110344)
* **Supervisor**: Dr. Khalifa Chekima
* **Institution**: Universiti Malaysia Sabah (UMS), Faculty of Computing and Informatics (FCI)
* **Degree Program**: Bachelor of Computer Science with Honours (Data Science)
* **Freeze Date**: October 6, 2026
* **Git Commit Hash**: `808cfc6da0f07181dfdc7726f96116e5d17c9455`

---

## 1. Executive Research Freeze Declaration

This document constitutes the definitive, immutable empirical record for the FYP2 research dissertation. All preprocessing funnels, temporal firewalls, feature representations, cross-validation partitions, statistical hypothesis tests, downstream chain evaluations, and visual figures documented herein are finalized, mathematically audited, and locked against future modifications.

---

## 2. Canonical Cohort Definitions and Dataset Specifications

### Table S.1: Canonical Cohorts and Sample Sizes
| Cohort Level | Target Objective | Population Size | Class Distribution (Tiers 2 / 1 / 0) | Inclusion and Firewall Criteria |
| :--- | :---: | :---: | :--- | :--- |
| **Level 1: Canonical Text** | PO1 Benchmark | $N = 819$ | 558 *Thiqah* (68.13%)<br>189 *Ṣadūq* (23.08%)<br>72 *Below Ṣadūq* (8.79%) | Non-Companions possessing verified *Taqrīb* grades and substantive 2nd-4th C. AH classical critic text; 11 post-classical authorities purged. |
| **Level 2: Multimodal Core** | PO2 13-Arm Ablation | $N = 335$ | 255 *Thiqah* (76.12%)<br>60 *Ṣadūq* (17.91%)<br>20 *Below Ṣadūq* (5.97%) | Intersecting Level 1 text, 3 structured metadata attributes, and connected transmission graph (177 undirected edges / 354 PyG links). |
| **Level 3: Evaluated Chains** | PO3 Downstream Auth | $N = 1,578$ | Mean 2.17 multimodal narrators per chain | Audited sequential qualifying isnād sample drawn from 16,754 multi-narrator candidate chains in the 80,731-isnād corpus. |
| **Dual Synthesizer Cohort** | Disagreement Study | $N = 238$ | 180 Consensus (75.63%)<br>58 Disagreements (24.37%) | Shared historical transmitters evaluated independently by Ibn Ḥajar (*Taqrīb*) and al-Dhahabī (*al-Kāshif*). |

---

## 3. Preprocessing Firewalls and Mapping Logic

### 3.1 Strict Temporal and Target Firewall
* **Excluded Later Compilers**: Eleven 7th to 9th century AH authorities were purged from all textual inputs:
  1. Ibn Ḥajar al-ʿAsqalānī (d. 852 AH)
  2. Shams al-Dīn al-Dhahabī (d. 748 AH)
  3. Yūsuf ibn al-Zakī al-Mizzī (d. 742 AH) - 676 quotes purged
  4. Mughulṭāy ibn Qalīj (d. 762 AH)
  5. Zayn al-Dīn al-ʿIrāqī (d. 806 AH)
  6. Khalīl ibn Kaykaldī al-ʿAlāʾī (d. 761 AH)
  7. Yaḥyā ibn Sharaf al-Nawawī (d. 676 AH)
  8. Burhān al-Dīn al-Ḥalabī (d. 841 AH)
  9. Ibn ʿAbd al-Hādī (d. 744 AH)
  10. Shams al-Dīn al-Sakhāwī (d. 902 AH)
  11. Jalāl al-Dīn al-Suyūṭī (d. 911 AH)
* **Retained Classical Critics (2nd to 4th Century AH)**: Yaḥyā ibn Maʿīn (d. 233), Aḥmad ibn Ḥanbal (d. 241), ʿAlī ibn al-Madīnī (d. 234), al-Bukhārī (d. 256), Abū Ḥātim al-Rāzī (d. 277), Abū Zurʿah al-Rāzī (d. 264), al-Nasāʾī (d. 303), al-Tirmidhī (d. 279), Ibn ʿAdī (d. 365), and al-Dāraquṭnī (d. 385).

### 3.2 Canonical Label Mapping (`src/canonical_data_pipeline.py`)
Negative-indicator precedence is strictly enforced prior to positive token matching:
* **Tier 0 (*Below Ṣadūq*)**: Matches negative indicators first: `ليس بثقة`, `ضعيف`, `متروك`, `كذاب`, `وضاع`, `منكر`, `واه`, `مجهول`, `ليس بالقوي`, `فيه مقال`, `سكتوا عنه`, `رمي بالتشيع`.
* **Tier 1 (*Ṣadūq*)**: Matches intermediate indicators: `صدوق`, `لا بأس به`, `ليس به بأس`, `محله الصدق`, `يخطئ`, `يهم`, `مقبول`, `حسن الحديث`.
* **Tier 2 (*Thiqah*)**: Matches high reliability indicators: `ثقة`, `ثبت`, `حافظ`, `متقن`, `حجة`, `عدل`, `إمام`.

---

## 4. Master Experimental Parameters and Evaluation Protocol

* **Evaluation Protocol**: 5x5 Repeated Stratified Cross-Validation.
* **Deterministic Random Seeds**: $\mathcal{S} = \{42, 123, 456, 789, 999\}$.
* **Total Paired Iterations**: $M = 25$ evaluations per model arm.
* **Graph Formulation**: Transductive node classification with label-disjoint training (80%) and test (20%) folds.
* **Class Weighting**: Inverse-frequency loss weighting applied across all neural architectures.
* **Hardware Profile**: Intel Core i7-10750H, 16GB RAM, NVIDIA GeForce RTX 2070 Max-Q (8GB VRAM), CUDA 12.4.

---

## 5. Definitive Empirical Benchmark Results

### 5.1 Project Objective 1: Lexical vs. Contextual Semantic NLP ($N = 819$, 25 Folds)
* **B0 (Majority Baseline)**: Macro-F1 = $0.2702 \pm 0.0006$, Accuracy = 68.13%, $\kappa_w = 0.0000$
* **B1 (Stratified Random)**: Macro-F1 = $0.3261 \pm 0.0398$, Accuracy = 50.84%, $\kappa_w = -0.0195$
* **P1-A (Lexical TF-IDF LR)**: **Macro-F1 = $0.6852 \pm 0.0514$**, Accuracy = **80.59%**, $\kappa_w = \mathbf{0.6697}$
* **P1-B (AraBERT Semantic LR)**: Macro-F1 = $0.5635 \pm 0.0373$, Accuracy = 69.99%, $\kappa_w = 0.5008$
* **Paired Hypothesis Test (AraBERT vs. TF-IDF)**:
  * Mean Difference: $\bar{\Delta} = -0.1218$ (95% CI: [-0.1438, -0.0998])
  * Paired $t$-test ($df=24$): $t = -11.4153$, $p = 3.49 \times 10^{-11}$
  * Wilcoxon Signed-Rank: $W = 0.0$, $p = 5.96 \times 10^{-8}$
  * Standardized Effect Size: Cohen's $d = -2.283$
  * **Formal Decision**: Reject $H_0$; empirical evidence significantly favors TF-IDF over AraBERT, contrary to $H_1$.

### 5.2 Project Objective 2: 13-Arm Multimodal Controlled Ablation ($N = 335$, 25 Folds)
* **B0 (Majority Baseline)**: Macro-F1 = 0.2881, Accuracy = 76.12%, $\kappa_w = 0.0000$
* **B1 (Stratified Random)**: Macro-F1 = 0.3272, Accuracy = 61.25%, $\kappa_w = -0.0133$
* **B2 (Name + Meta Probe)**: Macro-F1 = 0.3822, Accuracy = 50.69%, $\kappa_w = 0.0318$
* **G1 (Graph Features Alone MLP)**: Macro-F1 = 0.3556, Accuracy = 47.64%, $\kappa_w = 0.0899$
* **G2 (Graph GCN Message Passing)**: Macro-F1 = 0.3438, Accuracy = 44.30%, $\kappa_w = 0.0343$
* **P1-A (Lexical TF-IDF LR)**: **Macro-F1 = 0.5608**, Accuracy = **80.96%**, $\kappa_w = \mathbf{0.4932}$
* **P1-B (AraBERT Semantic LR)**: Macro-F1 = 0.4933, Accuracy = 72.84%, $\kappa_w = 0.4347$
* **P1-B_MLP (AraBERT Semantic MLP)**: Macro-F1 = 0.4454, Accuracy = 68.60%, $\kappa_w = 0.3295$
* **P2-A (Early Fusion MLP Tabular)**: **Macro-F1 = 0.4810**, Accuracy = 68.36%, $\kappa_w = 0.3602$ (F1 Below Saduq = **0.2682**)
* **P2-B_Concat (Multimodal GCN Concat)**: Macro-F1 = 0.4311, Accuracy = 64.96%, $\kappa_w = 0.2157$
* **P2-B_Attn (Multimodal GCN Attention)**: Macro-F1 = 0.4478, Accuracy = 66.27%, $\kappa_w = 0.2549$
* **P2-C (Topology Isolation MLP)**: Macro-F1 = 0.4531, Accuracy = 67.22%, $\kappa_w = 0.3318$
* **P2-D (Metadata Isolation MLP)**: Macro-F1 = 0.4315, Accuracy = 65.97%, $\kappa_w = 0.2919$
* **Core PO2 Statistical Proofs**:
  1. *GCN Message Passing vs. Non-GNN Early Fusion ($H_{1b}$)*: $\bar{\Delta} = -0.0499, t = -2.4595, p = 0.0215$ (Wilcoxon $p = 0.0296$). Message passing significantly degrades performance due to transmission network heterophily.
  2. *Attention GNN vs. Non-GNN Early Fusion*: $\bar{\Delta} = -0.0332, t = -2.1130, p = 0.0452$ (Wilcoxon $p = 0.0451$). Attention partially mitigates but remains inferior to tabular fusion.
  3. *Message Passing on Topology Alone (G2 vs. G1)*: $\bar{\Delta} = -0.0118, t = -0.7667, p = 0.4508$ (Wilcoxon $p = 0.4579$). Fail to reject null; message passing adds no predictive value.
  4. *Tabular Features over Text ($H_{1a}$)*: $\bar{\Delta} = +0.0357, t = 2.0044, p = 0.0564$ (Wilcoxon $p = 0.1267$). Marginal gain.

### 5.3 Project Objective 3: Downstream Isnād Chain Authentication ($N = 1,578$ Chains)
* Aggregation Model: Classical Weakest-Link Minimum Operator ($S_{\text{chain}} = \min_{i=1}^L y_i$).
* **Model P2-A (Early Fusion MLP)**:
  * Chain Exact Tier Accuracy: **62.10%** (vs. B1 Random 42.78%, B2 Probe 28.71%)
  * Chain Mean Absolute Error: **0.4278**
  * Error Propagation Ratio ($MAE_{\text{chain}} / MAE_{\text{node}}$): **1.2353** (strictly bounded error amplification)
  * Spearman Rank Correlation: **$\rho = 0.3918$** ($p = 4.29 \times 10^{-58}$)
  * Kendall Rank Correlation: **$\tau = 0.3730$** ($p < 10^{-15}$)
  * Chain Binary F1: **0.9378** (Binary Accuracy: 88.47%)
* **Baseline Contextualization**:
  * Majority Baseline B0 achieves 0.9618 binary F1 due to the 92.65% canonical base rate, but exhibits zero ordinal correlation ($\rho = 0.0000$).

---

## 6. Official Deliverable Artifact Directory

### 6.1 Documentation and Chapters
* Complete Chapter Drafts: `FYP2/report_draft/ch1_introduction.md` through `ch7_conclusion_and_references.md`
* Methodological Foundations:
  * `FYP2/research_design_table.md`
  * `FYP2/cohort_distribution.md`
  * `FYP2/attrition_table.md`
  * `FYP2/threats_to_validity.md`
  * `FYP2/limitations.md`
  * `FYP2/appendix_statistics.md`
  * `FYP2/appendix_reproducibility.md`
  * `FYP2/results_traceability_matrix.md`
  * `FYP2/thesis_audit_report.md`
  * `FYP2/examiner_attack_vectors.md`

### 6.2 Publication Figures (300 DPI Rendering)
* `FYP2/figures/01_pipeline.png` - End-to-end multi-modal research architecture
* `FYP2/figures/02_attrition.png` - Data attrition and cohort selection flow
* `FYP2/figures/03_po1_results.png` - PO1 lexical vs. semantic benchmark and significance
* `FYP2/figures/04_ablation_results.png` - PO2 13-arm ablation Macro-F1 and accuracy
* `FYP2/figures/05_significance_tests.png` - Forest plot of 6 formal paired hypothesis tests
* `FYP2/figures/06_graph_visualization.png` - 335-node transmission network and heterophily
* `FYP2/figures/07_chain_validation.png` - PO3 downstream chain metrics and rank correlation
* `FYP2/figures/08_confusion_matrix.png` - Multi-panel normalized confusion matrices
* `FYP2/figures/09_disagreement_cases.png` - Inter-scholar consensus and divergence analysis

---

## 7. Formal Verification Sign-Off

This empirical freeze guarantees that every numerical quantity, statistical claim, and methodological description across the entire dissertation is mutually consistent, leak-free, reproducible, and defended by empirical artifacts.
