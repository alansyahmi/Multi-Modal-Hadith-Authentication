# FYP2: Focused Research and Execution Plan

**Project Title:** A Multi-Modal Framework for Quantifying Narrator Reliability in ʿIlm ar-Rijal Using Semantic Sentiment Analysis and Graph Neural Networks  
**Student:** Alan Syahmi bin Sanih @ Sani (BI23110344) (Bachelor of Computer Science with Honours in Data Science)  
**Supervisor:** Dr. Khalifa Chekima  
**Faculty:** Faculty of Computing and Informatics, Universiti Malaysia Sabah (UMS)  
**Current Date / Academic Week:** 6 October 2026 (Week 3)  

---

## 1. Official Institutional Timeline and Milestones

| Milestone / Timeline | Official Date | Academic Week | Target Deliverable / Focus | Status |
|---|---|---|---|---|
| **Project II Commencement** | 21 Sept to 6 Nov 2026 | Weeks 1 to 7 | Product Development, Testing and Evaluation (Sprint 1) | In Progress |
| **Project II Progress Report Submission** | **30 Oct 2026** | **Week 6** | Formal Progress Report (Ch. 1 to 7 Complete Draft) | Drafting |
| **Project II Progress Presentation and Evaluation** | **4 and 6 Nov 2026** | **Week 7** | Viva / Progress Defense with Examiners (Dr. Lai and Dr. Tan) | Scheduled |
| **Project II Continuation** | 16 Nov 2026 to 1 Jan 2027 | Weeks 8 to 14 | Thesis Finalization, Visualizations, Defense Preparation (Sprint 2) | Ready |
| **Project II Final Report Submission** | **1 Jan 2027** | **Week 14** | Complete Thesis Submission (Chapters 1 to 9) | Scheduled |
| **Project II Final Progress Presentation** | **4 to 5 Jan 2027** | **Study Week** | Final Viva Presentation and System Demonstration | Scheduled |
| **Final Thesis Submission** | **5 Feb 2027** | Post-Exam | Final Hardcover / Bound Thesis with Viva Corrections | Scheduled |
| *Repeat Presentation (One Chance)* | *19 Feb 2027* | Contingency | *Contingency only* | Contingency |

---

## 2. Operational Execution Plan and Milestone Progress

### Sprint 1: Progress Defense Sprint (Weeks 1 to 7)
*Primary Goal:* Deliver an airtight Progress Report and Presentation demonstrating that FYP1 methodological flaws are resolved, entity resolution is verified, target leakage is completely eliminated, and experimental benchmarks across PO1, PO2, and PO3 are validated.

* **Week 2 (28 Sept to 4 Oct 2026): Research Design and Leakage Audit**
  - [x] **Lock Research Design Matrix:** Formalize the *RQ -> Hypothesis -> Experiment -> Metric -> Conclusion* structure (documented in `FYP2/research_design_table.md` v2.1).
  - [x] **Literature Comparison Table:** Complete benchmark comparison (*Mahmoud et al. / NarratorsKG*, *Narrator2Vec*, *Ahmad et al. 2025*, *Ghriss 2025*).
  - [x] **Target Leakage Audit:** Discovered and eliminated target leakage where Ibn Hajar's canonical *Taqrib* grade was duplicated in biographical quote arrays (1,659 statements across 937 cards). Filtered exclusively to earlier classical critics (2nd to 4th century AH).
  - [x] **Corpus Discrepancy Reconciled:** Explicitly audited SanadSet 650K vs. AR-Sanad 280K vs. Hadith-Viz live portal (2,894 cards total: 1,918 unranked placeholders, 821 graded non-Sahabah cards).
* **Week 3 (5 Oct to 11 Oct 2026): Clean Cohorts and Empirical Execution (COMPLETED EARLY)**
  - [x] **Phase 1 (Clean Dataset and Condensed Graph):** Executed `scripts/build_clean_dataset.py`. Produced verified cohorts: Level 1 canonical pre-synthesis text benchmark ($N=819$, documenting the $N=844$ pre-Mizzi baseline), Level 2 multimodal graph ($N=335$ nodes, 177 unique undirected relationships representing 354 directed message edges in PyG, 8 features), and 5-stage data hygiene funnel (`FYP2/attrition_table.md`). Purged all quotes from al-Mizzi (d. 742 AH) and 10 other post-classical authorities.
  - [x] **Phase 2 (13-Arm 25-Fold Controlled Benchmark):** Executed `scripts/run_clean_experiments.py` and `scripts/run_po1_benchmark.py`. Extracted 768-dim AraBERT embeddings. Completed 25 paired cross-validation folds across all 13 model arms under transductive node classification with label-disjoint splits.
  - [x] **Phase 3 (PO3 Downstream Chain Error Propagation):** Executed `scripts/run_po3_chain_validation.py` across 1,578 verified transmission chains under weakest-link aggregation ($S_{chain} = \min_i y_i$).
  - [x] **Update Action Plan and Documentation:** Synchronized research design, cohort distributions, attrition funnel, and experimental logs.
* **Week 4 (12 Oct to 18 Oct 2026): Progress Report Drafting (Chapters 1 to 5)**
  - [ ] Draft Chapter 1 (Introduction, Problem Statements, Bounded Scope).
  - [ ] Draft Chapter 2 (Literature Review and Critical Comparative Analysis).
  - [ ] Draft Chapter 3 (Methodology: Attrition Funnel, Multi-Critic Filtering, Graph Construction).
  - [ ] Draft Chapter 4 (System and Experiment Design: 13-Arm Matrix, 25-Fold Protocol, Metrics).
  - [ ] Draft Chapter 5 (Implementation Details: Pipeline Architecture, Hyperparameters).
* **Week 5 (19 Oct to 25 Oct 2026): Progress Report Drafting (Chapters 6 to 7) and Qualitative Disagreement Case Study**
  - [ ] Draft Chapter 6 (Results: PO1 Benchmark, PO2 13-Arm Ablation, PO3 Downstream Chain Validation).
  - [ ] Draft Chapter 7 (Conclusion, Graph Heterophily Analysis, References).
  - [ ] Conduct Qualitative Disagreement Case Study ($N=58$) analyzing Ibn Hajar versus al-Dhahabi variances against computational predictions.
* **Week 6 (26 Oct to 1 Nov 2026): Report Finalization and Submission**
  - [ ] Complete peer check of thesis constraints (zero em/en-dashes, clinical tone, structured lists).
  - [ ] Review report draft with supervisor Dr. Khalifa Chekima.
  - [ ] **DEADLINE: Friday, 30 October 2026: Project II Progress Report Submission.**
* **Week 7 (2 Nov to 8 Nov 2026): Progress Presentation and Viva Defense**
  - [ ] Prepare slide deck addressing examiner critiques, leakage elimination, the graph heterophily finding, and weakest-link chain validation.
  - [ ] **MILESTONE: 4 and 6 November 2026: Project II Progress Presentation and Evaluation.**

---

### Sprint 2: Final Thesis and Viva Preparation (Weeks 8 to 14)
*Primary Goal:* Expand the Progress Report into the full 9-chapter thesis, refine figures, polish scholarly discussion, and conduct final viva rehearsals.

* **Weeks 8 to 10 (16 Nov to 6 Dec 2026): Full Thesis Expansion and Figure Polish**
  - [ ] Expand Chapter 7 into dedicated Discussion chapter (FCI 9-chapter thesis format).
  - [ ] Generate publication-quality vector diagrams (system architecture, graph topology, confusion matrices).
  - [ ] Expand cross-domain comparisons with modern graph heterophily literature.
* **Weeks 11 to 12 (7 Dec to 20 Dec 2026): Comprehensive Error Analysis and Sensitivity Testing**
  - [ ] Run ablation on edge-density thresholds in transmission networks.
  - [ ] Document boundary conditions of the weakest-link aggregation rule.
* **Week 13 (21 Dec to 27 Dec 2026): Full Thesis Review with Supervisor**
  - [ ] Final compilation of Chapters 1 to 9.
  - [ ] Final supervisor review by Dr. Khalifa Chekima.
* **Week 14 (28 Dec 2026 to 1 Jan 2027): FINAL THESIS SUBMISSION**
  - [ ] Final proofreading, formatting verification, and reference check.
  - [ ] **DEADLINE: Friday, 1 January 2027: Project II Final Report Submission.**
* **Study Week (4 to 5 Jan 2027): FINAL PROGRESS PRESENTATION / VIVA**
  - [ ] Final viva presentation and system demonstration.
* **Post-Viva (5 Feb 2027): FINAL BOUND THESIS SUBMISSION**
  - [ ] Implement viva corrections and submit official hardcover thesis.

---

## 3. Research Design and Approved Objectives Mapping

**Core Research Question:** *How accurately can NLP models of classical rijal statements estimate narrator reliability when target leakage is eliminated, and does transmission network topology provide measurable predictive value?*

| Objective | FYP2 Experiment | Primary Metric | Empirical Finding / Conclusion |
|---|---|---|---|
| **PO1: NLP Reliability Index** | Compare Lexical TF-IDF baseline against AraBERT semantic classifier on canonical audited cohorts ($N=819$ and $N=335$) | **Macro-F1**, Accuracy, Quadratic Weighted Kappa ($\kappa$) | Text strongly dominates the task. On the canonical $N=819$ cohort, TF-IDF achieves Macro-F1 of **0.6852 ± 0.0514** and Accuracy of **80.59%** ($\kappa = 0.6697$); AraBERT achieves Macro-F1 of **0.5635 ± 0.0373** and Accuracy of **69.99%** ($\kappa = 0.5008$). Classical jarh wa-ta'dil vocabulary (`ثقة`, `ضعيف`) provides a dense predictive signal that regularized lexical models capture with higher precision ($p = 3.49 \times 10^{-11}$). |
| **PO2: Graph and Feasibility** | Controlled 13-arm ablation across 25 paired folds on $N=335$ nodes: Graph-only (G1, G2), Semantic-only (P1-A, P1-B, P1-B_MLP), and Multimodal Fusion (P2-A, P2-B_Concat, P2-B_Attn, P2-C, P2-D) | **Macro-F1**, Paired t-test ($p$-value), Cohen's Kappa | GNN message passing provides no predictive uplift over node features alone and underperforms tabular fusion: G2 (GCN with edges, **0.3438**) does not beat G1 (MLP without edges, **0.3556**; $p=0.451$). Multimodal GNN Concat (**0.4311**) and Attention (**0.4478**) significantly underperform non-GNN Early Fusion MLP (**0.4810**; $p=0.022$ and $p=0.045$). The narrator transmission network exhibits structural heterophily. |
| **PO3: Downstream Evaluation** | Downstream chain aggregation ($S_{chain} = \min_i y_i$) evaluated across 1,578 verified transmission chains from SanadSet containing $\ge 2$ cohort narrators | **Exact Tier Accuracy**, Chain MAE, Spearman Rank Correlation ($\rho$), Binary Acceptance F1 | Narrator-level predictions propagate effectively into chain validity: Early Fusion MLP achieves **62.10%** exact chain tier accuracy, Chain MAE of **0.4278**, and Spearman $\rho = \mathbf{0.3918}$ ($p < 10^{-15}$). High binary chain F1 (0.81 to 0.94) confirms bounded ordinal error propagation, but must be evaluated alongside the 92.6% canonical collection acceptance baseline. |

---

## 4. Evaluation Framework (Locked)

* **Primary Metric:** **Macro-F1** across three ordinal reliability tiers:
  - Class 2: *Thiqah* (Trustworthy, Grade 1)
  - Class 1: *Saduq* (Truthful, Grade 2)
  - Class 0: *Below Saduq* (*Da'if* / *Matruk*, Grades 3 and 4)
* **Secondary Supporting Metrics:** Overall Accuracy, Quadratic Weighted Kappa ($\kappa$), Confusion Matrix, Spearman Rank Correlation ($\rho$).
* **Cross-Validation Protocol:** 5-fold Stratified Cross-Validation repeated across 5 random seeds (total of 25 paired iterations). Zero overlap between training and testing identities is enforced in every fold.
* **Rigorous Hypothesis Testing:** Paired two-tailed t-tests with 95% confidence intervals and Wilcoxon signed-rank tests across all 25 paired test folds.

---

## 5. Scope Guardrails and Viva Defense Rules

1. **Narrators as the Primary Target:** Narrator reliability is the central target of prediction; hadith chain grading serves strictly as a downstream evaluation of practical utility.
2. **Computational Decision Support (No Theological Claim):** The system does not claim to establish independent theological authenticity or replace classical hadith scholarship. It provides a transparent, computational decision-support tool.
3. **Zero Target Leakage:** Ibn Hajar's canonical *Taqrib* grades and al-Dhahabi's summary verdicts are completely stripped from input text representations. Quotes from al-Mizzi and all 10 post-classical synthesizers (7th to 9th centuries AH) are also purged. Only quotes from earlier classical critics (2nd to 4th century AH) are retained as textual features.
4. **Authentic Reporting of Negative Findings:** The lack of performance gain from GNN message passing is embraced as a key scientific finding: hadith transmission networks connect trustworthy masters to students of mixed competence, creating structural heterophily where neighborhood averaging degrades classification.

---

## 6. Official Faculty Report and Thesis Structure (FCI UMS)

```
+------------------------------------+    +------------------------------------+
|         FYP2 Progress Report       |    |            Final Thesis            |
|         (Deadline: 30 Oct 2026)    |    |       (Deadline: 1 Jan 2027)       |
+------------------------------------+    +------------------------------------+
| 1. Introduction                    |    | 1. Introduction                    |
| 2. Literature Review               |    | 2. Literature Review               |
| 3. Project Methodology             |    | 3. Project Methodology             |
| 4. System / Experiment Design      |    | 4. System / Experiment Design      |
| 5. Implementation                  |    | 5. Implementation                  |
| 6. Results (Experiments / Testing) |    | 6. Results (Experiments / Testing) |
| 7. Conclusion and References       |    | 7. Discussion                      |
|                                    |    | 8. Conclusion and Future Works     |
|                                    |    | 9. References                      |
+------------------------------------+    +------------------------------------+
```

### Mapping of FYP2 Deliverables to Progress Report Chapters

| Progress Report Chapter | Primary Content and Focus | Core Artifacts and Reference Files |
|---|---|---|
| **1. Introduction** | Background in hadith validation, computational subjectivity problem, three formal objectives (PO1, PO2, PO3), explicit scope boundaries. | `FYP2/research_design_table.md` |
| **2. Literature Review** | Historical rijal terminology, limitations of keyword sentiment lexicons, Arabic transformers, GNNs on citation and transmission graphs, comparative benchmarks. | `FYP2/literature_comparison_table.md` |
| **3. Project Methodology** | Five-stage data hygiene funnel, multi-critic text corpus construction, strict zero-leakage author filtering, clean undirected transmission graph formation. | `FYP2/attrition_table.md`, `FYP2/cohort_distribution.md` |
| **4. System / Experiment Design** | RQ to Hypothesis mapping ($H_1, H_{0a}, H_{0b}, H_3$), 13-arm experimental matrix, 25-fold paired cross-validation protocol, weakest-link aggregation rule. | `FYP2/research_design_table.md` |
| **5. Implementation** | Arabic text preprocessing, TF-IDF and AraBERT embedding extraction, MLP, GCN, and Multimodal Attention module implementations. | `scripts/run_clean_experiments.py`, `scripts/run_po1_benchmark.py` |
| **6. Results** | Empirical results for PO1 ($N=819$), PO2 13-arm 25-fold ablation ($N=335$), paired statistical hypothesis tests, PO3 downstream chain validation ($N=1,578$). | `FYP2/clean_experiments_report.md`, `FYP2/po1_benchmark_report.md`, `FYP2/po3_chain_validation_report.md` |
| **7. Conclusion and References** | Resolution of research questions, domain explanation of graph heterophily, computational limitations, APA references. | `FYP2/research_design_table.md` |

---

## 7. Current Status Overview

| Component | Status | Details and Quantified Verification |
|---|---|---|
| **Academic Timeline** | **Week 3 of 14** | All computational experiments completed; report drafting initiated |
| **Examiner Audit Critiques** | Completed | Target leakage eliminated; cohort boundaries strictly documented |
| **Text Benchmark Cohort (PO1)** | Completed ($N=819$) | Canonical pre-synthesis critics evaluated over 25 paired folds |
| **Multimodal Graph Cohort (PO2)** | Completed ($N=335$) | 335 nodes, 177 undirected edges, 8 features; 13-arm ablation completed |
| **Downstream Isnād Chains (PO3)** | Completed ($N=1,578$) | 1,578 verified transmission chains evaluated via weakest-link aggregation |
| **Statistical Hypothesis Testing** | Completed | Paired two-tailed t-tests with 95% confidence intervals locked |
| **Progress Report Drafting** | In Progress | Updating 7 chapters in `FYP2/report_draft/` adhering to style guidelines |
