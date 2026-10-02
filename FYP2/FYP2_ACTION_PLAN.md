# FYP2: Focused Research and Execution Plan

**Project Title:** A Multi-Modal Framework for Quantifying Narrator Reliability in ʿIlm ar-Rijal Using Semantic Sentiment Analysis and Graph Neural Networks  
**Student:** Alan Syahmi bin Sanih @ Sani (BI23110344) — Bachelor of Computer Science with Honours (Data Science)  
**Supervisor:** Dr. Khalifa Chekima  
**Faculty:** Faculty of Computing and Informatics, Universiti Malaysia Sabah (UMS)  
**Current Date / Academic Week:** 1 October 2026 — **Week 2**  

---

## 1. Official Institutional Timeline & Milestones

| Milestone / Timeline | Official Date | Academic Week | Target Deliverable / Focus |
|---|---|---|---|
| **Project II Commencement** | 21 Sept – 6 Nov 2026 | Weeks 1–7 | Product Development, Testing and Evaluation (Sprint 1) |
| **Project II Progress Report Submission** | **30 Oct 2026** | **Week 6** | Formal Progress Report (Ch. 1–3 + Preliminary Ch. 4 results) |
| **Project II Progress Presentation & Evaluation** | **4 & 6 Nov 2026** | **Week 7** | Viva / Progress Defense with Examiners (Dr. Lai & Dr. Tan) |
| **Project II Continuation** | 16 Nov 2026 – 1 Jan 2027 | Weeks 8–14 | Multimodal GNN, Chain Evaluation, Full Thesis (Sprint 2) |
| **Project II Final Report Submission** | **1 Jan 2027** | **Week 14** | Complete Thesis Submission (Chapters 1–6) |
| **Project II Final Progress Presentation** | **4–5 Jan 2027** | **Study Week** | Final Viva Presentation & System Demonstration |
| **Final Thesis Submission** | **5 Feb 2027** | Post-Exam | Final Hardcover / Bound Thesis with Viva Corrections |
| *Repeat Presentation (One Chance)* | *19 Feb 2027* | Contingency | *Contingency only* |

---

## 2. Two-Sprint Operational Execution Plan

### Sprint 1: Progress Defense Sprint (Weeks 2–7)
*Primary Goal:* Deliver an airtight Progress Report and Presentation demonstrating that FYP1 methodological flaws are resolved, entity resolution is upgraded, target leakage is eliminated, and preliminary baseline benchmarks are locked.

* **Week 2 (28 Sept – 4 Oct 2026) — CURRENT WEEK: Research Design & Leakage Audit**
  - [ ] **Lock Research Design Matrix:** Formalize the *RQ → Hypothesis → Experiment → Metric → Conclusion* structure to directly answer Dr. Lai's critique.
  - [ ] **Literature Comparison Table:** Complete benchmark comparison (*Mahmoud et al. / NarratorsKG*, *Narrator2Vec*, *Ahmad et al. 2025*, *Ghriss 2025*).
  - [ ] **Target Leakage Audit:** Inspect node features and data pipelines; verify Ibn Ḥajar's grades are excluded from model inputs.
  - [ ] **Corpus Version Reconciled:** Explicitly document SanadSet 650K (runtime) vs. AR-Sanad 280K (citation reference).
* **Week 3 (5 Oct – 11 Oct 2026) — Entity Resolution & Cohort Audit**
  - [ ] **Upgrade Entity Resolution:** Replace naive fuzzy matching with hierarchical disambiguation (Normalized Name → Kunya / Nisba → Tabaqa / Death Year → Teacher/Student Links).
  - [ ] **Quantify Matching Precision:** Conduct a stratified manual audit to replace the fragile "3 exact / 1,099 fuzzy" statistic with an audited precision score.
  - [ ] **Freeze Evaluation Cohort:** Finalize the clean narrator evaluation cohort and document class distribution.
* **Week 4 (12 Oct – 18 Oct 2026) — Baseline & Semantic Modeling (PO1)**
  - [ ] Implement and evaluate the lexical/keyword rule baseline.
  - [ ] Train/evaluate the semantic classifier (AraBERT v2) on classical *rijāl* statements.
  - [ ] Benchmark PO1: Compare Lexical vs. Semantic using **Macro-F1**, Accuracy, and Confusion Matrix.
* **Week 5 (19 Oct – 25 Oct 2026) — Graph Setup & Progress Report Assembly**
  - [ ] Construct the clean transmission graph in PyTorch Geometric (leakage-free).
  - [ ] Run preliminary Graph-only vs. Semantic-only ablations.
  - [ ] Draft Progress Report: Chapter 1 & 2 updates, Chapter 3 (Research Design & Entity Resolution), Chapter 4 (Preliminary PO1 results).
* **Week 6 (26 Oct – 1 Nov 2026) — Report Finalization & Submission**
  - [ ] Review report with supervisor Dr. Khalifa Chekima.
  - [ ] **DEADLINE: Friday, 30 October 2026 — Project II Progress Report Submission.**
* **Week 7 (2 Nov – 8 Nov 2026) — Progress Presentation & Defense**
  - [ ] Prepare slides addressing examiner feedback, audited matching precision, and PO1 results.
  - [ ] **MILESTONE: 4 & 6 November 2026 — Project II Progress Presentation & Evaluation.**

---

### Sprint 2: Full Multimodal GNN, Chain Validation & Thesis (Weeks 8–14)
*Primary Goal:* Complete the multimodal fusion ablation matrix, downstream isnād chain validation, full thesis writeup, and final viva preparation.

* **Weeks 8–9 (16 Nov – 29 Nov 2026) — Multimodal GNN & Ablations (PO2)**
  - [ ] Train multimodal model fusing AraBERT embeddings with GNN transmission topology.
  - [ ] Execute formal Ablation Study:
    - Model A: Graph-only
    - Model B: Semantic-only
    - Model C: Multimodal (Graph + Semantic)
  - [ ] Report Macro-F1 and Cohen's Kappa to answer: *Does the transmission graph add measurable value over text alone?*
* **Week 10 (30 Nov – 6 Dec 2026) — Downstream Isnād Evaluation (PO3)**
  - [ ] Test chain aggregation methods (Minimum, Mean, Product, Geometric Mean) on held-out Hadith chains.
  - [ ] Evaluate agreement against reference classifications (*Ṣaḥīḥ, Ḥasan, Ḍaʿīf*).
  - [ ] Document limitation: chain consistency validation, not independent theological authentication.
* **Weeks 11–12 (7 Dec – 20 Dec 2026) — Results & Discussion Drafting**
  - [ ] Generate core figures (Architecture, KG subgraph, Reliability distribution, Ablation chart, Confusion matrix, Chain case study).
  - [ ] Draft Chapter 5 (Results & Scholarly Discussion) and Chapter 6 (Conclusion & Limitations).
* **Week 13 (21 Dec – 27 Dec 2026) — Full Thesis Polish & Supervisor Review**
  - [ ] Assemble full thesis (Chapters 1–6).
  - [ ] Review draft with supervisor Dr. Khalifa Chekima and apply revisions.
* **Week 14 (28 Dec 2026 – 1 Jan 2027) — FINAL REPORT SUBMISSION**
  - [ ] Final proofreading, formatting, and reference verification.
  - [ ] **DEADLINE: Friday, 1 January 2027 — Project II Final Report Submission.**
* **Study Week (4–5 Jan 2027) — FINAL PROGRESS PRESENTATION / VIVA**
  - [ ] Final project defense and live system demonstration before the evaluation panel.
* **Post-Viva (5 Feb 2027) — FINAL BOUND THESIS SUBMISSION**
  - [ ] Implement viva examiner corrections and submit final hardcover thesis.

---

## 3. Research Design & Approved Objectives Mapping

**Core Research Question:** *How well can NLP-based analysis of classical rijāl assessments estimate narrator reliability, and does transmission-network information improve that estimate?*

| Objective | FYP2 Experiment | Primary Metric | Target Conclusion |
|---|---|---|---|
| **PO1 — NLP Reliability Index** | Compare lexical/rule baseline against AraBERT semantic classifier on an audited narrator cohort | **Macro-F1** (with confusion matrix & class distribution) | Demonstrates whether deep semantic NLP extracts reliable signals from classical *rijāl* text |
| **PO2 — Graph & Feasibility** | Controlled ablation: (1) Graph-only vs. (2) Semantic-only vs. (3) Multimodal Fusion | **Macro-F1** (with strict train/test split & leakage controls) | Demonstrates whether transmission network topology adds measurable predictive value |
| **PO3 — Downstream Evaluation** | Downstream chain aggregation (Min, Mean, Product, GeoMean) evaluated on held-out Hadiths | **Macro-F1 / Cohen’s Kappa** against reference grades | Assesses narrator scores via chain-level consistency with reference classifications |

---

## 4. Evaluation Framework (Locked)

* **Primary Metric:** **Macro-F1** (MANDATORY due to severe class imbalance across reliability categories).
* **Secondary Supporting Metrics:** Accuracy, Cohen’s Kappa, Confusion Matrix, Mean Absolute Error (MAE).
* **Evaluation Rule:** Fix the cohort, split, and metric before final runs. **Never tune thresholds on the held-out test set.**

---

## 5. Scope Guardrails & Viva Defense Rules

1. **Narrators as Primary Target:** Keep narrator reliability as the central prediction target; treat Hadith grading strictly as a downstream validation check.
2. **No Claim of Theological Grading:** Do not claim the system independently authenticates Hadith or replaces traditional scholarship. Frame the score as a computational decision-support index.
3. **No Target Leakage:** Verify that Ibn Ḥajar’s target grade is never encoded as an input feature in graph nodes.
4. **Controlled Literature Comparison:** Compare results against prior works (*NarratorsKG*, *Narrator2Vec*) with task and dataset differences explicitly noted.
5. **Hadith-Viz Decision Gate:** If access is granted, test a bounded, matched subset. If access is delayed or denied, proceed strictly with SanadSet 650K + *Taqrīb/Tahdhīb*.

---

## 6. Current Status Overview

| Component | Status | Notes |
|---|---|---|
| **Academic Timeline** | **Week 2 of 14** | 4 weeks until Progress Report (30 Oct 2026) |
| **Examiner Feedback** | Cataloged | Directly mapped to Priority 1–7 action items |
| **Main Transmission Corpus** | SanadSet 650K | Document exact version/filtering in methodology |
| **Primary Metric** | Locked to Macro-F1 | Addresses Dr. Tan's feedback |
| **Narrator Matching** | In progress (Week 3 target) | Replacing naive fuzzy matching with hierarchical disambiguation |
| **Hadith-Viz Access** | Pending response | Execution plan proceeds independently without blocker |
