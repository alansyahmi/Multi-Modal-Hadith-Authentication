# FYP2 Recovery & Operational Execution Plan

**Project:** Evaluating Semantic Critic Appraisals and Isnād Network Information for Predicting Ibn Ḥajar's Narrator Reliability Assessments  
**Student:** Alan Syahmi bin Sanih @ Sani (BI23110344) — Data Science, FCI UMS  
**Supervisor:** Dr. Khalifa Chekima  
**Examiners:** Dr. Chin Kim On (Dr. Lai) & Dr. Tan Soo Fun  
**Date:** 5 October 2026 (Week 3 of 14)  
**Status:** Approved Operational Blueprint (Supersedes previous PO1/PO2 reports)

---

## 1. Executive Reset: The Clean Research Framing

The examiner audit demonstrated that attempting to force the claim that *"GNN message passing significantly improves reliability prediction"* is scientifically indefensible on the current isnād graph (label homophily = 0.505 vs. 0.470 chance; GNN ≈ non-GNN MLP). 

Instead of defending a fragile claim, the project is reset to a **rigorous empirical investigation**:

### 1.1 Central Research Question
> **To what extent can critic-text and isnād-network information predict Ibn Ḥajar's narrator reliability assessments, and does incorporating transmission-network information improve prediction beyond critic text alone?**

### 1.2 Formal Hypotheses
* **$H_{1a}$ (Feature-level gain):** Adding transmission-network and biographical node features to critic-text representations improves 3-tier Macro-F1 by $\ge 3.0$ percentage points across repeated grouped cross-validation folds.
* **$H_{1b}$ (Message-passing gain):** Message passing over the isnād graph (via GCN) improves Macro-F1 beyond an equivalent non-graph model (MLP) trained on identical node features, with a 95% confidence interval of the paired difference excluding zero.

### 1.3 The Three Defensible Conclusions
Whatever the experimental outcome, the thesis delivers a definitive, publishable scientific answer:
* **Case A ($GCN > MLP > Text$):** Both network features and transmission topology provide significant predictive uplift.
* **Case B ($MLP > Text$, but $GCN \approx MLP$):** Node-level transmission statistics (degree, generational flow) add predictive value, but relational message passing across isnād edges adds no marginal benefit due to low transmission homophily.
* **Case C ($Text \approx MLP \approx GCN$):** Narrator reliability is fundamentally determined by historical biographical appraisals; transmission network structure does not correlate with scholarly integrity.

All three conclusions are scientifically valid, high-distinction research outcomes.

---

## 2. Immediate Investigation Findings (Audit Checklist Verified)

| Item | Empirical Finding | Methodological Impact |
|---|---|---|
| **1. `grade_tier` Provenance** | `grade_tier` in Hadith-Viz is **100% derived from `ibn_hajar_grade`** (`ثقة` $\rightarrow$ `thiqat`, `صدوق` $\rightarrow$ `saduq`, `ضعيف` $\rightarrow$ `daif`). | **Must NOT be called "consensus".** The ground truth label is explicitly Ibn Ḥajar's synthesis grade (*Taqrīb al-Tahdhīb*). |
| **2. Cleaned Cohort Size** | **348 unique resolved Hadith-Viz identities** (454 graph nodes) meet all criteria: non-companion, non-single-token, valid Ibn Ḥajar grade, $\ge 1$ earlier critic statement. | **3-Tier Distribution:**<br>• Tier 1 (*Thiqah*): 252 (72.4%)<br>• Tier 2 (*Ṣadūq*): 66 (19.0%)<br>• Tier 3 (*Below Ṣadūq*): 30 (8.6%) |
| **3. Edge Direction** | In `graph_edges.csv`, `source_id` is the student (later narrator) and `target_id` is the teacher (earlier narrator). | Direction represents $Student \rightarrow Teacher$. `to_undirected()` must be applied during message passing so information flows bidirectionally. |
| **4. PO3 Coverage** | `sanadset.csv` contains zero Hadith grades. Stage 1 aligned only 1,030 hadiths against external datasets. | **Reframe PO3:** Measure chain-level error propagation (predicted vs. oracle narrator grades under weakest-link Min aggregation) rather than circular ḥadīth grading. |
| **5. Active NER Status** | No custom NER model is trained in FYP2. Narrator chains were pre-extracted in FYP1. | Remove claims of "NER model training" from FYP2 methodology. Treat parsed names as inputs to entity resolution. |

---

## 3. Phase 1 — Dataset & Identity Reconstruction

### Step 1.1: Canonical Identity Merging
* **Problem:** Multiple nodes represent the same historical narrator under different spelling variants (e.g., node 703 and node 751 are both `إبراهيم بن سعد`).
* **Action:** Build `output/cleaned_cohort/resolved_narrator_nodes.csv` mapping all graph surface nodes to their unique `hv_shamela_id`.
* **Graph Condensation:** Merge duplicate nodes into a single consolidated identity vertex, aggregating transmission edges and recomputing degree statistics.

### Step 1.2: Structural Elimination of Target Leakage
* **Problem:** Verbatim appraisal keywords (`ثقة`, `ضعيف`) in biographical text previously caused target leakage, requiring artificial regex masking.
* **Structural Fix:** 
  1. **Label:** Ibn Ḥajar's canonical *Taqrīb* grade (`ibn_hajar_grade`), binned into the 3-tier target.
  2. **Input Text:** Concatenate statements **exclusively from earlier classical critics** (Ibn Maʿīn, Aḥmad b. Ḥanbal, al-Bukhārī, Abū Ḥātim al-Rāzī, al-Nasāʾī, Ibn Ḥibbān, al-Dhahabī).
  3. **Strict Exclusion:** Completely strip Ibn Ḥajar's own evaluation block, `ibn_hajar_grade`, `grade_tier`, and `grades_by_source`.
  4. **Scientific Framing:** The NLP task is defined as *"Predicting Ibn Ḥajar's later synthesis verdict from the evaluative polarity and strength of earlier critics' appraisals."* No keyword masking is required; technical appraisal vocabulary is legitimate input.

### Step 1.3: Clean Feature Matrix Separation
Separate input features into three distinct, non-overlapping groups:
1. **Semantic Text:** Frozen AraBERT embeddings ($d=768$) extracted from earlier critic statements using `ArabertPreprocessor`.
2. **Topological Features ($d=5$):** Normalized appearances, in-degree, out-degree, net flow direction, average chain position.
3. **Biographical Metadata ($d=3$):** Generation index (`tabaqa_num`), binary death flag, binary ṭabaqa flag.

---

## 4. Phase 2 — The Controlled Experiment Matrix

Every experiment is executed using identical evaluation splits, identical loss functions (inverse-frequency weighted CrossEntropyLoss), and identical random seeds.

| Arm ID | Experimental Arm | Input Features | Architecture | Research Hypothesis Tested |
|---|---|---|---|---|
| **B0** | Majority Baseline | — | Constant class prediction | Establishes trivial accuracy floor (72.4%) |
| **B1** | Stratified Random | — | Empirical frequency sampling | Establishes trivial Macro-F1 chance floor (~0.33) |
| **B2** | Metadata / Name Probe | Name + Ṭabaqa template | Logistic Regression | Measures documentation density floor |
| **P1-A** | Lexical Baseline | Earlier critic text | TF-IDF (1–2 ngrams) + LR | PO1: Lexical appraisal baseline |
| **P1-B** | Contextual Semantic NLP | Earlier critic text | Frozen AraBERT + Linear Head | PO1: Deep semantic appraisal representation |
| **G1** | Non-GNN Graph Features | 5 Topology + 3 Metadata | 2-layer MLP (no edges) | PO2: Baseline predictive signal of node features |
| **G2** | Graph Message Passing | 5 Topology + 3 Metadata | 2-layer GCN (with edges) | PO2: Value of topological message passing alone |
| **P2-A** | Early Multimodal Fusion | Text (768) + All Features (8) | 2-layer MLP (no edges) | **Tests $H_{1a}$:** Does network data improve text? |
| **P2-B** | Full Multimodal GNN | Text (768) + All Features (8) | `ReliabilityFusionGNN` | **Tests $H_{1b}$:** Does GNN beat non-GNN fusion? |
| **P2-C** | Topology Isolation | Text (768) + Topology (5) | 2-layer MLP (no edges) | Disentangles topology gain from metadata gain |
| **P2-D** | Metadata Isolation | Text (768) + Metadata (3) | 2-layer MLP (no edges) | Isolates biographical metadata contribution |

---

## 5. Phase 3 — Robust Evaluation Protocol

### 5.1 Grouped Stratified Cross-Validation
* **Protocol:** 5-fold Stratified Cross-Validation, repeated across 5 distinct random seeds (**25 total paired evaluations**).
* **Identity Grouping (`GroupKFold`):** Grouped strictly by `hv_shamela_id`. **Zero narrators appear in both train and test partitions.**
* **Metric Reporting:** Report `Mean ± Standard Deviation` for all metrics. Single-seed point estimates are strictly forbidden.

### 5.2 Primary vs. Secondary Metrics
* **Primary Metric:** **Macro-F1 on 3 Canonical Tiers**:
  - Tier 1: *Thiqah* (Reliable / High Integrity)
  - Tier 2: *Ṣadūq* (Truthful / Intermediate)
  - Tier 3: *Below Ṣadūq* (Weak / Rejected / Unreliable)
* **Secondary Supporting Metrics:**
  - 5-Class Macro-F1 (*Matrūk*, *Ḍaʿīf*, *Maqbūl*, *Ṣadūq*, *Thiqah*)
  - Quadratic Cohen's $\kappa$ (measures ordinal grading penalty)
  - Per-class Precision, Recall, and F1-score
  - Aggregated Confusion Matrix across CV folds
  - Paired 95% Confidence Interval ($\Delta \text{Macro-F1}$) comparing P2-A vs. P1-B and P2-B vs. P2-A

---

## 6. Downstream Chain Validation (PO3 Reframing)

Instead of evaluating against external ḥadīth grades (which are sparse and circular), PO3 evaluates **chain-level error propagation**:
1. Take verified isnād chains from SanadSet containing narrators from the clean cohort.
2. Compute the ground-truth chain reliability index using **Oracle Narrator Scores** under weakest-link aggregation:
   $$S_{\text{oracle}} = \min_{i \in \text{chain}} y_i$$
3. Compute the predicted chain reliability index using **Model Narrator Scores**:
   $$S_{\text{predicted}} = \min_{i \in \text{chain}} \hat{y}_i$$
4. Evaluate Agreement:
   - Rank correlation (Spearman $\rho$, Kendall $\tau$) between $S_{\text{oracle}}$ and $S_{\text{predicted}}$.
   - Exact chain verdict agreement (Binary: Accepted vs. Rejected).
   - Document chain coverage rate explicitly.

---

## 7. Deliverables & Academic Timeline (Weeks 3–7)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SPRINT 1: WEEKS 3 TO 7 TIMELINE                       │
├─────────────┬───────────────────────────────────────────────────────────────┤
│ Week 3      │ • Build canonical cleaned cohort (348 identities).            │
│ (Current)   │ • Merge duplicate identity nodes & undirect graph edges.      │
│             │ • Separate feature arrays (text, topology, metadata).         │
├─────────────┼───────────────────────────────────────────────────────────────┤
│ Week 4      │ • Implement repeated 5-fold grouped CV evaluation engine.     │
│ (12–18 Oct) │ • Run B0–B2, P1-A, P1-B, G1, G2, P2-A–P2-D across 25 folds.   │
│             │ • Compute paired differences, 95% CIs, and hypothesis tests.  │
├─────────────┼───────────────────────────────────────────────────────────────┤
│ Week 5      │ • Run PO3 chain error propagation experiment.                 │
│ (19–25 Oct) │ • Draft Progress Report Chapters 1 to 5.                      │
│             │ • Insert audited tables and empirical figures into Chapter 6. │
├─────────────┼───────────────────────────────────────────────────────────────┤
│ Week 6      │ • Review draft with supervisor Dr. Khalifa Chekima.           │
│ (26–30 Oct) │ • Final polish (enforcing style guide: zero dashes).          │
│             │ • DEADLINE: Friday, 30 Oct 2026 — Progress Report Submission. │
├─────────────┼───────────────────────────────────────────────────────────────┤
│ Week 7      │ • Prepare defense slide deck incorporating examiner answers.  │
│ (4 & 6 Nov) │ • MILESTONE: Progress Presentation & Defense with Dr. Lai/Tan.│
└─────────────┴───────────────────────────────────────────────────────────────┘
```
