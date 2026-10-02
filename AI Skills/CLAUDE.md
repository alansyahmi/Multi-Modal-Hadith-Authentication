# CLAUDE.md — FYP Guide for Alan Syahmi

This file gives Claude full context about this Final Year Project so every response is grounded in the actual project goals, academic requirements, and examiner expectations. Read this before doing anything in this folder.

---

## 1. Who I Am & What This Project Is

**Student:** Alan Syahmi Bin Sanih @ Sani
**Matric:** BI23110344
**Program:** Bachelor of Computer Science with Honours (Data Science) — UMS
**Courses:** KD34603 Project 1 (Sem 6) → KD44303 Project 2 (Sem 7)
**Supervisor:** Dr. Khalifa Chekima
**FYP System:** obe.ums.edu.my/fyp

**Project Title:** A Multi-Modal Framework for Quantifying Narrator Reliability in ʿIlm ar-Rijal Using Semantic Sentiment Analysis and Graph Neural Networks
**Project Type:** Full Research Based (Data Science) — NLP + Knowledge Graphs + GNNs
**One-line pitch:** An end-to-end pipeline that automates Hadith narrator reliability scoring by combining Arabic NER (AraBERT v2), a Knowledge Graph (~1,500 narrator nodes), and semantic sentiment quantisation of Jarḥ wa Taʿdīl terms — producing an explainable Ṣaḥīḥ/Ḥasan/Ḍaʿīf/Mawḍūʿ grade per Hadith chain.

---

## 2. The Problem This Project Solves

- The authentication of Hadith through ʿIlm ar-Rijal (narrator criticism) is a cornerstone of Islamic scholarship, but manual verification cannot keep pace with digital dissemination of Hadith.
- Classical scholars tracked the life histories, teachers, students, and credibility assessments of over 500,000 narrators across centuries — a multi-generational effort with no automated equivalent.
- Existing computational approaches either (a) stop at NER extraction without assigning reliability scores, or (b) build graphs without integrating Jarḥ wa Taʿdīl semantic content.
- **The gap:** No published study has delivered a complete, end-to-end pipeline that unifies narrator extraction, identity disambiguation, and probabilistic reliability scoring into a single evaluatable system. The closest work (Oumer, 2025) builds a graph from Hadith text but explicitly leaves out narrator reliability scoring — the very question ʿIlm ar-Rijal asks.
- **The research element:** Translating qualitative scholarly judgments (Thiqa, Ṣadūq, Kaḏḏāb, etc.) into a continuous reliability index [0.0–1.0] via semantic sentiment analysis, then combining with GNN structural analysis to produce a chain-level grade. This multi-modal integration (semantic + structural) is the novelty.

---

## 3. Research Objectives (must map to methodology)

1. **PO1:** To develop an NLP classification model that maps qualitative scholarly judgments (Jarḥ wa Taʿdīl terms) to a standardised, quantitative reliability index (0.0–1.0).
2. **PO2:** To construct a Knowledge Graph representing teacher-student relationships and utilise GNNs supported with geospatial data to evaluate the structural integrity and geographical feasibility of transmission chains.
3. **PO3:** To evaluate the framework's accuracy by comparing automated reliability scores against established scholarly Hadith gradings (Ṣaḥīḥ, Ḥasan, Ḍaʿīf) from the Kutub as-Sitta.

> ⚠️ Every methodology decision must trace back to one of these three objectives. When writing Chapter 3, always show how each method achieves a specific PO.

---

## 4. Technology Stack

| Layer | Technology |
|---|---|
| NLP / NER Model | AraBERT v2 (fine-tuned on Sanadset 650K for BIO tagging) |
| Arabic NLP Utilities | CAMeL Tools, PyArabic |
| Knowledge Graph | NetworkX (primary), Neo4j (optional for scalability) |
| GNN Framework | PyTorch Geometric (two-layer GCN) |
| ML Framework | Python 3.11+, PyTorch |
| Semantic Sentiment | Fine-tuned AraBERT v2 regression head + baseline keyword lookup |
| Geospatial Verification | Geopy + gazetteer of classical Islamic cities |
| Fuzzy Matching | rapidfuzz (Levenshtein), scikit-learn (TF-IDF) |
| Training & Experimentation | Jupyter Notebook, Google Colab (GPU) |
| Datasets | OpenITI corpus, Shamela Digital Library, Sanadset 650K, AR-Sanad 280K, hadith_datasets (HuggingFace) |
| Evaluation | Accuracy, F1, RMSE, Cohen's Kappa against Kutub as-Sitta ground truth |
| Dev Tools | VS Code, Git/GitHub |

---

## 5. Pipeline Design (Three Stages)

### Stage 1: Data Ingestion & Preprocessing
- **Biographical Corpora:** Taqrīb al-Tahdhīb (Ibn Ḥajar al-ʿAsqalānī) and Tahdhīb al-Kamāl (al-Mizzī) from OpenITI and Shamela
- **Transmission Chain Corpora:** Sanadset 650K (650,986 Hadiths with isnād), hadith_datasets (HuggingFace) for ground-truth evaluation
- **Regex parser:** Extracts individual biographical entries (tarājim), discarding OpenITI YAML headers and editorial noise
- **Two-tier alignment:** TF-IDF sparse retrieval → Levenshtein fine-grained matching (threshold ≥ 0.95) for cross-referencing Hadith records
- **Reproducibility:** Deterministic seeding, SHA-256 hashing of text skeletons, quarantine routing for ambiguous matches

### Stage 2: NER + Knowledge Graph Construction
- **NER Model:** AraBERT v2 fine-tuned on Sanadset 650K with BIO tagging scheme. Entity types: ism, kunya, nasab, nisba, laqab. Target F1 ≥ 0.90.
- **Knowledge Graph:** ~1,500 canonical narrator nodes with 12-tier tabaqa system (Ṣaḥāba → Compilers), rich attributes (birth/death, travel history, Jarḥ wa Taʿdīl labels), edge types (heard_from, reported_from)
- **Entity disambiguation:** String similarity (Levenshtein on consonantal skeleton) + graph neighbourhood matching (shared teachers/students)

### Stage 3: Probabilistic Trust Scoring & Evaluation
- **Track A:** AraBERT v2 regression model — predicts reliability score from raw Jarḥ wa Taʿdīl text (contextual semantic understanding)
- **Track B:** Static keyword lookup baseline with weighted arithmetic mean and Bayesian posterior estimation
- **GCN:** Two-layer Graph Convolutional Network for structural reliability augmentation and edge-level anomaly detection (tadlīs, ikhtilāṭ, temporal/geographic impossibility)
- **Chain scoring:** Product of node weights × edge modifiers, mapped to Ṣaḥīḥ/Ḥasan/Ḍaʿīf/Mawḍūʿ thresholds
- **Evaluation:** Against 1,200 stratified Hadiths from Kutub as-Sitta (200 per book), metrics: Accuracy, F1, RMSE, Cohen's Kappa

### What is OUT of scope
- Matn (textual body) analysis or content-based Hadith grading
- Shīʿa isnād evaluation methodologies
- Real-time or web-based deployment of the scoring pipeline
- Languages other than Arabic (e.g., Urdu or Persian Hadith commentaries)
- Definitive Islamic rulings or fatwa — the framework is a decision-support tool for scholars

---

## 6. Dataset & Preprocessing Strategy

- **Primary Biographical:** Taqrīb al-Tahdhīb and Tahdhīb al-Kamāl — the most comprehensive and widely referenced collections of narrator assessments
- **Primary Transmission:** Sanadset 650K — 650,986 Hadiths with isnād chains for NER training
- **Evaluation Ground Truth:** hadith_datasets (HuggingFace) — contains the 'Grade' column (e.g., "Sahih (Darussalam)") from Kutub as-Sitta
- **Preprocessing steps:**
  - Regex parsing to isolate individual biographical entries (tarājim) from OpenITI-formatted text
  - BIO tagging of narrator name components (ism, kunya, nasab, nisba, laqab)
  - Two-tier alignment: TF-IDF candidate retrieval (character n-grams) → Levenshtein fuzzy matching (threshold ≥ 0.95)
  - Data augmentation: not heavily used — the focus is on handling orthographic variants of Arabic names
- **Known challenge:** Arabic naming conventions produce dozens of orthographic variants for a single narrator. The consonantal script and lack of diacritics in some manuscripts create ambiguity (e.g., the same rasm can represent multiple names). Resolving these variants into canonical identities is the core disambiguation challenge.

---

## 7. Evaluation Metrics

| Metric | What it measures |
|---|---|
| Accuracy | Overall correctness of predicted Hadith grades vs. ground truth |
| Precision | Of all Hadith predicted as a given grade, how many were correct |
| Recall | Of all actual Hadith of a given grade, how many were detected |
| F1-score | Harmonic mean of precision and recall, per grade and macro-averaged |
| RMSE | Root Mean Square Error between continuous reliability scores and ground-truth ordinal grades |
| Cohen's Kappa | Agreement between predicted and actual grades, corrected for chance |

Evaluation is per-grade (Ṣaḥīḥ, Ḥasan, Ḍaʿīf, Mawḍūʿ) and aggregate. Baselines include a naive majority-class classifier and a simple lexical lookup model (term-to-grade mapping without graph features).

---

## 8. FYP Academic Timeline & Milestones

| Milestone | When | Notes |
|---|---|---|
| One-Page Project Summary | FYP1 Week 1 | Submit to review committee |
| Find supervisor + write proposal | FYP1 Week 1–6 | Email politely, state interests & proposed titles |
| Proposal Defense (5% of 30%) | FYP1 Week 7 | See defence checklist below |
| Progress Report (25% of 30%) | FYP1 Week 14 | Show preliminary work — Stages 1 & 2 |
| Progress Presentation | FYP2 Week 7 | |
| Final Presentation | FYP2 Study Week | |
| Final Thesis + Article Summary + Source Code | FYP2 End of Semester | |

**Grading split:** Supervisor 40% | Examiner 30% | (rest from reports/presentations)

**Late submission:** -5% per day after 5 PM deadline until passing mark (35%)
**Plagiarism:** Turnitin <30% (excluding references); source code/public code usage <30%
**Supervisor meetings:** Minimum 5 per semester — log in FYP system within 1 week of each meeting; include printed/screenshot logs in appendices

---

## 9. Proposal/Presentation Defence Checklist (Week 7)

The examiner checks these in order:
1. **Introduction** — is the context clearly established? (Ḥadīth authentication, ʿIlm ar-Rijal, digital challenge)
2. **Problem Statement / Gap** — what is broken with existing solutions? (Existing work stops at extraction/graph construction; no one has built an end-to-end trust scoring pipeline)
3. **Objectives** — specific, measurable, tied to a real gap (PO1: reliability index, PO2: KG + GNN, PO3: evaluation against ground truth)
4. **Literature Review** — does it fill the identified gap? Are existing techniques critiqued? (NER, KG, GNN, semantic sentiment — each reviewed with limitations identified)
5. **Methodology** — describes the method used to solve the problem; each objective must map to a method (3-stage pipeline: Data → NER+KG → Scoring)
6. **Presentation quality**

---

## 10. Examiner's Key Evaluation Criteria (READ THIS CAREFULLY)

These are the actual questions the examiner will probe during defence:

### The "Why" Framework — justify everything
- **Why this problem?** Manual ʿIlm ar-Rijal verification cannot scale to the volume of digitally circulating Hadith. Existing computational work fragments the pipeline — no end-to-end trust scoring exists.
- **Why AraBERT v2?** What makes it better than rule-based NER or older transformer models for classical Arabic narrator extraction?
- **Why Sanadset 650K + AR-Sanad 280K?** Why not another dataset? What gap does combining them fill?
- **Why GNN + Knowledge Graph?** What does graph-structural analysis add that semantic scoring alone cannot capture? (Temporal impossibility detection, neighbourhood-based reliability propagation)
- **Research ≠ Integration.** Simply applying AraBERT to Arabic text is not research. The research element is: (1) the multi-modal integration of semantic sentiment + GNN structural analysis, (2) the systematic comparison of transformer-based vs. keyword-lookup reliability scoring, (3) the edge-level anomaly detection with risk modifiers.

### What they specifically want to see
- **Arabic NLP competency** — can you actually fine-tune AraBERT, handle Arabic morphological complexity, and evaluate on classical texts? Not just run a pre-trained demo.
- **Graph/Network competency** — can you construct a meaningful Knowledge Graph, implement a GCN, and demonstrate that graph features improve scoring accuracy?
- **Novelty** — how is this different from existing work? The end-to-end scoring pipeline (NER → KG → GNN → chain-level grade) is the differentiator. Articulate this clearly.
- **Research element** — e.g., Track A vs. Track B ablation (does transformer-based scoring beat keyword lookup?), GCN contribution (does structural analysis improve agreement with ground truth?), edge modifier impact on chain scores.

### At Week 14 they will check
- Have you responded to Week 7 panel comments? Show changes made.
- Have you started coding? Demonstrate preliminary work: data ingestion pipeline, Regex parser output, initial NER training results (F1 scores), Knowledge Graph statistics (nodes, edges, tabaqa distribution).
- Understanding depth: explain why specific preprocessing choices were made (Regex patterns for OpenITI, TF-IDF character n-gram range, Levenshtein threshold rationale), how AraBERT was configured for BIO tagging, how the KG schema maps to classical ʿIlm ar-Rijal concepts.

---

## 11. How Claude Should Help in This Project

### Writing academic chapters
- Always use formal academic English. Avoid passive abuse; prefer clear, precise prose.
- Every claim in Lit Review must be supported by a citation (use `[n]` format matching `papers_reference.md`).
- Objectives must always map to methodology — if a method is described in Chapter 3, it must link to a PO.
- Turnitin limit is 30% — paraphrase and synthesise; do not copy from papers.
- Do not add sections not in the proposal template unless asked.
- Use the **ghost.skill** when humanising AI-generated text for the report to reduce AI detection risk.

### Writing code
- All code in Python 3.11+. Use PyTorch and HuggingFace transformers for model training and inference.
- NER: Fine-tune AraBERT v2 using the `transformers` library with BIO tagging scheme.
- KG: Use NetworkX for graph construction; PyTorch Geometric for GCN implementation.
- Arabic text processing: Use CAMeL Tools for morphological analysis, sentence boundary detection.
- Always write modular, well-commented code — this is a portfolio piece and will be reviewed by examiners.
- Prioritise reproducibility: set random seeds, log hyperparameters, save model weights after each training run, use deterministic data splitting.

### Research decisions
- When proposing a method or architecture change, always justify it with: (1) what problem it solves in the context of Ḥadīth authentication, (2) what the literature says, (3) how it maps to a PO.
- If asked "should I use X or Y", give a recommendation with reasoning tied to this project's specific context (classical Arabic NLP, narrator disambiguation, chain-level reliability scoring).

### Things to avoid
- Do not suggest scope creep (e.g., Matn analysis, Shīʿa isnād, real-time web app, mobile deployment) unless asked — this is a one-year FYP with a solo developer.
- Do not write filler or padding in academic text. Every sentence should carry meaning.
- Do not hallucinate citations. Only reference papers listed in the project's reference list or papers the student provides.
- Do not reference WasteSortNet, OptiPlate, YOLOv8, waste detection, or any previous abandoned project ideas — those are not this project.

---

## 12. Folder Structure

```
FYP Project/
├── AI Skills/
│   ├── CLAUDE.md                              ← this file
│   └── ghost.skill                            ← humanisation skill for report writing
├── Report Drafts/
│   ├── FYP Project 1 Revised Edition_UPDATED.docx  ← current working report (Ch 1-3)
│   └── FYP Project 1 Revised Edition.docx
├── Proposal Draft/
│   └── FYP Project 1 Proposal.docx
├── FYP Guides/
├── Templates/
├── Summary/
└── WasteSortNet/                              ← abandoned previous idea, ignore entirely
```

**Active project:** Hadith Narrator Reliability Framework — all new work relates to ʿIlm ar-Rijal, Arabic NER, KG, GNN, and semantic sentiment quantisation.
**Current report:** `Report Drafts/FYP Project 1 Revised Edition_UPDATED.docx` — Chapters 1–3 complete.
**Citation source of truth:** References section of the current report — always check existing citations before adding new ones.
**Ignore:** `WasteSortNet/` folder — this was an earlier abandoned project idea and is no longer relevant.

---

## 13. Tools & Extensions Claude Should Use for This FYP

Claude should proactively use the following tools and capabilities when helping with FYP tasks. These should be invoked without asking when relevant.

### Document & Report Creation
- **ghost.skill** — Use whenever writing or humanising text for the academic report. Reduces AI detection risk. Invoke before generating any `.docx` content.
- **`pptx` skill** — Use when creating or editing PowerPoint presentations. Essential for Week 7 Proposal Defence, Week 14 Progress Presentation, and FYP2 Final Presentation slides.
- **`xlsx` skill** — Use when producing spreadsheets. Useful for tracking evaluation results (F1, accuracy, RMSE, Cohen's Kappa per experiment), hyperparameter logs, and dataset statistics.

### Data Visualisation & Analysis
When writing Python code for this project, use the following libraries for all visualisation and analysis tasks:

| Library | Purpose |
|---|---|
| `matplotlib` | Training curves, bar charts, KG visualisations |
| `seaborn` | Heatmaps, distribution plots, styled charts |
| `plotly` | Interactive KG visualisation, dashboard |
| `scikit-learn` | TF-IDF vectorisation, metrics computation, confusion matrix |
| `pandas` | Tabular results (per-class metrics, experiment logs) |
| `numpy` | Numerical computation |
| `transformers` | AraBERT v2 fine-tuning and inference |
| `torch` / `torch_geometric` | PyTorch deep learning, GCN implementation |
| `networkx` | Knowledge Graph construction and querying |
| `camel-tools` | Arabic morphological analysis, sentence boundary detection |
| `rapidfuzz` | Levenshtein distance for fuzzy matching |
| `geopy` | Geocoding narrator locations |
| `tqdm` | Progress bars |

> ⚠️ When running any `pip install` command, always append `--break-system-packages` flag.

### Figures Claude Must Generate for the Thesis
The following figures are expected in the final thesis. Produce them when the relevant data is available:

1. **KG degree distribution histogram** — number of teachers/students per narrator
2. **Tabaqa distribution bar chart** — narrator count per generation (Ṣaḥāba through Compilers)
3. **NER training curves** — loss and F1 over epochs for AraBERT v2 fine-tuning
4. **Per-entity-type F1 bar chart** — ism, kunya, nasab, nisba, laqab
5. **Confusion matrix heatmap** — predicted vs. actual Hadith grades
6. **Term weight distribution** — box plot of predicted reliability scores per Jarḥ wa Taʿdīl term
7. **Chain score histogram** — distribution of chain-level reliability scores coloured by ground-truth grade
8. **KG subgraph visualisation** — sample teacher-student network with node colours by tabaqa
9. **Ablation study bar chart** — comparing full pipeline vs. baseline (no GCN, keyword-only) on Cohen's Kappa

### How Claude Should Use These Tools
- When producing any figure, save it to a `figures/` folder in the project root.
- When producing any document (report, chapter, slides), save it to the appropriate subfolder.
- Always install missing Python packages before running scripts — do not ask permission, just install.
- When producing thesis-ready figures, always: set `dpi=300`, use legible font sizes (`fontsize ≥ 11`), include axis labels, title, legend.

---

## 14. Quick Reference — Key Definitions

| Term | Definition in this project's context |
|---|---|
| Ḥadīth | A report of the sayings, actions, or approvals of the Prophet Muhammad ﷺ |
| Sanad / Isnād | The chain of transmission — the sequence of narrators who passed down a Ḥadīth |
| Matn | The textual body/content of a Ḥadīth (out of scope for this project) |
| ʿIlm ar-Rijal | "The science of men" — the study of narrator biographies and credibility |
| Jarḥ wa Taʿdīl | "Criticism and commendation" — the discipline of evaluating narrator reliability |
| Thiqa | "Reliable" — the highest standard narrator grade |
| Ṣaḥīḥ | "Sound" — the highest Hadith grade; all narrators are thiqa, chain is continuous |
| Ḥasan | "Good" — a sound Hadith with minor shortcomings in narrator memory |
| Ḍaʿīf | "Weak" — a Hadith failing one or more authenticity conditions |
| Mawḍūʿ | "Fabricated" — a Hadith whose narrator is known to have lied |
| Tabaqa (pl. Tabaqāt) | Generation/stratum — narrators grouped by when they lived relative to the Prophet ﷺ |
| Kunya | Teknonym — an honorific name (e.g., Abū Hurayra = "father of the kitten") |
| Nisba | Geographic or tribal affiliation (e.g., al-Bukhārī = "from Bukhara") |
| Tadlīs | "Concealment" — when a narrator uses ambiguous phrasing to hide a weak link in the chain |
| Ikhtilāṭ | "Confusion" — when a narrator's memory deteriorated later in life |
| OpenITI | A corpus of digitised classical Islamic texts (source for biographical entries) |
| AraBERT v2 | A pre-trained Arabic BERT model — the backbone for NER and semantic sentiment |
| GCN | Graph Convolutional Network — used for structural reliability analysis of isnād chains |
| Cohen's Kappa | A statistic measuring inter-rater agreement, corrected for chance — key evaluation metric |
