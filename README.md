# FYP Project: Multi-Modal Hadith Authentication Framework

> **Topic:** Multi-Modal Framework for Quantifying Narrator Reliability in *ʿIlm ar-Rijāl* using Semantic Sentiment Analysis and Graph Neural Networks  
> **Student:** Alan Syahmi Bin Sanih @ Sani (BI23110344)  
> **Supervisor:** Dr. Khalifa Chekima  
> **Institution:** Faculty of Computing and Informatics, Universiti Malaysia Sabah (UMS)

---

## 📌 Project Overview

This repository contains the end-to-end implementation, evaluation pipelines, and research documentation for an automated, multi-modal Hadith authentication system. The framework bridges classical Islamic scholarship (*ʿIlm ar-Rijāl* and *Takhrij*) with modern NLP and Graph Machine Learning across three sequential stages:

1. **Stage 1 (Data Ingestion & Alignment):** Ingestion and cleaning of the 650K Sanadset dataset cross-referenced with Kutub as-Sitta ground truth using character n-gram TF-IDF retrieval and Levenshtein alignment.
2. **Stage 2 (Knowledge Graph Construction):** Construction of heterogeneous narrator transmission networks containing narrator nodes, student-teacher edges, family lineage relations, and historical geographic metadata.
3. **Stage 3 (GNN Scoring & Corroboration):** Narrator reliability quantification using Graph Neural Networks (GCN/GAT/GraphSAGE) combined with fine-tuned AraBERT sentiment scoring on biographical evaluations (*Jarḥ wa Taʿdīl*), multi-path corroboration (*Turūq*), and adversarial penalty validation.

---

## 🗂️ Reorganized Project Structure

```
FYP-Project/
│
├── 📜 run_stage1.py             # Pipeline Stage 1: Data ingestion & two-tier alignment
├── 📜 run_stage2.py             # Pipeline Stage 2: Narrator graph construction & export
├── 📜 run_stage3.py             # Pipeline Stage 3: GNN training, sentiment scoring & evaluation
├── 📜 requirements.txt          # Python dependencies
├── 📜 README.md                 # Project documentation
├── ⚙️ .gitignore                # Production ignore rules (excludes models, datasets, caches)
├── ⚙️ .gitattributes           # Git line endings configuration
│
├── 📁 src/                      # Core Production Package
│   ├── 📁 alignment/            # Two-tier TF-IDF & Levenshtein dataset aligner
│   ├── 📁 data/                 # Sanadset and HuggingFace Kutub as-Sitta loaders
│   ├── 📁 preprocessing/        # Arabic text normalization & sanad parsing
│   ├── 📁 shamela/              # Shamela scraper, ISO extractor & bio parsers
│   ├── 📁 stage2/               # Graph builder, feature extractors & network exporter
│   ├── 📁 stage3/               # GNN models, PyG trainers, AraBERT sentiment scorer
│   └── 📁 utils/                # Helper utilities and shared functions
│
├── 📁 scripts/                  # Research, Experiment & Evaluation Scripts
│   ├── 📁 extractors/           # Shamela biographical scraper & parsing utilities
│   ├── 📁 runners/              # Environment installers and batch training launchers
│   └── *.py                     # 150+ focused experiment, calibration & evaluation scripts
│
├── 📁 findings/                 # Experimental Research Reports & Milestone Logs
│   ├── 00_findings_index.md     # Index of research breakthroughs
│   ├── 01_... to 23_...md       # Documented milestones, metrics, and ablation studies
│   └── eda_results.json         # Exploratory data analysis outputs
│
├── 📁 Diagrams/                 # Architectural Flowcharts & System Diagrams
│   ├── multimodal_framework.drawio
│   ├── Three-Phase Hadith Authentication Flowchart.png
│   └── *.png
│
├── 📁 Slides/                   # Presentation Decks & Automated PPTX Generators
│   ├── create_pptx.py           # Automated PPTX presentation generator v1
│   ├── create_pptx_v2.py        # Automated PPTX presentation generator v2
│   ├── create_pptx_from_deck.py # HTML-to-PPTX generator
│   ├── FYP1_Presentation.pptx   # Final presentation deck
│   ├── FYP1_Presentation_v2.pptx
│   └── *.html                   # Interactive web presentation previews
│
├── 📁 Report Drafts/            # Academic Thesis Chapters & Deliverables
│   ├── 📁 notes/                # Formatting notes, transliteration guides & audits
│   ├── 📁 new_report_draft_md/  # Markdown source for thesis chapters
│   └── *.docx / *.pdf           # Full chapter drafts (v1 through v6)
│
├── 📁 Proposal Draft/           # Initial Proposal Chapters & Deliverables
├── 📁 Summary/                  # Project Summary Sheets & Official Endorsements
├── 📁 FYP2/                     # FYP2 Action Plan, Roadmap & Task Tracker
│
├── 📁 FYP Guides/               # University Briefing Guidelines & Requirements
├── 📁 Templates/                # Faculty Formatting Templates
├── 📁 References/               # Academic Papers & Literature Review PDFs
├── 📁 Meeting Logs & PDF Drafts/# Supervisor Meeting Minutes & Notes
│
├── 📁 notebooks/                # Jupyter Notebooks Directory (with .gitkeep)
├── 📁 tests/                    # Unit & Integration Tests
├── 📁 lib/                      # Frontend Visualization Assets (vis.js, tom-select)
│
├── 📁 Dataset/                  # [Local / Gitignored] Raw Sanadset & Kutub as-Sitta CSVs
├── 📁 output/                   # [Local / Gitignored] Models, checkpoints, graphs, embeddings
├── 📁 Tools/                    # [Local / Gitignored] Shamela ISO, setup EXEs & Chrome extension
└── 📁 scratch/                  # [Local / Gitignored] Ad-hoc debugging & scratch scripts
```

---

## 🚀 Quick Start & Usage

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Run Pipeline Stages
```bash
# Stage 1: Ingestion, Arabic Normalization & 1,200 Balanced Alignment Set
python run_stage1.py

# Stage 2: Heterogeneous Narrator Graph Construction
python run_stage2.py

# Stage 3: GNN Reliability Modeling & Multimodal Evaluation
python run_stage3.py
```

### 3. Generate Presentation Decks
```bash
python Slides/create_pptx_v2.py
```
Outputs the presentation deck directly to `Slides/FYP1_Presentation_v2.pptx`.
