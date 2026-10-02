# NLP & Hadith System Development

## Overview
Instructions for assisting with the technical, data processing, and machine learning components of the text-mining pipeline.

## 1. Tech Stack & Recommended Libraries
- **Arabic NLP Tools**: Utilize specialized tools designed for Arabic morphology: `CAMeL Tools`, `Farasa`, or `PyArabic` for tokenization and diacritic removal.
- **Transformer Models**: Favor Arabic-specific pretrained models such as **AraBERT**, **CamelBERT**, or **MARBERT** via HuggingFace `transformers` for NER and text classification tasks.
- **Knowledge Graphs**: Store and query narrator relationships using `NetworkX` (for localized, smaller-scale graphs) or `Neo4j` (if the dataset scales up).
- **Core Processing**: Use `pandas` for dataset manipulation and `spaCy` for rapid rule-based matching pipelines.

## 2. Pipeline Execution Phases
- **Phase A (Data Ingestion)**: Rigorous parsing of AR-Sanad/Sanadset. *Caution*: Ensure UTF-8 encoding is strictly enforced. Strip non-essential diacritics (Tashkeel) if it improves NER performance, unless vowel combinations are strictly required for disambiguating names.
- **Phase B (NER & Entity Linking)**: Extract chains verbatim, split by transition words (e.g., 'عن', 'حدثنا'). Map alias variations of narrators (e.g., "Muhammad ibn Ishaq") to canonical KG nodes.
- **Phase C (Probabilistic Trust Scoring)**: Aggregate sentiment or Jarh wa Ta'dil classification scores across the chain nodes. Implement Bayesian networks or heuristic scoring functions to calculate a final reliability confidence.

## 3. Evaluation & Benchmarking
Guide the user to set up formal evaluation scripts using `scikit-learn` or `seqeval`:
- **For NER/Extraction**: Use **Precision, Recall, and F1-Score**.
- **For Entity Linking**: Use **Accuracy** and **Mean Reciprocal Rank (MRR)**.
- **For Reliability Scoring**: If continuous, use **Root Mean Square Error (RMSE)** against baseline human-annotated grades. If categorical (Sahih, Hasan, Da'if), use **Classification Accuracy and Confusion Matrices**.

## 4. Coding Standards 
- Write clean, modular Python functions. Add standard docstrings (`""" """`) explaining input/output shapes.
- Handle Right-to-Left (RTL) text effectively in print statements and user interfaces (e.g., via `bidi.algorithm` and `arabic_reshaper` if building visualizations or terminal outputs).
