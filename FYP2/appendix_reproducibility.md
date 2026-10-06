# Appendix B: Reproducibility and Computational Specification

This appendix provides the complete computational, architectural, and operational specification required to replicate the experimental workflows, data curation pipelines, and statistical outcomes reported in this study.

---

## 1. Hardware Environment Specification

All preprocessing routines, neural embedding extractions, cross-validation folds, and statistical tests were executed on a dedicated research workstation with the following hardware configuration:

* **Central Processing Unit (CPU)**: Intel Core i7-10750H @ 2.60 GHz (6 physical cores, 12 logical threads).
* **Random Access Memory (RAM)**: 16.0 GB DDR4 @ 2933 MHz.
* **Graphical Processing Unit (GPU)**: NVIDIA GeForce RTX 2070 with Max-Q Design.
  * Dedicated Video Memory (VRAM): 8,192 MB (8 GB) GDDR6.
  * Compute Capability: 7.5 (Turing Architecture).
* **Storage Subsystem**: 1 TB NVMe M.2 Solid State Drive (PCIe 3.0 x4).
* **Operating System**: Microsoft Windows 11 Home 64-bit (Build 22631).

---

## 2. Software Runtime and Library Dependencies

The computational pipeline was constructed using standard open-source Python data science and deep learning frameworks:

### Table B.1: Software Environment and Core Package Versions
| Component / Package | Version | Functional Role in Pipeline |
| :--- | :---: | :--- |
| **Python** | 3.13.0 | Core programming runtime |
| **CUDA Driver / Runtime** | 12.4 | Hardware acceleration and tensor parallelism |
| **PyTorch (`torch`)** | 2.5.1+cu124 | Neural network modeling and tensor operations |
| **Transformers (`transformers`)** | 4.49.0 | HuggingFace AraBERTv0.2 model architecture and tokenization |
| **Scikit-Learn (`scikit-learn`)**| 1.6.1 | Cross-validation splits, TF-IDF vectorization, Logistic Regression, metric scoring |
| **SciPy (`scipy`)** | 1.15.1 | Paired Student's t-tests, Wilcoxon signed-rank tests, rank correlations |
| **NumPy (`numpy`)** | 2.2.3 | Multi-dimensional array operations and feature serialization |
| **Pandas (`pandas`)** | 2.2.3 | Tabular dataset management and CSV manipulation |
| **NetworkX (`networkx`)** | 3.4.2 | Graph analysis, degree distribution, and topological processing |
| **Matplotlib (`matplotlib`)** | 3.10.0 | Publication-quality figure generation (300 DPI rendering) |
| **Seaborn (`seaborn`)** | 0.13.2 | Statistical distribution plotting and confusion matrix visualization |

---

## 3. Seed Management and Deterministic Control

To guarantee strict deterministic reproducibility across repeated runs:
* **Master Random Seed Suite**: The five master seeds governing cross-validation splits and model weight initialization are:
  $$\mathcal{S} = \{42, 123, 456, 789, 999\}$$
* **Deterministic Configuration**:
  ```python
  import os
  import random
  import numpy as np
  import torch

  def set_deterministic_seed(seed: int):
      random.seed(seed)
      os.environ["PYTHONHASHSEED"] = str(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      if torch.cuda.is_available():
          torch.cuda.manual_seed(seed)
          torch.cuda.manual_seed_all(seed)
          torch.backends.cudnn.deterministic = True
          torch.backends.cudnn.benchmark = False
  ```

---

## 4. Pipeline Execution Sequence

A prospective researcher seeking to re-execute the complete empirical workflow from scratch should execute the following scripts sequentially from the repository root directory:

```bash
# Step 1: Execute centralized data cleaning, leakage elimination, and cohort curation
python scripts/build_clean_dataset.py

# Step 2: Extract frozen 768-dimensional AraBERT embeddings for textual narratives
python scripts/extract_clean_arabert_embeddings.py

# Step 3: Execute PO1 classical lexical vs. semantic benchmark (N = 819, 25 paired folds)
python scripts/run_po1_benchmark.py

# Step 4: Execute PO2 13-arm controlled multimodal ablation benchmark (N = 335, 25 paired folds)
python scripts/run_clean_experiments.py

# Step 5: Execute PO3 downstream chain-level validation (N = 1,578 isnads)
python scripts/run_po3_chain_validation.py

# Step 6: Generate all 9 publication-quality figures at 300 DPI
python scripts/generate_all_figures.py
```

---

## 5. Script Artifact Mapping and Execution Parameters

### Table B.2: Script Inputs, Parameters, and Generated Artifacts
| Execution Stage | Primary Script | Core Inputs | Hyperparameters and Flags | Generated Artifact Outputs |
| :--- | :--- | :--- | :--- | :--- |
| **1. Cohort Curation** | `scripts/build_clean_dataset.py` | `data/raw/`, `output/stage1/` | Classical critics only; 11 post-classical authors purged | `output/cleaned_cohort/clean_narrators_cohort.csv`<br>`output/cleaned_cohort/clean_texts.json`<br>`output/cleaned_cohort/clean_graph.npz`<br>`output/cleaned_cohort/cohort_summary.json` |
| **2. Feature Extraction** | `scripts/extract_clean_arabert_embeddings.py` | `aubmindlab/bert-base-arabertv02`<br>`clean_texts.json` | Max token length = 512; Mean token pooling; Batch size = 16 | `output/cleaned_cohort/clean_arabert_embeddings.npy`<br>`output/cleaned_cohort/po1_strict_arabert_embeddings.npy` |
| **3. PO1 Benchmark** | `scripts/run_po1_benchmark.py` | `clean_texts.json`<br>`po1_strict_arabert_embeddings.npy` | 5x5 Stratified CV; 1,000 TF-IDF features; Inverse-frequency class weights | `output/cleaned_cohort/po1_results.json`<br>`FYP2/po1_benchmark_report.md` |
| **4. PO2 Benchmark** | `scripts/run_clean_experiments.py` | `clean_graph.npz`<br>`clean_arabert_embeddings.npy` | 13 arms; 2-layer GCN (hidden = 64); 2-layer MLP (hidden = 64); 200 epochs; Early stopping = 20 | `output/cleaned_cohort/clean_experiments_results.json`<br>`output/cleaned_cohort/clean_experiments_summary.json`<br>`FYP2/clean_experiments_report.md` |
| **5. PO3 Validation** | `scripts/run_po3_chain_validation.py` | `output/stage1/sanadset_cleaned.csv`<br>`clean_experiments_summary.json` | 1,578 chains; Weakest-link aggregation ($S = \min_i y_i$) | `output/cleaned_cohort/po3_chain_validation_results.json`<br>`FYP2/po3_chain_validation_report.md` |
| **6. Visualization** | `scripts/generate_all_figures.py` | Output JSON / NPZ files | DPI = 300; Font = DejaVu Sans | `FYP2/figures/01_pipeline.png` through `09_disagreement_cases.png` |

---

## 6. Centralized Logic Enforcement

To prevent silent label divergence or discordant filtering criteria, all experiment scripts import canonical definitions directly from `src/canonical_data_pipeline.py`:
* **Categorical Label Mapping**: `map_ibn_hajar_tier(grade_text)` applies negative-indicator precedence first (e.g., checking `"ليس بثقة"` prior to `"ثقة"`), guaranteeing zero label noise across all 13 experimental arms.
* **Classical Critic Filter**: `is_earlier_critic(scholar_name)` returns `False` for any scholar dying in or after 700 AH, enforcing an uncompromised temporal firewall against post-classical synthesis leakage.
