# CHAPTER 4: EXPERIMENTAL DESIGN

This chapter presents the experimental design constructed to validate the three research objectives of this project: (PO1) developing an NLP-based qualitative-to-quantitative reliability scoring model, (PO2) constructing a multi-relational Knowledge Graph with Graph Autoencoder (GAE) structural representation learning, and (PO3) evaluating the resulting pipeline against canonical Hadith gradings. 

The experiments are organised chronologically across the three processing stages of the pipeline: (1) Data Ingestion and Preprocessing, (2) Knowledge Graph Construction and Node Alignment, and (3) Probabilistic Trust Scoring, Graph Neural Network (GNN) learning, and Chain-Level Soft Aggregation.

---

## 4.1 Research Methodology and Environment

All experimental tasks were conducted in a standardised local development environment to ensure strict reproducibility:
* **Hardware Configuration**: Local workstation equipped with an Intel Core i7-10750H CPU, 16 GB of DDR4 RAM, and an NVIDIA GeForce RTX 2070 Max-Q GPU (8 GB VRAM) for accelerating transformer inference and Graph Neural Network training.
* **Software Configuration**: The pipeline is implemented in Python (v3.13) utilising `PyTorch` (v2.3) and `PyTorch Geometric` (v2.5) for GNN operations, `NetworkX` (v3.2) for Graph construction, `CAMeL-Tools` (v1.2.0) and `transformers` (v4.40) for Arabic sentence classification, and `pandas` (v2.2) and `numpy` (v1.26) for vector operations.

---

*

## 4.2 Stage 1: Data Ingestion and Preprocessing Experiments

The first experimental stage focuses on clean block extraction and vector-space record alignment to build the foundational data matrix.

### 4.2.1 Experiment 1: Biographical Text Parsing
* **Objective**: To evaluate the regular expression (Regex) parser's ability to cleanly segment raw, Shamela-formatted classical Arabic biographical volumes into isolated narrator files.
* **Design**: The parsing script (`run_shamela_extract.py` and `run_shamela_target_extract.py`) scans Maktaba Shamela raw text files. It strips editorial headers, footers, and page numbers, and then isolates narrator text blocks (*tarājim*) based on structural pattern boundaries (such as numeral prefixes or `# $Tarjama` tags).
* **Parameters**:
  - Raw Input: 35 volumes of *Tahdhīb al-Kamāl* and 1 volume of *Taqrīb at-Tahdhīb*.
  - Regex Boundary Matchers: Coded matching patterns targeting entry markers (e.g. `^[\d٠-٩]{1,4}\s*[-–—]`).
* **Expected Output**: A structured JSON narrator encyclopedia containing 15,448 segmented biography files, with fields for name, full text, listed teachers, listed students, and listed historical grades. The target is an extraction confidence rate of $\ge 90\%$.

### 4.2.2 Experiment 2: Two-Tier Alignment Framework
* **Objective**: To evaluate the accuracy of the record-linking pipeline in cross-referencing Hadith reports between the unstructured Sanadset corpus and the annotated Hadith evaluation corpora.
* **Design**: The alignment script (`run_stage1.py`) runs the hybrid indexing model:
  - **Tier 1 (IR Layer)**: Character $n$-gram TF-IDF ($n \in [3, 5]$) similarity retrieval extracts the top 10 candidates.
  - **Tier 2 (Fuzzy Verification)**: Enforces a strict Levenshtein distance threshold ($t \ge 0.95$) using the `RapidFuzz` library to confirm matches.
* **Parameters**:
  - **Four Evaluation Dataset Parameters with Fixed Pre-Alignment Sampling Strategy**: The benchmarks set target goals of 1,200 and 6,000 successfully aligned Hadith chains across two balancing strategies (Book-Balanced and Grade-Balanced). To ensure target alignment goals are approached despite Tier 2 Levenshtein quarantine filtering ($t \ge 0.95$), fixed pre-alignment sampling targets of 1,800 and 7,000 query Hadiths are drawn:
    1. *1,200 Book-Balanced Parameter*: 1,800 query Hadiths sampled uniformly across the six canonical books (300 per book) before alignment.
    2. *1,200 Grade-Balanced Parameter*: Up to 1,800 query Hadiths sampled across canonical grade categories (up to 450 per grade, with the actual count constrained by available graded entries) before alignment.
    3. *6,000 Book-Balanced Parameter*: 7,000 query Hadiths sampled uniformly across the six canonical books (~1,166 per book) before alignment.
    4. *6,000 Grade-Balanced Parameter*: Up to 7,000 query Hadiths sampled across canonical grade categories, with the actual count constrained by the number of available graded entries across each grade category before alignment.
  - Seed: Global seed fixed to $42$ to preserve sample selection.
* **Expected Output**: Aligned datasets mapping each query Hadith to its counterpart index in the 650K Sanadset corpus, with failed matches ($t < 0.95$) automatically routed to a JSON quarantine log.

---

## 4.3 Stage 2: Entity Linking and Graph Construction Experiments

This stage evaluates the classification of narrator reliability, mapping of text name mentions to canonical network entities, and topological Knowledge Graph construction.

### 4.3.1 Experiment 3: Narrator Reliability Classification
* **Objective**: To test the efficacy of fine-tuning an AraBERT v2 transformer model as a 5-class ordinal classifier to categorize narrators into quality tiers (*matrūk*, *ḍaʿīf*, *maqbūl*, *ṣadūq*, *thiqah*), comparing its performance directly against a continuous regression baseline to evaluate mitigation of score compression.
* **Design**: The model fine-tuning script (`train_arabert_classifier.py`) processes 10,913 narrator profiles containing explicit classical grades. The training dataset is balanced across classes (5,580 training entry-grade pairs, with 1,636 held-out test pairs). The experimental design evaluates two distinct model formulations on identical test splits:
  1. **Continuous Regression Baseline**: Trained with a Mean Squared Error (MSE) loss objective predicting continuous reliability scalar values.
  2. **Proposed 5-Class Ordinal Classifier**: Trained with a Cross-Entropy loss objective with inverse-frequency class weighting across the five discrete quality tiers.
* **Parameters**:
  - Model Backbone: `aubmindlab/bert-base-arabertv2`.
  - Batch Size: 16; Learning Rate: $2 \times 10^{-5}$; Epochs: 4.
  - Classes: *matrūk* (0), *ḍaʿīf* (1), *maqbūl* (2), *ṣadūq* (3), *thiqah* (4).
* **Expected Output**: Evaluation metrics on the held-out test split documenting Precision, Recall, Macro F1-Score, and Quadratic Weighted Cohen's $\kappa$. The target is demonstrating that 5-class ordinal classification significantly mitigates score compression toward the mean compared to continuous regression (target: Macro F1 $\ge 0.80$, Cohen's $\kappa \ge 0.80$).

### 4.3.2 Experiment 4: Narrator Disambiguation and Name Matching
* **Objective**: To test the coverage and accuracy of resolving variations in transmitter name spellings against the canonical narrator encyclopedia index.
* **Design**: Mentions extracted from the transmission chains are normalised by stripping vocalisation marks (*tashkīl*), standardising alif variants, and standardising yā and hā spellings. Lookups are executed against the alias index (`alias_index.json`). If exact matching fails, fuzzy lookup is applied using the `RapidFuzz` token-sort ratio with a threshold of $0.80$.
* **Parameters**:
  - Input Names: Distinct name strings extracted from the aligned transmission chains.
  - Verification check: Cross-referencing immediate predecessors/successors (teachers and students) to resolve ties.
* **Expected Output**: A JSON name matching log (`narrator_matches.json`) categorising matches as exact, fuzzy, or quarantined. The target is an overall name resolution rate of $\ge 80\%$, with matched entries mapped to their canonical biographies.

### 4.3.3 Experiment 5: Knowledge Graph Construction and Topology
* **Objective**: To construct a unified, multi-relational Knowledge Graph of the classical Islamic transmission network and evaluate the impact of scale scaling on global topological properties.
* **Design**: The graph builder walks through transmission records, creating nodes for unique narrators and directed edges ($u \to v$) representing teacher-student propagation. A separate sweep integrates alternative alias names and family ties. Each node is annotated with temporal generations (*tabaqāt*, coded 1 to 12) and travel history metadata. The experiment evaluates graph construction across two distinct corpus scale parameters to measure structural density expansion.
* **Parameters**:
  - **Corpus Scale Parameters**:
    1. *Subset Graph Scale (100K Rows)*: Built upon a 100,000-row sample of the Sanadset corpus to observe topological properties at a smaller scale.
    2. *Full-Scale Graph Parameter (650K Rows)*: Built upon the complete 650,986-row Sanadset transmission corpus to evaluate structural scaling and label propagation density.
  - Relational parameters: Direct transmission edges ($u \to v$), family links, and alias nodes.
  - Temporal bounds: Coded 1–12 matching classical generations from 622 CE to 870 CE.
* **Expected Output**: A structured comparative analysis documenting node count expansion, edge density growth, and grade annotation coverage rates across the two scale parameters (target: scaling from 42,113 nodes / 147,795 edges on the 100K sample to 262,497 nodes / 987,036 edges on the 650K full corpus).

---

## 4.4 Stage 3: Scoring, GNN Representation, and Aggregation Experiments

The third stage validates the neural learning, semantic text classification, and parametrised soft aggregation components.

### 4.4.1 Experiment 6: Graph Autoencoder (GAE) Link Prediction
* **Objective**: To train a self-supervised Graph Autoencoder to learn dense, 64-dimensional structural node representations for transmission validation.
* **Design**: Rather than predicting narrator grades directly, the network training uses a GAE architecture:
  - **Encoder**: A 2-layer Graph Convolutional Network (GCN) that takes structural features (centrality, generation, region availability) and adjacency $A$ to output latent embeddings $Z \in \mathbb{R}^{N \times 64}$.
  - **Decoder**: An inner product computing link compatibility likelihood: $P_e = \sigma(z_u^T z_v)$.
  - **Training**: Semi-supervised reconstruction loss minimisation over 200 epochs using the Adam optimiser.
* **Parameters**:
  - Hidden dimension size: 128.
  - Latent vector size: 64.
  - Learning rate: $1 \times 10^{-3}$; Dropout: $0.3$.
  - Evaluation: Held-out 20% test split.
* **Expected Output**: Trained GCN node embeddings saved in `narrator_embeddings.npy`. The model must achieve a Link Prediction ROC-AUC of $\ge 90\%$ and Average Precision (AP) of $\ge 90\%$.

### 4.4.2 Experiment 7: Sentiment Quantisation and Score Mapping
* **Objective**: To evaluate the classification performance of the AraBERT-based sentiment pipeline in mapping unstructured biographical criticisms to continuous reliability and consensus scores.
* **Design**: Textual entries from the encyclopedia are segmented into individual sentences. The fine-tuned AraBERT model categorises each sentence into quality tiers, which are then mapped to continuous weights:
  - *Thiqah*: $0.85$
  - *Ṣadūq*: $0.70$
  - *Maqbūl*: $0.40$
  - *Ḍaʿīf*: $0.15$
  - *Matrūk*: $0.05$
  
  The narrator's mean reliability ($r_i$) and consensus agreement score ($d_{agree, i}$) are computed.
* **Parameters**:
  - Classifier: AraBERT sentence classifier.
  - Bounded Score Range: $[0.05, 0.85]$ (to reflect the historical uncertainty of human transmission).
  - Maximum consensus variance normalisation denominator: $0.16$.
* **Expected Output**: A generated score database (`narrator_scores_sentiment.csv`) mapping narrator node IDs to continuous $r_i$ and agreement scores.

### 4.4.3 Experiment 8: Chain-Level Aggregation and Ablation Study
* **Objective**: To optimise the soft aggregation parameter $\alpha$ balancing weakest-link bottlenecks against overall transmission quality in chain grading, and run a modular ablation study.
* **Design**: The evaluation script (`evaluate_final_with_geo.py`) computes chain scores using the parametrised aggregation formula:
  $$S_{base} = \alpha \times \min_{i} (W_i) + (1 - \alpha) \times \left( \frac{1}{M}\sum_{i=1}^M W_i \right)$$
  The parameter $\alpha$ is swept across the set $\{0.25, 0.50, 0.75\}$ across all four evaluation dataset parameters (1,200 book-balanced, 1,200 grade-balanced, 6,000 book-balanced, and 6,000 grade-balanced). In addition, the pipeline is evaluated across modular ablation layers (Semantic Only, Semantic + GNN, and Full Hybrid with Geo and Temp).
* **Parameters**:
  - Aggregation parameter values: $\alpha \in \{0.25, 0.50, 0.75\}$.
  - Output metrics: Accuracy, Macro F1-score, and Quadratic Cohen's $\kappa$.
* **Expected Output**: Validation performance matrices per $\alpha$ and per ablation layer, identifying empirical optimums.

### 4.4.4 Experiment 9: Multi-Book Performance Breakdown
* **Objective**: To analyse the optimal pipeline's performance across individual books in the *Kutub as-Sittah* to verify alignment with classical strictness.
* **Design**: Calibration results are grouped by canonical compiler. The script evaluates the grading accuracy for *Ṣaḥīḥ al-Bukhārī*, *Ṣaḥīḥ Muslim*, *Sunan an-Nasāʾī*, *Sunan Abī Dāwūd*, *Jāmiʿ at-Tirmidhī*, and *Sunan Ibn Mājah*.
* **Parameters**:
  - Evaluation metric: Macro F1, classification accuracy, and sample count per collection.
* **Expected Output**: A performance matrix demonstrating high classification accuracy on the strictly authenticated collections (target: $\ge 70\%$ accuracy on *Ṣaḥīḥ al-Bukhārī* and *Ṣaḥīḥ Muslim*), corresponding to the historical hierarchy of compilation standards.

---

## 4.5 Summary

This chapter has detailed the complete experimental design mapping to the three core research objectives of this project. Nine sequential experiments have been defined to validate raw biographical text segmentation (Experiment 1), record alignment (Experiment 2), narrator reliability classification (Experiment 3), name disambiguation (Experiment 4), graph construction (Experiment 5), Graph Autoencoder topological representation (Experiment 6), sentiment score mapping (Experiment 7), aggregation parameter optimisation and ablation (Experiment 8), and multi-book performance breakdown (Experiment 9). 

The following chapter documents the results obtained from executing these experiments and discusses their implications.
