# Chapter 5: Implementation

## 5.1 Computing Environment and Hardware Configuration

The computational framework was developed, trained, and evaluated on a dedicated machine learning workstation. Table 5.1 details the specific hardware and software components utilized throughout the experimental pipeline.

### Table 5.1: Computing Environment and Core Software Specifications

| Component Category | Hardware / Software Element | Detailed Specification and Version |
|---|---|---|
| **Operating System** | Platform Architecture | Microsoft Windows 11 Enterprise (64-bit) |
| **Graphics Processing Unit** | GPU Hardware | NVIDIA GeForce RTX 2070 with Max-Q Design (8 GB GDDR6 VRAM) |
| **CUDA Acceleration** | Driver and Runtime | NVIDIA CUDA Toolkit 12.4 with cuDNN acceleration |
| **Programming Language** | Runtime Environment | Python 3.13.0 |
| **Deep Learning Framework** | Core Framework | PyTorch 2.5.1 with CUDA compute capability support |
| **Graph Neural Network Library**| Graph Extension | PyTorch Geometric (PyG) 2.6.1 |
| **Natural Language Processing** | Transformer Models | Hugging Face Transformers 4.46.2 |
| **Classical Machine Learning** | Baseline and Metrics | Scikit-Learn 1.5.2, SciPy 1.14.1 |
| **Network Graph Analysis** | Graph Modeling | NetworkX 3.4.2 |
| **Data Manipulation** | Table Processing | Pandas 2.2.3, NumPy 2.1.3 |

---

## 5.2 Text Preprocessing and Semantic Feature Extraction Pipeline

The textual pipeline transforms classical Arabic biographical records into fixed-dimensional numeric feature representations while eliminating target leakage.

### 5.2.1 Arabic Orthographic Normalization
Classical Arabic texts exhibit diverse orthographic variations across digital transcriptions. Prior to feature extraction, text strings are preprocessed through a standardized cleaning routine:
* Elimination of non-lexical diacritics (*tashkīl*), including fatḥah, ḍammah, kasrah, sukūn, and tanwīn.
* Removal of typographical elongation glyphs (*kashīdah* or *taṭwīl*).
* Orthographic unification of hamza variants: converting `إ`, `أ`, and `آ` into bare alif `ا`.
* Standardization of word-ending characters: unifying alif maqṣūrah `ى` to yāʾ `ي`, and tāʾ marbūṭah `ة` to hāʾ `ه`.

### 5.2.2 Lexical Feature Extraction (TF-IDF Pipeline)
For experimental arm P1-A, normalized biographical statements are processed through a scikit-learn `TfidfVectorizer`:
* Tokenization spans both word-level n-grams (ranges 1 to 3 words) and character-level n-grams (ranges 3 to 5 characters) to capture morphological root patterns.
* Sublinear term frequency scaling is enabled: $\text{tf} = 1 + \log(\text{tf})$.
* Minimum document frequency is set to $\text{min\_df} = 2$ to suppress typographical anomalies.
* The vocabulary is capped at the top 5,000 most informative features ranked by corpus term frequency.

### 5.2.3 Contextual Semantic Feature Extraction (AraBERT Backbone)
Contextual embeddings are extracted using the dedicated script `scripts/extract_clean_arabert_embeddings.py`. The extraction workflow executes four sequential steps:
1. **Model Initialization:** Loads the pretrained `aubmindlab/bert-base-arabertv2` transformer model and its associated Farasa-based tokenizer. Model parameters are frozen to serve as a pure static feature extractor.
2. **Tokenization and Masking:** Multi-critic statements are tokenized into subword tokens with padding and truncation enforced at a maximum sequence length of 512 tokens:
   $$\text{Input IDs} \in \mathbb{Z}^{B \times 512}, \quad \text{Attention Mask} \in \{0, 1\}^{B \times 512}$$
3. **Forward Pass and Mean Pooling:** Sequences are processed through the 12 transformer encoder layers on the GPU. To obtain a sequence representation invariant to sequence length, mean pooling is computed across all non-padded token vectors:
   $$e_i = \frac{\sum_{t=1}^T m_{it} h_{it}}{\sum_{t=1}^T m_{it}} \in \mathbb{R}^{768}$$
   where $h_{it} \in \mathbb{R}^{768}$ denotes the output hidden vector of token $t$ in sequence $i$, and $m_{it} \in \{0, 1\}$ represents the corresponding attention mask value.
4. **Deterministic Storage:** Extracted vectors are serialized into `output/cleaned_cohort/clean_arabert_embeddings.npy` ($335 \times 768$) and `output/cleaned_cohort/po1_845_arabert_embeddings.npy` ($844 \times 768$).

---

## 5.3 Graph Feature Construction and Network Formulation

The graph processing module (`scripts/build_clean_dataset.py`) parses transmission interactions from SanadSet and compiles the tabular node feature matrix.

### 5.3.1 Adjacency Matrix and Graph Serialization
The transmission network is formulated as an unweighted, undirected NetworkX graph $G = (V, E)$. Node identifiers are mapped to contiguous integers from 0 to $N-1$. The symmetric adjacency matrix $A \in \{0, 1\}^{N \times N}$ is saved in compressed sparse format alongside edge lists in `output/cleaned_cohort/clean_graph.npz`.

### 5.3.2 Extraction of Topological Metrics
For each node $v_i \in V$, five structural metrics are computed:
* **Degree Centrality:** $k_i = \sum_{j} A_{ij}$.
* **Directed In-Degree:** Number of distinct teachers listed in verified transmission chains.
* **Directed Out-Degree:** Number of distinct students listed in verified transmission chains.
* **PageRank Prestige:** Calculated using power iteration with a damping factor of $\alpha = 0.85$ and a convergence tolerance of $1 \times 10^{-6}$.
* **Clustering Coefficient:** Ratio of closed triangles connected to node $i$ relative to total possible connections:
  $$C_i = \frac{2 T_i}{k_i (k_i - 1)}$$
  where $T_i$ represents the number of triangles through node $i$.

### 5.3.3 Biographical Metadata Processing
Three biographical demographic attributes are extracted from Hadith-Viz cards:
* **Generation (*Ṭabaqa*):** Mapped to an integer from 1 to 12. Missing values are imputed using the cohort median.
* **Normalized Death Year:** The recorded Hijrī year of death, min-max scaled to the continuous interval $[0, 1]$.
* **Transmission Span:** The difference in death years between the earliest recorded teacher and latest recorded student.

All 8 tabular features are standardized using z-score normalization ($z = \frac{x - \mu}{\sigma}$) prior to neural network ingestion.

---

## 5.4 Neural Architecture Specifications and Training Hyperparameters

All neural network models (P1-B_MLP, G1, G2, P2-A, P2-B_Concat, and P2-B_Attn) are implemented in PyTorch and PyTorch Geometric within `scripts/run_clean_experiments.py`.

### Table 5.2: Neural Network Hyperparameter Configuration Across Experimental Arms

| Hyperparameter Setting | Configuration Value | Methodological Rationale |
|---|---|---|
| **Optimizer** | AdamW | Decoupled weight decay prevents overfitting on small cohorts. |
| **Base Learning Rate** | $1 \times 10^{-3}$ | Standard learning rate for transformer embedding classification. |
| **Weight Decay ($L_2$ Regularization)** | $1 \times 10^{-4}$ | Constrains weight magnitudes to control gradient variance. |
| **Training Epochs** | 150 Maximum | Sufficient capacity for convergence on tabular and GNN layers. |
| **Early Stopping Strategy** | Patience: 25 epochs | Training halts if validation Macro-F1 does not improve for 25 epochs. |
| **Loss Function** | Weighted Cross-Entropy | Loss weighted by inverse class frequency to handle class imbalance. |
| **Activation Function** | ReLU | Rectified Linear Unit activation between hidden layers. |
| **Dropout Probability** | $p = 0.3$ | Injected after hidden layers to mitigate co-adaptation. |
| **Hidden Layer Dimensions** | 64 (MLP) / 32 (GNN) | Constrained layer sizes prevent parameter explosion on $N=335$. |

### 5.4.1 Inverse-Frequency Loss Weighting
To force neural models to pay proportional attention to the minority reliability class (*Below Ṣadūq*, representing only 3.9% of the multimodal cohort), the cross-entropy loss function is weighted using inverse class frequencies:

$$\mathcal{L} = - \sum_{c=1}^C w_c \cdot y_c \log(\hat{p}_c), \quad w_c = \frac{N}{C \cdot N_c}$$

The variables within this loss function are defined as follows:
* $N$ represents the total training set sample count.
* $C$ represents the number of target classes ($C = 3$).
* $N_c$ represents the count of training samples belonging to class $c$.
* $y_c$ represents the binary ground-truth indicator for class $c$.
* $\hat{p}_c$ represents the softmax predicted probability for class $c$.

---

## 5.5 Verification, Unit Testing, and Pipeline Diagnostics

Before initiating production benchmarking, the computational pipeline was verified through four diagnostic unit tests:

1. **Target Leakage String Scan:** An automated inspection script verified that zero biographical text inputs in `clean_texts.json` contained target phrases such as `قال ابن حجر` or `قال الذهبي`. All 335 entries passed the verification test.
2. **Identity Partition Integrity:** An assertion test verified that for every fold $k \in \{1, \dots, 25\}$, the intersection between training identity IDs and testing identity IDs was strictly empty ($\text{len}(V_{\text{train}} \cap V_{\text{test}}) = 0$).
3. **Graph Adjacency Symmetry Check:** Verified that the transmission adjacency matrix $A$ satisfies $A = A^T$ with zero self-loops on the diagonal prior to convolution normalization.
4. **Downstream Isnād Index Alignment:** Verified that all narrator references in the 1,578 downstream test chains correctly mapped to valid out-of-fold predictions without index mismatches or missing values.
