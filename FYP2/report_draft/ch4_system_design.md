# Chapter 4: System and Experiment Design

## 4.1 Research Questions to Hypotheses Mapping

To evaluate the predictive validity of biographical text and transmission network topology, this study establishes a formal hypothesis testing framework. Each research question corresponds to specific null ($H_0$) and alternative ($H_1$) hypotheses:

### 4.1.1 Research Question 1 (PO1: Linguistic and Semantic Text Modeling)
* **$H_{0,1}$ (Null Hypothesis):** Pretrained contextual AraBERT semantic embeddings produce no statistically significant improvement in 3-tier Macro-F1 over a regularized lexical TF-IDF baseline when evaluated on earlier-critic biographical texts ($p \ge 0.05$).
* **$H_{1,1}$ (Alternative Hypothesis):** Pretrained contextual AraBERT semantic embeddings produce a statistically significant improvement in 3-tier Macro-F1 over the lexical TF-IDF baseline ($p < 0.05$).

### 4.1.2 Research Question 2 (PO2: Graph Neural Networks and Multimodal Fusion)
* **$H_{0,2a}$ (Null Hypothesis on Graph Message Passing):** Graph Convolutional message passing over transmission edges (G2) produces no statistically significant improvement in 3-tier Macro-F1 compared to an MLP trained on node topological features without graph edges (G1) ($p \ge 0.05$).
* **$H_{1,2a}$ (Alternative Hypothesis on Graph Message Passing):** Graph Convolutional message passing (G2) produces a statistically significant improvement in Macro-F1 compared to the node-feature MLP (G1) ($p < 0.05$).
* **$H_{0,2b}$ (Null Hypothesis on Multimodal Fusion):** Fusing transmission graph topology with AraBERT text embeddings via Multimodal GNN Cross-Attention (P2-B_Attn) produces no statistically significant improvement in Macro-F1 compared to an Early Fusion MLP operating without transmission edges (P2-A) ($p \ge 0.05$).
* **$H_{1,2b}$ (Alternative Hypothesis on Multimodal Fusion):** Multimodal GNN Cross-Attention (P2-B_Attn) produces a statistically significant improvement in Macro-F1 compared to Early Fusion MLP (P2-A) ($p < 0.05$).

### 4.1.3 Research Question 3 (PO3: Downstream Chain Consistency)
* **$H_{0,3}$ (Null Hypothesis on Downstream Validation):** Weakest-link aggregation of predicted narrator reliability scores produces no statistically significant correlation with ground truth chain tiers across historical transmission paths ($p \ge 0.05$).
* **$H_{1,3}$ (Alternative Hypothesis on Downstream Validation):** Weakest-link aggregation of predicted narrator reliability scores produces a statistically significant positive rank correlation with ground truth chain tiers ($\rho > 0, p < 0.05$).

---

## 4.2 The Ten-Arm Experimental Matrix

To isolate the independent and interactive contributions of text representations, graph topology, and demographic metadata, the experimental suite executes ten distinct model arms. Table 4.1 outlines the configuration, input modalities, parameter dimensions, and specific objectives of each experimental arm.

### Table 4.1: Ten-Arm Controlled Experimental Matrix

| Arm ID | Model Identifier | Input Modalities | Feature Dimensions | Classifier Architecture | Research Objective Evaluated |
|---|---|---|---|---|---|
| **B0** | Majority Baseline | None (Class distribution) | 0 | Dummy Majority Class | Establishes the lower bound of naive majority guessing (Tier 2). |
| **B1** | Stratified Random | Prior class probabilities | 0 | Multinomial Sampling | Establishes chance performance preserving class imbalance. |
| **B2** | Name + Meta Probe | Name n-grams + Metadata | 53 | Balanced Logistic Regression | Tests whether surface name tokens or metadata leak target labels. |
| **P1-A** | Lexical TF-IDF LR | Earlier-critic text n-grams | 5,000 | Balanced Logistic Regression | Establishes the regularized n-gram keyword baseline. |
| **P1-B** | AraBERT Semantic LR | Earlier-critic text embeddings | 768 | Balanced Logistic Regression | Evaluates linear separability of frozen AraBERT embeddings. |
| **P1-B_MLP** | AraBERT Semantic MLP | Earlier-critic text embeddings | 768 | 2-layer MLP (Hidden: 64) | Evaluates non-linear transformations of text embeddings alone. |
| **G1** | Graph Features MLP | Topological + Meta features | 8 | 2-layer MLP (Hidden: 32) | Tests tabular network and metadata features without edge topology. |
| **G2** | Graph GCN | Graph adjacency matrix + Features | 8 | 2-layer GCN (Hidden: 32) | Evaluates message passing over transmission edges alone. |
| **P2-A** | Early Fusion MLP | AraBERT (768) + Tabular (8) | 776 | 2-layer MLP (Hidden: 64) | Evaluates multimodal concatenation without graph message passing. |
| **P2-B_Concat** | Multimodal GCN Concat | Graph adjacency + AraBERT + Tabular | 776 | 2-layer GCN (Hidden: 64) | Evaluates standard GCN message passing over fused multimodal nodes. |
| **P2-B_Attn** | Multimodal GCN Attention | Graph adjacency + AraBERT + Tabular | 776 | Cross-Modal GNN Attention | Evaluates learned attention gating between text and graph features. |
| **P2-C** | Topology Isolation MLP | Topological features only ($x_{\text{topo}}$) | 5 | 2-layer MLP (Hidden: 16) | Controls for topological influence without biographical metadata. |
| **P2-D** | Metadata Isolation MLP | Biographical metadata only ($x_{\text{meta}}$) | 3 | 2-layer MLP (Hidden: 16) | Controls for generation and death year influence without network topology. |

---

## 4.3 Model Architectures and Multimodal Fusion Schemes

Figure 4.1 illustrates the structural data flow across the primary model families investigated in this study.

```
+-----------------------------------------------------------------------------------+
| Text Pipeline: Earlier-Critic Biographical Statements (Zero Target Leakage)       |
| Text Normalization -> Subword Tokenization -> Frozen AraBERT v2 -> 768-dim Vector |
+-----------------------------------------------------------------------------------+
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
         [P1-B: Text-Only MLP]                     [P2-A: Early Fusion MLP]
       (768 -> 64 -> 3 classes)                   (776 -> 64 -> 3 classes)
                                                              ▲
                                                              │ Concatenation
+-------------------------------------------------------------┴---------------------+
| Graph Pipeline: Transmission Adjacency Matrix A (354 Undirected Edges)            |
| Tabular Features: 5 Topological Metrics + 3 Demographic Metadata (8-dim Vector)   |
+-----------------------------------------------------------------------------------+
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
            [G2: Graph GCN]                      [P2-B_Attn: Multimodal GNN]
       (8 -> 32 -> 3 classes)                    (Cross-Attention Gating)
```

### 4.3.1 Text Classification Architectures
Text representations are modeled through two complementary methodologies:
* **Lexical TF-IDF Model (P1-A):** Text is tokenized into word and character n-grams (ranges 1 to 3 words, 3 to 5 characters). Sublinear term-frequency scaling is applied: $\text{tf} = 1 + \log(\text{count})$. The feature matrix is restricted to the top 5,000 components and classified using regularized Logistic Regression with inverse class frequency weighting:

$$w_c = \frac{N}{C \cdot N_c}$$

The variables within this weighting formula are defined as follows:
* $w_c$ represents the penalty weight assigned to class $c$.
* $N$ represents the total number of training samples.
* $C$ represents the total number of classes ($C = 3$).
* $N_c$ represents the number of training samples belonging to class $c$.

* **AraBERT Contextual Semantic Models (P1-B and P1-B_MLP):** Biographical text is encoded through the frozen transformer backbone of `aubmindlab/bert-base-arabertv2`. The mean-pooled vector $e_i \in \mathbb{R}^{768}$ across all non-padding tokens is extracted as the semantic representation. P1-B applies regularized Logistic Regression directly to $e_i$. P1-B_MLP passes $e_i$ through a fully connected layer (dimension 64), Batch Normalization, Dropout ($p = 0.3$), ReLU activation, and a final linear projection to the 3 output logits.

### 4.3.2 Graph-Only Architectures
Graph representation learning evaluates whether network position correlates with narrator reliability:
* **Graph Features MLP (G1):** Evaluates the 8-dimensional node feature vector $x_i$ through a 2-layer MLP (hidden dimension 32, dropout 0.2) without providing the adjacency matrix $A$. This arm serves as the non-relational tabular baseline.
* **Graph Convolutional Network (G2):** Ingests node features $X \in \mathbb{R}^{N \times 8}$ and the adjacency matrix $A$. The 2-layer GCN updates representations via neighborhood smoothing:

$$Z = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} \text{ReLU} \left( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} X W_0 \right) W_1$$

The variables within this operation are defined as follows:
* $\tilde{A}$ represents the adjacency matrix with added identity self-loops ($A + I_N$).
* $\tilde{D}$ represents the diagonal degree matrix of $\tilde{A}$.
* $W_0 \in \mathbb{R}^{8 \times 32}$ and $W_1 \in \mathbb{R}^{32 \times 3}$ represent trainable weight matrices.
* $Z \in \mathbb{R}^{N \times 3}$ represents the output logits for the three reliability classes.

### 4.3.3 Multimodal Fusion Schemes
To evaluate the interaction between text semantics and transmission topology, three fusion strategies are implemented:
* **Early Fusion MLP (P2-A):** Concatenates the 768-dimensional AraBERT text embedding $e_i$ and the 8-dimensional tabular feature vector $x_i$ into a joint vector $u_i = [e_i \,\|\, x_i] \in \mathbb{R}^{776}$. The joint vector is classified using a 2-layer MLP (hidden dimension 64, dropout 0.3) without graph edge propagation.
* **Multimodal GCN Concatenation (P2-B_Concat):** Treats $u_i \in \mathbb{R}^{776}$ as the initial node feature matrix in a 2-layer GCN. Message passing smooths both text embeddings and topological features simultaneously across transmission edges.
* **Multimodal GNN with Cross-Attention (P2-B_Attn):** Ingests text embeddings and graph features through parallel projection heads. A cross-modal attention gating mechanism computes an attention coefficient $\alpha_i \in [0, 1]$ determining whether node $i$ should rely predominantly on its biographical text signal or its transmission neighborhood signal.

---

## 4.4 Evaluation Metrics and Decision Criteria

Due to the pronounced class imbalance in historical biographical corpora (where authentic and truthful narrators represent over 85% of canonical transmitters), evaluation metrics must be chosen with strict methodological care.

### 4.4.1 Primary Metric: Macro-Averaged F1-Score
Overall accuracy is structurally flawed on imbalanced datasets because a trivial majority classifier achieves high accuracy while failing to identify unreliable narrators. Therefore, the **3-Tier Macro-F1 score** is locked as the primary evaluation metric:

$$\text{Macro-F1} = \frac{1}{C} \sum_{c=1}^{C} F1_c = \frac{1}{3} \left( F1_{\text{Below Ṣadūq}} + F1_{\text{Ṣadūq}} + F1_{\text{Thiqah}} \right)$$

The variables within this metric are defined as follows:
* $C$ represents the total number of evaluation tiers ($C = 3$).
* $F1_c$ represents the harmonic mean of precision and recall for class $c$:

$$F1_c = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

### 4.4.2 Secondary Evaluation Metrics
Supporting evaluation metrics provide secondary perspectives on classification behavior:
* **Overall Classification Accuracy:** Percentage of total out-of-fold test instances correctly classified.
* **Quadratic Weighted Cohen's Kappa ($\kappa$):** Measures inter-rater agreement between predicted and ground-truth ordinal tiers, penalizing distant category confusions (such as misclassifying *Below Ṣadūq* as *Thiqah*) substantially more heavily than adjacent confusions:

$$\kappa_w = 1 - \frac{\sum_{i=1}^C \sum_{j=1}^C w_{ij} O_{ij}}{\sum_{i=1}^C \sum_{j=1}^C w_{ij} E_{ij}}, \quad w_{ij} = \frac{(i - j)^2}{(C - 1)^2}$$

The variables within the weighted Kappa calculation are defined as follows:
* $O_{ij}$ represents the observed proportion of narrators with true tier $i$ and predicted tier $j$.
* $E_{ij}$ represents the expected proportion under chance agreement.
* $w_{ij}$ represents the quadratic distance penalty between tier $i$ and tier $j$.

### 4.4.3 Statistical Hypothesis Testing Protocol
All pairwise model comparisons are evaluated across 25 paired cross-validation folds using two-tailed paired Student's t-tests ($df = 24$) with a significance threshold of $\alpha = 0.05$. If the $p$-value satisfies $p < 0.05$, the null hypothesis is rejected; otherwise, the study fails to reject the null hypothesis.

---

## 4.5 Downstream Chain Validation Formulation

Project Objective 3 (PO3) validates individual narrator predictions by examining how classification errors propagate into downstream transmission chains (*isnāds*).

### 4.5.1 The Weakest-Link Aggregation Operator
Classical hadith methodology evaluates an isnād according to the principle that a chain of transmission is only as strong as its weakest link (*al-isnād yattabiʿ adʿaf man fīh*). Formally, let $C_k = (v_1, v_2, \dots, v_m)$ represent a transmission chain of $m$ historical narrators. The ground-truth reliability score of the chain $S_{\text{oracle}}(C_k)$ is determined by the minimum narrator grade in the sequence:

$$S_{\text{oracle}}(C_k) = \min_{i \in \{1, \dots, m\}} y_i$$

The computationally predicted chain reliability score $S_{\text{pred}}(C_k)$ applies identical weakest-link aggregation to out-of-fold narrator predictions:

$$S_{\text{pred}}(C_k) = \min_{i \in \{1, \dots, m\}} \hat{y}_i$$

The variables within these formulas are defined as follows:
* $C_k$ represents the $k$-th verified historical isnād chain.
* $y_i \in \{0, 1, 2\}$ represents the ground-truth reliability tier of narrator $v_i$.
* $\hat{y}_i \in \{0, 1, 2\}$ represents the model's out-of-fold predicted reliability tier for narrator $v_i$.
* $S_{\text{oracle}}(C_k)$ represents the definitive chain tier according to canonical synthesizers.
* $S_{\text{pred}}(C_k)$ represents the aggregate reliability index produced by the computational pipeline.

### 4.5.2 Downstream Evaluation Metrics
Downstream chain validity is quantified using five diagnostic metrics:
1. **Chain Tier Accuracy (%):** Exact match percentage between $S_{\text{pred}}(C_k)$ and $S_{\text{oracle}}(C_k)$.
2. **Chain Binary F1:** Evaluates binary acceptance where Tiers 2 and 1 represent acceptable narrations (*Maqbūl*) and Tier 0 represents rejected narrations (*Mardūd*).
3. **Chain Mean Absolute Error (MAE):** Average distance between predicted and true chain tiers:

$$\text{MAE}_{\text{chain}} = \frac{1}{K} \sum_{k=1}^K |S_{\text{pred}}(C_k) - S_{\text{oracle}}(C_k)|$$

4. **Error Propagation Ratio:** The ratio of chain-level MAE to node-level MAE ($\text{EPR} = \text{MAE}_{\text{chain}} / \text{MAE}_{\text{node}}$), measuring whether weakest-link aggregation amplifies or dampens individual classification errors.
5. **Spearman Rank Correlation ($\rho$):** Non-parametric rank correlation between predicted and ground-truth chain scores.
