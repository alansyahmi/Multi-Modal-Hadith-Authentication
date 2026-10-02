# CHAPTER 3: PROJECT METHODOLOGY

This chapter presents the methodology and system architecture designed to solve the three core research objectives of this project: (PO1) translating qualitative biographical assessments into continuous narrator reliability scores, (PO2) constructing a multi-relational Knowledge Graph with structural representation learning, and (PO3) verifying chain-level authenticity against classical scholarly gradings. 

The developed pipeline transitions from raw texts to structured predictions through three distinct phases: (1) Data Ingestion and Preprocessing, (2) Knowledge Graph Construction and Node Alignment, and (3) Probabilistic Trust Scoring, Graph Neural Network (GNN) structural learning, and Chain-Level Soft Aggregation.

---

## 3.1 Research Design

This study adopts an experimental research design utilizing the Design Science Research Methodology (DSRM). The central artifact is a multi-modal computational pipeline that models both the textual sentiment of classical biographical evaluations (biographical criticism or *ʿIlm al-Jarḥ wa-t-Taʿdīl*) and the structural topology of transmission chains (*isnād* networks). The complete framework architecture is illustrated in Figure 3.1.

```
+-----------------------------------------------------------------------------------+
|                        STAGE 1: INGESTION & PREPROCESSING                         |
|  [Raw Biographical Text (Shamela)]        [Raw Hadith Transmission Chains]        |
|                  |                                        |                       |
|        [Regex Biography Parser]                 [Two-Tier Alignment]             |
|                  |                     (TF-IDF Retrieval -> RapidFuzz Matcher)    |
|                  |                                        |                       |
|        Isolated Narrator Blocks                  Aligned Sanad Sequences          |
+----------------------------------+------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------------------+
|                     STAGE 2: KNOWLEDGE GRAPH CONSTRUCTION                         |
|                     - 262,497 Narrator/Alias Nodes                                |
|                     - 987,036 Teacher-Student Directed Edges                      |
|                     - Ancestry-aware Disambiguation & SHA256 Key Hashing          |
+----------------------------------+------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------------------+
|                        STAGE 3: SCORING & PROPAGATION                             |
|                                                                                   |
|  [3.1 Jumhūr Sentiment Pipeline]       [3.2 Graph Autoencoder (GAE) Learning]     |
|   - Statement Segmentation              - Self-Supervised Link Prediction         |
|   - AraBERT Sentence Classification     - Node Embeddings (64-dimensional)        |
|   - Mean Reliability Score ($w$)        - Edge Compatibility Likelihood ($P_e$)   |
|   - Normalized Consensus Var ($\sigma$) - GNN Modulation (15% Max Mitigation)     |
|                  \                                       /                        |
|                   \                                     /                         |
|                    v                                   v                          |
|         [3.3 Soft Aggregation: $\alpha \times \text{Weakest} + (1-\alpha) \times \text{Mean}$]        |
|                                  |                                                |
|                   [3.4 Geospatial & Risk Penalties]                               |
|                                  |                                                |
|                       FINAL CALIBRATED HADITH GRADE                               |
+-----------------------------------------------------------------------------------+
```
**Figure 3.1: Flowchart of the Multi-Modal Hadith Authenticity Verification Pipeline**

---

## 3.2 Dataset Corpus Characteristics

Before executing the experimental pipeline, an exploratory data analysis was conducted to understand the statistical properties of the primary datasets across three distinct modalities: the textual/label space of the *Jarḥ* wa *Taʿdīl* classification, the topological structure of the narrator transmission network, and the spatiotemporal distribution of biographical metadata. This analysis informs parameter choices, motivates architectural decisions, and provides context for interpreting experimental outcomes.

### 3.2.1 Textual and Label Distribution (NLP Dimension)

#### Grade Classification Imbalance

The biographical encyclopedia contains 12,019 explicit *Jarḥ* wa *Taʿdīl* grade annotations across 15,448 narrator entries. The grade distribution is heavily right-skewed toward positive classifications:

* **Thiqah** (trustworthy): 3,409 (28.4%)
* **Ṣadūq** (truthful): 2,607 (21.7%)
* **Maqbūl** (acceptable): 2,088 (17.4%)
* **Majhūl** (unknown): 1,157 (9.6%)
* **Ḍaʿīf** (weak): 851 (7.1%)
* **Matrūk** (abandoned): 215 (1.8%)
* Other / unclassified: 1,692 (14.1%)

Positive classes (Thiqah, Ṣadūq, Maqbūl) collectively account for 67.4% of all grade annotations, while negative classes (Ḍaʿīf, Matrūk, Majhūl) represent only 18.5%, yielding a positive-to-negative ratio of approximately 3.6:1. This skew is not a sampling artifact but a genuine historical property — classical *Jarḥ* wa *Taʿdīl* scholarship was predominantly concerned with documenting reliable transmitters, with critical (*jārḥ*) assessments being the minority reserved for known problematic narrators. This imbalance motivates the use of both book-balanced and grade-balanced sampling strategies in the experimental design (Section 4.2.2), as raw accuracy would be misleadingly high if evaluated on the natural distribution alone.

*[VIS: Bar chart of grade category counts with log-scale y-axis]*

#### Biography Text Length Variation

The extracted biography texts exhibit extreme variance in length, ranging from empty entries to entries exceeding 3,000 characters. The distribution is heavily right-skewed:

* **Mean character length**: 1,098 | **Median**: 332
* **P10**: 68 chars | **P90**: 3,000 chars
* **IQR**: 2,137 chars (P25 = 103, P75 = 2,240)
* **Estimated mean word count**: 183 | **Median**: 55

Critically, 43.3% of entries (6,695 narrators) contain fewer than 200 characters — insufficient textual context for reliable sentiment-based scoring. This data sparsity at the short-entry tail directly explains the high quarantine rate observed during the alignment phase (Section 4.2.2), where length disparities between query texts and encyclopedia entries caused substantial alignment failure. Narrators with short biographies are disproportionately represented among the cold-start population that depends on GNN-based structural inference.

*[VIS: Histogram of biography text lengths (characters) with log-scale x-axis and annotation marking the alignment challenge threshold]*

#### Alias Density

The alias index contains 24,545 alternative name forms mapped to canonical narrator identities, with an average of 4.02 orthographic variants per canonical narrator (median = 1, max = 1,848 for highly-cited figures). Notably, 7.9% of narrators (1,935 individuals) have five or more distinct name forms, and 2.7% (652 narrators) have twenty or more. These variations arise from classical Arabic orthographic conventions — *tashkīl* (vocalisation marks), alif variants (alif maqṣūrah, alif mamdūdah, hamzah), and the *nisbah* suffix — all of which cause the same historical figure to appear under substantially different surface forms across Sanadset chain records.

The density of alias forms confirms that a simple string-matching lookup strategy is inadequate for this task and justifies the hybrid two-tier alignment framework (TF-IDF $n$-gram retrieval followed by Levenshtein threshold verification) described in Section 3.3.3.

*[VIS: Histogram of alias forms per canonical narrator with log x-axis to show heavy-tailed distribution]*

### 3.2.2 Graph Topology and Structural Sparsity (KG/GNN Dimension)

#### Scale-Free Power-Law Topology

The Hadith transmission network constructed from the full 650,986-chain Sanadset corpus comprises 262,497 unique narrator nodes connected by 987,036 directed teacher-student transmission edges. The degree distribution — computed from the 262,497-node full graph — follows a scale-free power law with exponent $\gamma \approx 2.15$, consistent with other historical citation networks and real-world social graphs. Within the 2,172-node graded subgraph (the subset of narrators linkable to biographical encyclopedia entries), the degree distribution exhibits a similar heavy-tailed pattern ($\gamma \approx 1.61$, mean degree 4.12, median 2, max 291), with the top 1% of nodes accounting for 19.8% of total connection degree.

This topological property has two direct implications for the experimental pipeline. First, the presence of high-degree hub narrators (major compilers and prolific teachers) means that even modest improvements in scoring accuracy for a single hub propagate to hundreds or thousands of downstream chains, amplifying the impact of the GAE structural embeddings. Second, the long tail of low-degree peripheral narrators (each appearing once or twice) constitutes the majority of the cold-start challenge — narrators whose reliability cannot be established from local evidence and must instead be inferred from neighbourhood context.

*[VIS: Log-log degree distribution plot with fitted power-law regression line]*
*[VIS: Complementary CDF (CCDF) plot showing heavy-tailed connectivity]*

#### The Cold-Start Challenge

Attribute sparsity is the defining structural challenge of this KG. While the full graph contains 262,497 narrator nodes, only 2,172 nodes (0.83%) are linkable to the biographical encyclopedia and therefore carry explicit grade, tabaqa, or death metadata. Within this graded subgraph, coverage is itself incomplete: 91.48% carry grade labels, 88.17% carry generation (tabaqa) information, and 63.81% record death years, leaving 8.52% of subgraph nodes with none of the three attributes. Extrapolating to the full graph, only 1,987 nodes (0.76% of 262,497) carry an explicit reliability score from the sentiment pipeline, while the remaining 260,325 nodes (99.17%) are structurally cold-start — they exist in the KG as nodes with known neighbours but no intrinsic attribute signal.

This near-total attribute absence is the primary motivation for the Graph Autoencoder (GAE) component described in Section 3.5.2. The GAE learns dense 64-dimensional node embeddings ($Z \in \mathbb{R}^{N \times 64}$) purely from the adjacency structure of the transmission network, enabling the model to propagate reliability signal from the 0.83% of labelled nodes to the unlabelled majority through neighbourhood-based message passing.

*[VIS: Venn diagram or stacked bar chart showing attribute coverage overlap across grade, tabaqa, and death metadata]*

### 3.2.3 Spatiotemporal Metadata (Domain Dimension)

#### Chronological Distribution Across Generations

The graded subgraph contains 1,915 narrator nodes with extractable generation (*tabaqa*) information, spanning the classical 11-generation framework from the Companions (Gen 2, ca. 643–664 CE) through the later transmitters (Gen 11, ca. 832–853 CE). The distribution across generations is markedly bimodal, with a pronounced peak in the later periods:

| Generation | Approximate Period | Narrators | Percentage |
|:---:|:---|:---:|:---:|
| 2 | 643 — 664 CE | 48 | 2.5% |
| 3 | 664 — 685 CE | 124 | 6.5% |
| 4 | 685 — 706 CE | 49 | 2.6% |
| 5 | 706 — 727 CE | 56 | 2.9% |
| 6 | 727 — 748 CE | 124 | 6.5% |
| 7 | 748 — 769 CE | 135 | 7.0% |
| 8 | 769 — 790 CE | 76 | 4.0% |
| 9 | 790 — 811 CE | 169 | 8.8% |
| 10 | 811 — 832 CE | 394 | 20.6% |
| 11 | 832 — 853 CE | 592 | 30.9% |

The dominance of Generations 10 and 11 (together accounting for 51.5% of all dated narrators) reflects the period of the major Hadith collectors — al-Bukhārī (d. 256 AH/870 CE), Muslim (d. 261 AH/875 CE), Abū Dāwūd (d. 275 AH/889 CE), al-Tirmidhī (d. 279 AH/892 CE), al-Nasāʾī (d. 303 AH/915 CE), and Ibn Mājah (d. 273 AH/887 CE) — whose biographical dictionaries provide the densest coverage in the *Tahdhīb al-Kamāl* source corpus. The absence of Generation 1 data reflects the convention that *Tahdhīb al-Kamāl* focuses on post-Companion narrators.

*[VIS: Bar chart of narrator count per tabaqa generation (11 groups)]*
*[VIS: Timeline with key compiler lifetimes overlaid on generation distribution]*

#### Geographic Concentration

The geo-spatial index maps 5,553 narrator entries to 29 unique municipal locations across 27 distinct coordinate pairs in the classical Islamic world. The geographic distribution is highly concentrated in four metropolitan hubs:

* **Kūfa, Iraq**: 1,374 entries (24.7%)
* **Medina, Hejaz**: 1,189 entries (21.4%)
* **Basra, Iraq**: 983 entries (17.7%)
* **Mecca, Hejaz**: 274 entries (4.9%)
* **Greater Syria (al-Shām)**: 226 entries (4.1%)
* **Egypt**: 195 entries (3.5%)
* **Damascus**: 194 entries (3.5%)
* **Baghdad**: 186 entries (3.3%)
* **Other centres** (Marw, Wāsiṭ, Rayy, Nishapur, Ḥimṣ, al-Jazīrah, Khurāsān, Ḥarrān): 872 entries (15.7%)

The four dominant centres — Kūfa, Medina, Basra, and Mecca — collectively account for 68.7% of all geo-mapped transmitter entries. The geographic distribution reflects the historical reality of early Islamic knowledge transmission: the Hejaz (Medina and Mecca) served as the origin of the prophetic tradition, while the nascent Iraqi metropolises of Kūfa and Basra became the primary centres of Hadith scholarship and legal theory during the 2nd–3rd Islamic centuries. This concentration validates the need for the geospatial penalty mechanism described in Section 3.5.3, as transmitters from disparate geographic regions would have had limited opportunity for direct face-to-face transmission.

*[VIS: Dot-distribution map over early Islamic world showing transmitter density by city]*
*[VIS: Bubble chart of narrator count per city with region colour-coding]*

---

## 3.3 Stage 1: Data Ingestion and Preprocessing

### 3.3.1 Data Sources
The framework integrates structured transmission data with unstructured biographical judgments:
1. **Biographical Corpora**: Sourced from the Shamela Digital Library, targeting Ibn Hajar al-Asqalani's *Taqrīb al-Tahdhīb* and al-Mizzi's *Tahdhīb al-Kamāl*. These serve as the basis for extracting consensus textual judgments.
2. **Transmission Corpora**: Built upon the Sanadset 650K dataset, which provides 650,986 Hadith chains, mapped against four evaluation dataset parameters across two targeted alignment benchmarks (1,200 and 6,000 successfully aligned Hadiths) and two balancing strategies (Book-Balanced and Grade-Balanced). To ensure target alignment goals are achieved despite Levenshtein quarantine filtering, fixed pre-alignment sampling targets of 1,800 and 7,000 query Hadiths are drawn from HuggingFace datasets with canonical grades (e.g., *Ṣaḥīḥ*, *Ḍaʿīf*, *Ḥasan*).

### 3.3.2 Biographical Text Parsing and Block Extraction
Raw texts are preprocessed using a custom Regular Expression (Regex) parser. The parser identifies structured markup (e.g., `# $Tarjama` tags) and entry delimiters, isolating independent text blocks for each narrator. It filters out metadata headers, page numbers, and publisher footnotes to yield clean, consonantal Arabic skeletons for analysis.

### 3.3.3 Two-Tiered Algorithmic Alignment Framework
To link transmission chains to biographical entries without a shared index, the system implements a two-tiered alignment pipeline:

```
[Raw Chain Narrator Name] 
      |
      v
+----------------------------------------------------------+
| Tier 1: TF-IDF Candidate Retrieval                       |
| - Character n-gram (char_wb, n=3 to 5)                   |
| - Cosine similarity search against 262k graph index      |
| - Extracts Top-10 closest candidates                     |
+----------------------------------------------------------+
      |
      v
+----------------------------------------------------------+
| Tier 2: Fine-Grained Lexical Verification                |
| - Levenshtein Distance (RapidFuzz token-sort matching)   |
| - Enforcement of strict threshold (t >= 0.95)            |
+----------------------------------------------------------+
      |
      +---> [Multiple Ties / Ambiguous] 
      |           |
      |           v
      |     [Ancestral Tie-Break Verification] 
      |     (Resolves via teacher/student intersection)
      |
      +---> [Failed Match (t < 0.95)] 
      |           |
      |           v
      |     [Quarantine Routing Vector] (Manual Audit Log)
      |
      +---> [Successful Unique Match] 
                  |
                  v
            [SHA256 Text Hash Key Generation]
```
**Figure 3.2: Two-Tiered Algorithmic Alignment Architecture**

1. **Tier 1: Heuristic Search Space Reduction (Information Retrieval Layer)**: Computes TF-IDF vectors using character-level $n$-grams ($n \in [3, 5]$) on normalized text. This isolates a candidate pool of the top 10 matches for any input query out of the 262k entries, bypassing computational bottlenecks.
2. **Tier 2: Fine-Grained Lexical Verification (Fuzzy Matching Layer)**: Applies token-sort Levenshtein distance calculations via the `RapidFuzz` library to select the optimal candidate. Matches must exceed a strict similarity threshold ($t \ge 0.95$).
3. **Ancestral Node Tie-Breaking**: When multiple candidates yield identical fuzzy ratios, the algorithm resolves the tie-break by checking the immediate predecessor and successor nodes (the narrator's teachers and students) against the biographical network.
4. **SHA256 Identity Preservation**: Aligned entities are assigned an immutable, unique primary key calculated as a SHA-256 hash of their normalized Arabic consonantal skeleton.
5. **Quarantine Routing**: Records failing to meet the fuzzy matching threshold or showing irreconcilable network overlaps are routed to a JSON quarantine log for manual scholar-in-the-loop audit.

---

## 3.4 Stage 2: Knowledge Graph Construction

### 3.4.1 Relational Graph Dimensions
Rather than utilizing a small sample, the system builds a comprehensive Knowledge Graph (KG) mapping the historical transmission network. The dimensions of the constructed graph are detailed in Table 3.1.

| Graph Entity / Relation | Actual Count |
| :--- | :--- |
| **Total Graph Nodes** | 262,497 |
| **Total Graph Edges** | 987,036 |
| **Narrator Encyclopedia Entries** | 15,448 |
| **Alternative Aliases** | 24,545 |
| **Inter-Narrator Family Links** | 13,592 |

**Table 3.1: Dimensions of the Constructed Relational Knowledge Graph**

### 3.4.2 Node & Edge Attribute Representation
Each narrator node is populated with rich attributes extracted from classical literature:
- **Tabaqa (Generation)**: Coded as an integer range from 1 to 12 matching Ibn Hajar's temporal classification (ranging from the *Ṣaḥāba* [Companions, ~1st Century AH] to late compilers [~3rd Century AH]).
- **Geographic Travel Regions**: Categorized lists of historically visited cities (e.g., Medina, Kufa, Basra, Damascus, Baghdad, Cordoba).
- **Relational Edges**: Directed connections mapping teacher-to-student propagation ($u \to v$).

### 3.4.3 Algorithmic Name Mapping (Alias & Ancestry Matcher)
Due to the absence of trained Named Entity Recognition (NER) models with verified entity-level F1 metrics in this domain, narrator boundary parsing is executed via heuristic token extraction, index matching, and alias resolution. This maps narrator chains directly to the canonical node database. Token classification via fine-tuned Transformer-based NER models (AraBERT sequence labeling) is demoted to future work.

---

## 3.5 Stage 3: Probabilistic Trust Scoring and Evaluation

### 3.5.1 Semantic Sentiment Analysis for Continuous Reliability Scoring
Instead of assigning binary weights, the framework implements a multi-tiered **Jumhūr Sentiment Aggregation pipeline** to estimate narrator reliability. This models the differences of opinion among classical critics (*Jumhūr* or consensus) as a continuous scoring task.

```
[Raw Biographical Entry Text] 
              |
              v
  [Critic Statement Segmentation] 
  (Isolates statements per critic via CAMeL Tools punctuation/prefix parser)
              |
              v
  [AraBERT Sentence Classification] 
  (5-Class Ordinal Classification of praise [Taʿdīl] vs criticism [Jarḥ])
              |
              v
       [Score Mapping] 
       (Converts classifications to continuous weights in [0.05, 0.85])
              |
              v
     +--------+--------+
     |                 |
     v                 v
[Jumhūr Aggregation] [Consensus Normalization]
(Mean Score: $r_i$)    (Normalized Consensus Variance: $\sigma_{consensus}$)
```
**Figure 3.3: Multi-Tiered Jumhūr Sentiment Aggregation Pipeline**

1. **Critic Statement Segmentation**: Extracts individual evaluative statements using Arabic punctuation patterns and scholar-specific prefix indicators (e.g., *"قال أحمد"* [Ahmad said]).
2. **AraBERT Sentence-Level Classification**: Predicts the sentiment of each statement using an AraBERT-based classifier pre-trained on Arabic sentiment data. To model narrator reliability from evaluative biographical statements, the system formulates the task as discrete ordinal classification rather than direct continuous regression. In qualitative NLP tasks, a common limitation known as Deep Imbalanced Regression (DIR) is where continuous regression objectives frequently suffer from severe score compression and regression toward the empirical mean when predicting unbounded numerical values across skewed qualitative classes (Yang et al., 2021). By formulating the task as a 5-class ordinal classification problem (*matrūk*, *ḍaʿīf*, *maqbūl*, *ṣadūq*, *thiqah*), the architecture enforces crisp decision boundaries between quality bands before mapping predictions to continuous ordinal score weights $[0.05, 0.85]$.
3. **Score Mapping**: Translates raw sentiment predictions into ordinal class scores ranging from $0.05$ (severely weak / *matrūk*) to $0.85$ (highly reliable / *thiqah*). This ceiling of $0.85$ represents the inherent historical uncertainty of human transmission; classical Hadith criticism is fundamentally a *via negativa* process focusing on defect detection, meaning no narrator is modeled with absolute mathematical certainty ($1.0$). Ultra-high terms (*Thābit*, *Ḥāfiẓ*, *Imām*) under Class 4 and severe defects (*Waddāʿ*, *Dajjāl*) under Class 0 are grouped to prevent the AraBERT model from suffering from data sparsity issues. The mapping of qualitative scholarly judgments to numerical values and their classical Arabic synonyms is detailed in Table 3.2:

Table 3.2: Semantic Score Mapping and Classical Arabic Synonyms for Biographical Tiers
| Model Quality Tier | Classical Arabic Grade Synonyms | Sentiment Polarity | Numeric Score Mapping ($s_i$) | Academic Explanation |
| :--- | :--- | :---: | :---: | :--- |
| **Thiqah** (Class 4) | **ثقة**, **ثقة ثقة**, **ثقة ثبت**, **ثقة حافظ**, **ثقة متقن**, **ثبت**, **حافظ**, **إمام**, **مأمون**, **صحابي** | Taʿdīl (Praise) | **$0.85$** | Maximum reliability tier for the textual semantic layer. |
| **Ṣadūq** (Class 3) | **صدوق**, **لا بأس به**, **ليس به بأس**, **محله الصدق** | Taʿdīl (Praise) | **$0.70$** | Truthful narrator but susceptible to minor errors. |
| **Maqbūl** (Class 2) | **مقبول**, **صدوق يهم**, **صدوق سيء الحفظ**, **سيء الحفظ** | Marginal / Neutral | **$0.40$** | Acceptable only when supported by corroborating chains. |
| **Ḍaʿīf** (Class 1) | **ضعيف**, **ضعيف الحديث**, **لين**, **لين الحديث**, **ليس بالقوي**, **فيه ضعف**, **مستور**, **مجهول الحال** | Jarḥ (Criticism) | **$0.15$** | Unreliable or weak narrator. |
| **Matrūk** (Class 0) | **متروك**, **متروك الحديث**, **ساقط**, **مختلط**, **مجهول العين**, **كذاب**, **وضاع**, **دجال** | Jarḥ (Criticism) | **$0.05$** | Abandoned narrator (minimum baseline boundary). |

4. **Jumhūr Aggregation**: Computes the narrator's base semantic reliability score ($r_i$) as the mean of their individual statement scores:
   $$r_i = \frac{1}{N} \sum_{i=1}^{N} s_i$$
5. **Consensus Score Calculation**: Measures the variance of opinions for each narrator. To represent scholarly agreement on a normalized scale, the consensus score is calculated as:
   $$\text{agreement\_score} = 1.0 - \frac{\text{Var}(s)}{0.16}$$
   Where the denominator ($0.16$) represents the maximum possible variance of a continuous score bounded on the interval $[0.05, 0.85]$. The maximum variance occurs at a 50/50 split at the boundaries:
   $$\text{Var}_{max} = \left(\frac{0.85 - 0.05}{2}\right)^2 = (0.40)^2 = 0.16$$

### 3.5.2 Unsupervised Graph Autoencoders (GAE) for Structural Representation Learning
Instead of attempting to predict narrator grades directly from a supervised GNN (which suffers from label scarcity and introduces bias), the structural analysis component uses an unsupervised **Graph Autoencoder (GAE)**.

#### GAE Architecture
The GAE encodes node structures into a latent space and reconstructs network topology using a link prediction objective:
- **Encoder**: A 2-layer Graph Convolutional Network (GCN) that takes node feature vectors $X$ and the adjacency matrix $A$ to compute 64-dimensional latent embeddings $Z$:
  $$Z = \text{GCN}(X, A)$$
- **Decoder**: A simple inner product that reconstructs the link compatibility score ($P_e$) for any edge $(u, v)$:
  $$\hat{A}_{u,v} = \sigma(z_u^T z_v)$$
- **Objective Function**: The reconstruction loss is defined as binary cross-entropy on link existence:
  $$\mathcal{L} = -\sum_{(u,v) \in E} \log(\sigma(z_u^T z_v)) - \sum_{(u,v') \notin E} \log(1 - \sigma(z_u^T z_{v'}))$$

The GAE is trained for 200 epochs using the Adam optimizer (learning rate = $1 \times 10^{-3}$, dropout = $0.3$). The model achieves robust link prediction performance on a 20% test split:
- **ROC-AUC**: 92.39%
- **Average Precision (AP)**: 94.04%

#### GNN Uncertainty Modulation
The GNN embeddings modulate the narrator reliability score rather than overriding textual criticism. For a narrator with a marginal base score, the system checks their structural embedding. If the narrator exhibits high link compatibility ($P_e \ge 0.50$) and is well-integrated within reliable transmission paths, a GNN mitigation factor is applied to reduce their base weakness penalty by up to a maximum of 15%:
$$\text{effective\_penalty} = \text{base\_penalty} \times (1 - \text{GNN\_mitigation})$$

---

### 3.5.3 Chain-Level Scoring (Soft Aggregation)

The final authenticity score of a Hadith transmission chain ($S_{chain}$) is computed by combining modulated individual narrator reliability scores using a soft aggregation function, which is then adjusted by topological, geospatial, and temporal penalty modifiers.

#### 1. Modulated Narrator Score ($W_i$)
For each narrator $i$ in a transmission chain of length $M$, the system first computes an individual modulated score $W_i$:
$$W_i = w_i \times \text{risk}_i$$
where $\text{risk}_i \in [0.0, 1.0]$ represents narrator-specific risk modifiers. To extract these modifiers, a rule-based linguistic pattern-matching pipeline scans the normalized biographical text during the data preparation stage for classical Arabic root patterns representing specific transmission defects:
| Risk Modifier Type | Arabic Lexical Patterns | Multiplier | Technical Description |
| :--- | :--- | :---: | :--- |
| **Concealed Weakness (*tadlīs*)** | `تدليس`, `مدلس`, `يدلس` | $0.80$ | Narrator conceals their immediate teacher's identity using ambiguous terms. |
| **Mental Deterioration (*ikhtilāṭ*)** | `مختلط`, `اختلط بآخره`, `تغير بآخره`, `تغير قبل` | $0.50$ | Narrator suffered memory loss or cognitive decline in later life. |
| **Identity Uncertainty (*jahālah*)** | `مجهول العين`, `مجهول الحال` | $0.30$ | Narrator's status or reliability is unknown to biography compilers. |
| **Fabrication (*kadhib*)** | `كذاب`, `يضع الحديث`, `وضاع`, `متهم بالكذب` | $0.00$ | Narrator is accused of lying or deliberate fabrication (chain-breaking). |

If a narrator's biography contains multiple defects, the system selects the most severe penalty by calculating the minimum modifier:
$$\text{risk}_i = \min_{k} (\text{modifier}_k)$$
If no defects are detected, the modifier defaults to $\text{risk}_i = 1.0$.

The base narrator score $w_i$ combines their semantic reliability score $r_i$ with severity-band and consensus modulations:
$$w_i = r_i \times d_i \times d_{agree, i}$$
The variables in this calculation are defined as follows:
* **Severity-Band Discount ($d_i$)**: Base discount factor mapped to narrator reliability ranges:
  $$d_i = \begin{cases} 
    1.0 & \text{if } r_i \ge 0.80 \text{ (Thiqah)} \\
    1.0 & \text{if } r_i \ge 0.70 \text{ (Saduq)} \\
    0.85 & \text{if } r_i \ge 0.40 \text{ (Weak)} \\
    0.60 & \text{if } r_i \ge 0.20 \text{ (Very Weak)} \\
    0.25 & \text{if } r_i < 0.20 \text{ (Severe Weakness)} 
  \end{cases}$$
* **GNN Uncertainty Modulation**: If a narrator falls into a "Weak" or "Very Weak" category, but exhibits high structural integration in the GAE Knowledge Graph (indicated by a low GNN Anomaly Score $A_i < 0.25$), a structural mitigation factor is applied to the discount:
  $$d_i^{mitigated} = d_i + (1.0 - d_i) \times 0.15$$
* **Disagreement Risk Modifier ($d_{agree, i}$)**: Penalizes narrators with high scholarly disputes (low consensus scores):
  $$d_{agree, i} = 1.0 - 0.20 \times (1.0 - \text{agreement\_score}_i)$$
  where $\text{agreement\_score}_i = 1.0 - \frac{\text{Var}(s)}{0.16}$ represents the consensus rating.

#### 3.5.3.2 Base Chain Score ($S_{base}$)
Rather than multiplying narrator scores ($S_{chain} = \prod W_i$), which causes exponential score decay on longer chains, the pipeline uses a parameterized **Soft Aggregation** formula:
$$S_{base} = \alpha \times \min_{i} (W_i) + (1 - \alpha) \times \left( \frac{1}{M}\sum_{i=1}^M W_i \right)$$
where $\alpha \in \{0.25, 0.50, 0.75\}$ is empirically optimised to balance the weakest-narrator bottleneck with overall chain quality.

#### 3.5.3.3 Chain-Level Penalty Modifiers
The base chain score is adjusted by multiplying three independent penalty factors:
1. **Ambiguous Transmission (ʿAnʿanah) Penalty ($P_{amb}$)**: Applied if a narrator categorized as a *Mudallis* (known to practice tadlīs) transmits using the ambiguous particle *"عن"* (*ʿan*, from) instead of explicit direct audition (e.g., *"حدثنا"*, *haddathana*).
2. **Temporal Validation (Inqitāʿ) Penalty ($P_{temp}$)**: Evaluates chronological feasibility between adjacent narrator generations ($tabaqāt$). If a teacher is younger than the student (chronological impossibility) or exhibits significant gaps, a broken transmission (*inqitāʿ*) is flagged:
  $$P_{temp} = \begin{cases} 
    0.40 & \text{if severe gap (e.g., teacher } \ge \text{ student } + 5 \text{ generations)} \\ 
    0.60 & \text{if moderate gap (e.g., teacher } \ge \text{ student } + 3 \text{ generations)} \\
    1.0 & \text{otherwise} 
  \end{cases}$$
3. **Geospatial Validation Penalty ($P_{geo}$)**: Compares travel paths between adjacent transmitters. If lifespans overlap but travel records show no plausible geographic intersection (distance between locations $> 3,000$ km), a distance penalty is applied:
  $$P_{geo} = \prod_{j=1}^{M-1} g_j, \quad \text{where } g_j = \begin{cases} 0.85 & \text{if distance } > 3,000 \text{ km} \\ 1.0 & \text{otherwise} \end{cases}$$
4. **GNN Path Compatibility Penalty ($P_{gnn}$)**: Uses the GAE link prediction scores to check structural continuity between adjacent nodes in the chain:
  $$P_{gnn} = \begin{cases} 0.85 & \text{if } \min_{j} (\sigma(z_j^T z_{j+1})) < 0.25 \\ 1.0 & \text{otherwise} \end{cases}$$

#### 3.5.3.4 Final Chain Score ($S_{chain}$)
The final authenticated score is the product of the soft-aggregated base score and the edge penalty modifiers:
$$S_{chain} = S_{base} \times P_{temp} \times P_{geo} \times P_{gnn}$$
This continuous score is subsequently mapped to canonical grades (*Ṣaḥīḥ*, *Ḥasan*, *Maqbul*, *Ḍaʿīf*, *Matrūk*) using calibrated percentile thresholds.

---

### 3.5.4 Evaluation Protocol
The pipeline's grading accuracy is validated against classical Hadith classifications across multiple datasets at the following scales:
- **1,200 Aligned Benchmarks**: Evaluated across Book-Balanced (300 per collection) and Grade-Balanced parameters using fixed pre-alignment sampling targets of 1,800 query Hadiths.
- **6,000 Aligned Benchmarks**: Evaluated across Book-Balanced (~1,166 per collection) and Grade-Balanced parameters using fixed pre-alignment sampling targets of up to 7,000 query Hadiths (utilising all available graded records for grade-balanced sets).
- **435,000 Bulk Grading**: Scale testing performed on 435,351 valid chains from the Sanadset corpus to analyze the overall score distribution.

The model maps the final continuous chain score to five ordinal classes: *Ṣaḥīḥ* (Authentic), *Ḥasan* (Good), *Maqbul* (Acceptable), *Ḍaʿīf* (Weak), and *Matrūk* (Abandoned).
Threshold calibration is performed using percentile-matching rather than standard regression to prevent threshold collapse.
