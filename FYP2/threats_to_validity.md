# Threats to Validity

This document provides a formal, academically rigorous evaluation of potential threats to the validity of the empirical findings presented in this study. Following established methodological frameworks in empirical software engineering and computational machine learning (Wohlin et al., 2012), the threats are structured into four core dimensions: Internal Validity, Construct Validity, External Validity, and Statistical Conclusion Validity.

---

## 1. Internal Validity

Internal validity concerns the extent to which the observed experimental outcomes are attributable to the investigated independent variables (representation modalities and model architectures) rather than extraneous confounding factors, procedural artifacts, or data leakage.

### 1.1 Target and Temporal Data Leakage
A critical vulnerability in automated rijal evaluation is temporal and target leakage:
* **Synthesis Authority Contamination**: Biographers from later Islamic centuries frequently summarized earlier critical verdicts while integrating their own editorial synthesis. In particular, later monumental biographical encyclopedias (such as *Tahdhib al-Kamal* by al-Mizzi, d. 742 AH) contain editorial introductions and synthesized commentary alongside verbatim classical quotations.
* **Mitigation Protocol**: The data curation pipeline enforces a strict temporal firewall implemented in `src/canonical_data_pipeline.py`. Eleven post-classical compiling and synthesizing authorities who operated between the 7th and 9th centuries AH were systematically purged from all textual inputs. Specifically, all quotations attributed to al-Mizzi (676 comments removed), Ibn Hajar al-Asqalani, al-Dhahabi, al-Birzali, al-Iraqi, Ibn Kathir, Ibn Rajab, al-Sakhawi, al-Suyuti, Badr al-Din al-Ayni, and Mughultay were excluded. Only verbatim statements attributed to classical critics operating between the 2nd and 4th centuries AH (such as Yahya ibn Ma'in, Ahmad ibn Hanbal, al-Bukhari, Abu Hatim al-Razi, Abu Zur'ah al-Razi, and al-Nasa'i) were retained as input features.
* **Label Disjointness**: Target labels were extracted exclusively from Ibn Hajar's *Taqrib al-Tahdhib*. Because Ibn Hajar's own statements were purged from the input features, the models were strictly prevented from reading the target label or its immediate stylistic paraphrases within the input text.

### 1.2 Information Leakage Across Cross-Validation Folds
Data leakage during feature preprocessing can artificially inflate predictive performance:
* **Lexical Feature Extraction**: In all evaluation folds, TF-IDF vectorizers were fitted strictly on the training partition of that specific fold. Vocabulary construction, document frequency thresholds, and inverse document frequency weights were computed without exposure to validation or test instances.
* **Continuous Feature Scaling**: StandardScaler instances for tabular and topological features were fitted exclusively on training node features. Validation and test node features were transformed using the pre-fitted training parameters.
* **Neural Early Stopping**: Early stopping checkpoints for neural multi-layer perceptrons and graph convolutional networks were monitored on an internal validation split isolated entirely within the training fold (80% train-sub, 20% validation-sub), ensuring that test partitions remained completely untouched until final evaluation.

### 1.3 Transductive Graph Representation Constraints
In the Graph Convolutional Network (GCN) benchmark, the transmission network was evaluated under a transductive graph classification formulation:
* **Adjacency Matrix Access**: All 335 nodes in the connected multimodal core were present within the PyTorch Geometric adjacency matrix. While node labels were strictly partitioned into 80% training and 20% test subsets per fold, the message passing operator aggregated neighborhood embeddings across both labeled and unlabeled nodes.
* **Assessment**: This setup reflects the historical reality that narrator transmission networks are structurally static historical graphs rather than dynamically evolving social networks. However, to eliminate the possibility that performance was driven solely by transductive edge snooping, inductive tabular baselines (P2-A Early Fusion MLP) were evaluated alongside GCN models, demonstrating that non-graph tabular architectures achieved superior Macro-F1 scores without relying on message passing across test nodes.

---

## 2. Construct Validity

Construct validity examines whether the operationalized metrics and computational representations accurately reflect the theoretical constructs they intend to measure.

### 2.1 Operationalization of Narrator Reliability
In Islamic hadith scholarship (*'Ilm ar-Rijal*), narrator reliability (*ta'dil*) is a multifaceted qualitative judgment comprising moral integrity (*'adalah*) and retentive precision (*dabt*):
* **Three-Tier Categorization**: This study operationalizes continuous scholarly evaluations into three discrete ordinal tiers:
  * Tier 2: *Thiqah* (Trustworthy, Reliable)
  * Tier 1: *Saduq* (Truthful, Acceptable with Reservation)
  * Tier 0: *Below Saduq* (*Da'if*, Weak, Rejected)
* **Threat**: Nuanced classical distinctions (such as distinguishing between minor memory lapses and severe biographical fabrications) are collapsed within Tier 0. Furthermore, narrators on the boundary between *Saduq* and *Thiqah* (such as *Saduq yahim*, truthful but prone to error) may experience label noise.
* **Mitigation**: A secondary validation study was conducted on a consensus cohort of 180 narrators where Ibn Hajar and al-Dhahabi independently assigned identical categorical verdicts. All models demonstrated strong performance preservation on this consensus subset, confirming that the operationalized construct aligns with authoritative consensus.

### 2.2 Weakest-Link Isnād Aggregation Assumption
The chain-level evaluation (PO3) formalizes the classical principle that an isnad is only as strong as its weakest transmitter:
$$\text{Grade}(\text{Isnad}) = \min_{i=1}^L \text{Grade}(v_i)$$
Where:
* $\text{Grade}(\text{Isnad})$ represents the predicted status of the transmission chain.
* $L$ denotes the total number of transmitters in the isnad.
* $v_i$ represents the $i$-th individual narrator in the chain.

* **Threat**: In traditional jurisprudence, a chain containing a weak narrator may be elevated through corroborating routes (*mutaba'at* and *shawahid*), producing a *Hasan li-ghayrihi* (sound due to external corroboration) ruling. Evaluating isolated isnads using pure weakest-link aggregation ignores structural multi-chain corroboration.
* **Mitigation**: The scope of PO3 is explicitly designated as chain-level (*tariq*-level) transmission validity rather than global hadith text (*matn*) authenticity. The weakest-link model accurately operationalizes the formal condition of continuous reliable transmission for a single distinct chain.

### 2.3 Semantic Representation via Pre-trained Language Models
AraBERTv0.2 was selected as the foundational semantic representation:
* **Threat**: AraBERT was pre-trained predominantly on Modern Standard Arabic (MSA) corpora, including modern news, Wikipedia articles, and contemporary web text. Classical Arabic (*Fusha al-Turath*) employed in 2nd-4th century *jarh wa-ta'dil* treatises possesses specialized technical terminology, syntactic idioms, and condensed rhetorical expressions that diverge from modern semantic distributions.
* **Mitigation**: Mean pooling across contextual token embeddings was utilized to capture distributed phrase-level semantics. The empirical finding that TF-IDF outperformed frozen AraBERT embeddings is explicitly analyzed as a domain-mismatch phenomenon, illustrating that rare classical technical keywords carry greater discriminative weight than generic contextual embeddings.

---

## 3. External Validity

External validity addresses the generalizability of the findings across different hadith corpora, historical periods, and biographical reference sources.

### 3.1 Corpus Scope Conditioning
The empirical evaluation was conducted on narrator identities and transmission chains extracted from the canonical *Kutub al-Sittah* (the Six Canonical Hadith Books):
* **Threat**: Narrators included in the canonical collections represent a pre-filtered historical distribution characterized by high base reliability. Transmitters preserved by al-Bukhari and Muslim were predominantly accepted scholars, resulting in an inherently skewed distribution where 68.4% of the cohort belongs to Tier 2 (*Thiqah*).
* **Mitigation**: The attrition pipeline documented in `FYP2/attrition_table.md` transparently records the retention criteria. Findings are explicitly bounded to canonical Sunnite transmission networks rather than universal hadith literature.

### 3.2 Biographical Coverage and Regional Diversity
The multimodal graph cohort ($N = 335$) requires joint biographical records, textual evaluations, and transmission graph co-occurrences:
* **Threat**: Narrators with dense transmission ties and rich biographical citations are predominantly prominent scholars located in major historical learning centers (Medina, Mecca, Kufa, Basra, and Baghdad). Obscure regional narrators (*al-ruwat al-majhulun*) with isolated single transmissions may exhibit divergent network structures.
* **Mitigation**: Topological features (in-degree, out-degree, net-flow, and mean chain position) capture both central and peripheral transmission roles across the 335-node network.

---

## 4. Statistical Conclusion Validity

Statistical conclusion validity examines whether the quantitative conclusions regarding relationships between variables are justified based on statistical design, test assumptions, and measurement reliability.

### 4.1 Evaluation Protocol and Paired Statistical Testing
To prevent seed-dependent anomalies and fold selection bias:
* **Repeated Stratified Cross-Validation**: All benchmarks were executed using 5x5 repeated stratified cross-validation (5 distinct random seeds: 42, 123, 456, 789, 999; 5 stratified folds per seed), producing exactly 25 paired evaluation instances per model arm.
* **Identical Fold Partitions**: Every model within each experimental objective was evaluated on the exact same cross-validation splits.
* **Dual Statistical Testing**: Both parametric paired Student's t-tests and non-parametric Wilcoxon signed-rank tests were performed on all primary hypothesis comparisons. The consistency of p-values across parametric and non-parametric tests confirms that conclusions do not depend on distributional normality assumptions.

### 4.2 Class Imbalance and Metric Selection
The canonical cohort exhibits severe class imbalance (PO1: 558 Thiqah, 189 Saduq, 72 Below Saduq; PO2: 255 Thiqah, 60 Saduq, 20 Below Saduq):
* **Threat**: Standard classification accuracy is heavily biased toward the majority class. A trivial baseline predicting *Thiqah* for all instances achieves 76.12% accuracy in PO2 while demonstrating zero diagnostic utility for compromised transmitters.
* **Mitigation**: Macro-averaged F1-score (Macro-F1) was established as the primary evaluation metric across all objectives. Macro-F1 treats each tier with equal weighting, penalizing models that fail to identify minority classes. Furthermore, Quadratic Weighted Kappa ($\kappa_w$) was measured to incorporate the ordinal penalty structure of the target variable.

### 4.3 Multiple Comparison Control
In PO2, 13 distinct ablation arms were evaluated across multiple pairwise comparisons:
* **Threat**: Conducting multiple unadjusted pairwise hypothesis tests inflates Family-Wise Error Rate (FWER), increasing Type I error risk.
* **Mitigation**: Rather than performing all 78 possible pairwise comparisons, testing was restricted to six theoretically motivated *a priori* hypothesis contrasts defined prior to model execution. The primary empirical conclusions (e.g., TF-IDF superiority with $p = 3.49 \times 10^{-11}$; GCN degradation with $p = 0.0215$) satisfy Bonferroni-corrected significance thresholds ($\alpha = 0.05 / 6 \approx 0.0083$).
