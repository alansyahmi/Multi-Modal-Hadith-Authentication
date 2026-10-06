# Chapter 2: Literature Review

## 2.1 The Tradition of ʿIlm ar-Rijāl and Biographical Criticism

Hadith authentication relies fundamentally on the critical evaluation of the chain of transmission (*isnād*). The historical development of this discipline produced a specialized branch of scholarship known as *ʿIlm ar-Rijāl* (the science of biographical evaluation) and *ʿIlm al-Jarḥ wa al-Taʿdīl* (the science of narrator criticism and accreditation).

### 2.1.1 Historical Evolution of Evaluative Dictionaries
The critical appraisal of narrators originated as an oral practice among early scholars of the 2nd century AH. Scholars such as Shuʿbah ibn al-Ḥajjāj (d. 160 AH) and Sufyān al-Thawrī (d. 161 AH) systematically scrutinized the veracity and memory of transmitters. Over successive generations, this oral tradition expanded into massive encyclopedic compendia:
* The formative period (2nd to 3rd century AH): Scholars such as Yaḥyā ibn Maʿīn (d. 233 AH), Aḥmad ibn Ḥanbal (d. 241 AH), and ʿAlī ibn al-Madīnī (d. 234 AH) compiled initial treatises documenting biographical details and personal appraisals.
* The systematization period (3rd to 4th century AH): Muḥammad ibn Ismāʿīl al-Bukhārī (d. 256 AH) compiled *al-Tārīkh al-Kabīr*, and Ibn Abī Ḥātim al-Rāzī (d. 327 AH) authored *al-Jarḥ wa al-Taʿdīl*, introducing early multi-tiered taxonomic hierarchies.
* The encyclopedic synthesis period (7th to 9th century AH): Al-Mizzī (d. 742 AH) assembled *Tahdhīb al-Kamāl fī Asmāʾ al-Rijāl*, which unified biographical narratives across all six canonical collections (*al-Kutub al-Sittah*). Al-Dhahabī (d. 748 AH) condensed these records in *Mīzān al-Iʿtidāl* and *al-Kāshif*. Finally, Ibn Ḥajar al-ʿAsqalānī (d. 852 AH) produced *Tahdhīb al-Tahdhīb* and its celebrated condensation, *Taqrīb al-Tahdhīb*.

### 2.1.2 Dual Pillars of Reliability: ʿAdālah and Ḍabṭ
In classical hadith methodology, narrator reliability is not a generic binary label; it is a composite evaluation defined by two independent criteria:
* Uprightness (*ʿAdālah*): The narrator must exhibit personal religious integrity, truthfulness, and emotional maturity. Transgressions against uprightness include intentional lying (*kidhb*), severe religious innovation (*bidʿah*), or general moral dissolution (*fīsq*). Narrators who fail the integrity threshold are categorized as *matrūk* (abandoned) or *kadhdhāb* (fabricator).
* Precision and Retention (*Ḍabṭ*): The narrator must demonstrate accurate mnemonic recall or rigorous manuscript custody. Precision is subdivided into memory retention (*ḍabṭ al-ṣadr*) and written fidelity (*ḍabṭ al-kitāb*). Failures in precision include frequent negligence (*ghaflah*), severe divergence from peer narrators (*fuhsh al-ghalaṭ*), or memory deterioration due to senility (*ikhtilāṭ*).

### 2.1.3 Hierarchical Categorization Schemes
Ibn Ḥajar al-ʿAsqalānī formalized a twelve-tier evaluative taxonomy in the introduction to *Taqrīb al-Tahdhīb*. In modern computational modeling, these historical grades are systematically mapped into standardized ordinal reliability classes:
* Tier 2 (*Thiqah* / Trustworthy): Encompasses Ibn Ḥajar's top grades, including *Ṣaḥābī* (Companion, excluded from ML modeling), *Thiqah Thiqah* (eminently trustworthy), and *Thiqah* (trustworthy and precise).
* Tier 1 (*Ṣadūq* / Truthful): Corresponds to narrators who possess unquestioned moral uprightness but exhibit minor or occasional memory shortcomings (for example, *ṣadūq yahim* or *ṣadūq lahu awhām*). Narrations transmitted by *Ṣadūq* narrators form the backbone of good (*ḥasan*) hadiths.
* Tier 0 (*Below Ṣadūq* / Unreliable or Qualified): Encompasses narrators with unverified identity (*majhūl*), weak memory (*ḍaʿīf*), extreme unreliability (*matrūk*), or verified falsehood (*kadhdhāb*).

---

## 2.2 Classical Technical Vocabulary vs. General Sentiment Analysis

Biographical evaluation texts (*kutub al-rijāl*) cannot be processed using standard sentiment analysis lexicons. The language of *Jarḥ wa Taʿdīl* functions as a rigorous, domain-specific technical vocabulary:

### 2.2.1 Structural Failure of Generic Sentiment Dictionaries
Generic Arabic sentiment lexicons evaluate words based on emotional polarity (positive versus negative). Applying such lexicons to biographical text causes severe misclassification:
* Domain-specific polarity inversion: In general Arabic literature, terms such as *fāḍil* (virtuous), *zāhid* (ascetic), or *rajul ṣāliḥ* (pious man) carry strongly positive sentiment. In classical *rijāl* criticism, however, these phrases denote personal piety without attesting to mnemonic precision (*ḍabṭ*). Early critics frequently remarked: "He was a righteous man, but he was careless in recording hadith." A naive sentiment classifier classifies such biographies as highly reliable, whereas classical methodology treats the narrator as weak in transmission.
* Complex contextual qualifiers: Critics routinely compound positive and negative technical terms. Phrases such as *ṣadūq illā annahu yukhṭiʾ* (truthful, except that he makes mistakes) or *lā baʾsa bihi fī ghayr ḥadīth al-Zuhrī* (acceptable, except when transmitting from al-Zuhrī) require deep semantic parsing of syntactic modifiers rather than keyword aggregation.
* Scalar technical nuance: Technical descriptors form a delicate semantic gradient. For example, *thiqah* (trustworthy) represents established accreditation; *ṣadūq* (truthful) denotes acceptable uprightness with average precision; *maqbūl* (acceptable only if corroborated) denotes marginal utility; and *ḍaʿīf* (weak) denotes active unreliability.

---

## 2.3 Computational Approaches to Hadith and Isnād Analysis

Over the past decade, computer science researchers have introduced computational tools to analyze hadith corpora. Table 2.1 summarizes the major recent studies in this domain, comparing their target tasks, architectures, datasets, evaluation metrics, and methodological limitations.

### Table 2.1: Critical Comparative Benchmark of Prior Computational Literature

| Study | Primary Method and Architecture | Target Task | Dataset and Scale | Primary Metric | Reported Performance | Methodological Limitations Relative to This Work |
|---|---|---|---|---|---|---|
| **Mahmoud et al. (2022, 2024)** | Graph query embedding + AraBERT re-ranking over *NarratorsKG* | Narrator entity disambiguation in transmission chains | AR-Sanad 280K synthetic chains | Top-1 Accuracy | 94.6% Accuracy | Focused strictly on entity identification rather than reliability quantification; evaluated on synthetic chains rather than canonical historical collections. |
| **Mghari et al. (2024)** (*Narrator2Vec*) | Skip-Gram and CBOW embeddings on narrator sequence tokens (100 dimensions) | Unsupervised narrator representation and missing link prediction | SanadSet (over 650,000 chains) | Cosine similarity | Captures transmission co-occurrence | Operates purely on syntactic chain co-occurrence; completely ignores classical biographical evaluation texts and cannot output reliability scores. |
| **Ahmad et al. (2025)** | Sequence tagging (HMM/POS) + supervised classifiers on flattened isnāds | Chain continuity flagging and direct binary hadith grading (*Ṣaḥīḥ* vs *Ḍaʿīf*) | Canonical *Kutub al-Sittah* sample | Accuracy, Precision, Recall | 75% to 82% Accuracy | Bypasses individual narrator modeling; relies on dataset-wide grade correlations, creating a black-box model unable to explain why an isnād fails. |
| **Ghriss (2025)** | Bipartite graph co-update algorithm modeling multi-path trust propagation | Information authentication and collective trust propagation | Curated mathematical toy sub-graphs | Ranking correlation | Theoretical convergence proven | Formulates an algorithmic proof-of-concept on synthetic graphs; does not parse real classical Arabic biographical texts. |
| **This Study (Alan Syahmi, 2026)** | Fused multimodal framework combining frozen AraBERT text embeddings with transmission graph features | Narrator reliability quantification (PO1, PO2) and downstream chain error propagation (PO3) | Audited cohort from Hadith-Viz and SanadSet ($N=844$ text, $N=335$ graph, $N=1,578$ chains) | **Macro-F1**, Quadratic $\kappa$, Paired t-tests | Locked Macro-F1 across 25 paired cross-validation folds | Eliminates target data leakage; establishes rigorous zero-overlap identity splits; quantifies graph heterophily. |

---

## 2.4 Pretrained Arabic Language Models

Natural language processing in Arabic has advanced substantially from static word representations (such as Word2Vec and FastText) to deep contextual transformers.

### 2.4.1 Transformer Architectures in Arabic NLP
Contextual transformers (Vaswani et al., 2017) resolve polysemy by calculating self-attention across an entire sentence. In Arabic NLP, models must address rich morphological inflections, diacritical ambiguity, and prefix/suffix agglutination:
* **AraBERT (Antoun et al., 2020):** Built on the BERT-Base architecture (12 transformer layers, 768 hidden dimensions, 12 attention heads). AraBERT v2 utilizes the Farasa morphological segmenter to separate prefixes and suffixes prior to subword tokenization, substantially improving vocabulary efficiency on Arabic text.
* **CAMeLBERT (Inoue et al., 2021):** Pretrained on diverse Arabic language variants, including Classical Arabic (CA), Modern Standard Arabic (MSA), and dialectal corpora, providing specialized weights for historical literature.

### 2.4.2 Challenges in Modeling Classical Arabic Texts
While modern Arabic transformers demonstrate high capability on contemporary news and social media benchmarks, classical biographical texts introduce distinct domain challenges:
* Archaic syntactic structures: Classical Arabic constructs sentences with extensive subordination and parenthetical honorific formulas (*tarāḥum* and *taʿẓīm*).
* Lexical drift: Words common in classical scholarship possess substantially different semantics in Modern Standard Arabic.
* Out-of-vocabulary technical terms: Rare classical descriptor compounds frequently decompose into fragmented subword pieces, diluting semantic representations unless handled by domain-aware tokenizers.

---

## 2.5 Graph Neural Networks and the Graph Heterophily Challenge

Graph Neural Networks (GNNs) extend deep learning to non-Euclidean graph topologies by updating node representations through neighborhood aggregation (message passing).

### 2.5.1 Foundations of Graph Representation Learning
Let $G = (V, E)$ represent a graph where $V$ denotes the set of $N$ nodes (narrators) and $E$ represents the set of edges (transmission connections). A standard Graph Convolutional Network (GCN) layer updates the node embedding matrix $H^{(l)}$ at layer $l$ using the following localized first-order spectral convolution:

$$H^{(l+1)} = \sigma \left( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)} \right)$$

The variables within this mathematical formulation are defined as follows:
* $\tilde{A} = A + I_N$ represents the adjacency matrix with added self-loops.
* $\tilde{D}$ represents the diagonal degree matrix where $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$.
* $W^{(l)}$ represents the trainable layer weight transformation matrix.
* $\sigma$ represents an element-wise non-linear activation function (such as ReLU).
* $H^{(l)}$ represents the node feature matrix at layer $l$, where $H^{(0)} = X$.

### 2.5.2 The Homophily Assumption and Its Limits
Standard spatial and spectral GNN architectures operate under the fundamental assumption of strong graph homophily (McPherson et al., 2001). Under homophily, connected nodes are expected to share identical or highly similar labels:

$$h = \frac{|(u, v) \in E : y_u = y_v|}{|E|}$$

When edge homophily $h$ is high ($h \to 1$), neighborhood message passing acts as a beneficial low-pass filter, smoothing feature representations and enhancing classification accuracy (Kipf and Welling, 2017).

### 2.5.3 The Phenomenon of Graph Heterophily in Transmission Networks
When a network exhibits strong heterophily ($h \ll 0.5$), connected nodes predominantly belong to different classes. Recent literature in graph machine learning (Zhu et al., 2020; Lim et al., 2021) has proven that standard GNN message passing on heterophilous networks causes over-smoothing, mixing distinct class representations into uniform distributions and actively degrading classification performance.

In hadith transmission networks, structural heterophily is a natural consequence of historical transmission dynamics:
* Eminent master scholars (*shuyūkh*, such as Sufyān al-Thawrī or Imām Mālik) delivered public lectures attended by hundreds of students across decades. These students ranged from exceptionally precise scholars to careless compilers.
* Diligent students of hadith (*ṭullāb al-ʿilm*) traveled across Islamic provinces (*riḥlah fī ṭalab al-ḥadīth*), deliberately collecting narrations from local community elders and obscure informants to record all regional variations.
* Consequently, a transmission edge denotes an educational or chronological encounter rather than a shared degree of mnemonic retention. Aggregating the features of adjacent nodes in a transmission network averages reliable and unreliable signals together, undermining individual reliability quantification.

---

## 2.6 Summary and Identification of Research Gaps

This literature review isolates three distinct research gaps in existing computational hadith studies:
1. **The Semantic-Structural Separation:** Prior works either analyze transmission graph topology without processing classical evaluative texts (Mghari et al., 2024; Ghriss, 2025) or perform entity resolution without evaluating reliability (Mahmoud et al., 2024; Mosa, 2025). No audited framework has systematically fused deep Arabic text embeddings with transmission graph topology to quantify reliability.
2. **Methodological Vulnerability and Target Data Leakage:** Prior predictive attempts frequently allowed 9th-century synthesis judgments to leak into input features, producing artificially inflated metrics. Furthermore, evaluations frequently relied on raw Accuracy despite extreme class imbalances.
3. **Unexamined Graph Homophily Assumptions:** Prior proposals to apply GNNs to transmission chains assumed without empirical verification that transmission topology improves prediction. No prior work has conducted a controlled statistical ablation to measure whether GNN message passing aids or harms narrator classification.
