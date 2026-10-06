# Literature Comparison & Research Gap Analysis

**Project Title:** A Multi-Modal Framework for Quantifying Narrator Reliability in ʿIlm ar-Rijal Using Semantic Sentiment Analysis and Graph Neural Networks  
**Author:** Alan Syahmi bin Sanih @ Sani (BI23110344)  
**Supervisor:** Dr. Khalifa Chekima  
**Target Chapter:** Chapter 2 (Literature Review) & Chapter 5 (Comparative Discussion)  
**Purpose:** Formally benchmark this project against existing state-of-the-art computational Hadith and *rijāl* literature, directly resolving examiner feedback from Dr. Lai.

---

## 1. Literature Comparison Matrix

| Study | Primary Method / Architecture | Target Task | Dataset & Scale | Evaluation Metric | Reported Performance | Fundamental Limitations & Differences with This Work |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mahmoud et al. (2022, 2024)**<br>*(Neural Computing & Applications)* | Two-stage retrieval: Graph query embedding + AraBERT re-ranker over `NarratorsKG` | Narrator Disambiguation / Entity Identification in transmission chains | AR-Sanad 280K & AR-Sanad 280K-v2 (Synthetic isnāds) | Top-1 Accuracy | **94.6%** Accuracy | **Task mismatch:** Solves *entity disambiguation* (identifying *who* a narrator is), not *reliability quantification* (how trustworthy they are). Evaluated on synthetic chains rather than multi-source canonical collections. |
| **Mghari et al. (2024)**<br>*Narrator2Vec*<br>*(Arabian J. Sci. Eng.)* | Word2Vec-inspired Skip-Gram & CBOW on narrator sequence tokens (100-dim) | Distributed Narrator Representation & Missing Narrator Prediction | SanadSet (>650,000 Hadith chains) | Cosine similarity & sequence recovery | Captures name relatedness; detects chain co-occurrence | **Lack of semantic & evaluative depth:** Operates purely on syntactic co-occurrence in chains; completely ignores classical *Jarḥ wa Taʿdīl* evaluative texts, biographical metadata, and GNN message-passing. Cannot output reliability scores. |
| **Mosa (2024/2025)**<br>*(Knowledge-Based Systems / Benchmark)* | Hybrid Knowledge Graph topology + Transformer candidate scoring | Narrator Disambiguation in complex isnād networks | AR-Sanad 280K-v2 benchmark | Top-1 Accuracy | **97.8%** Accuracy | **Disambiguation focus:** Establishes SOTA on narrator entity resolution on synthetic chains, but does not quantify narrator trustworthiness or validate downstream chain authenticity. |
| **Ahmad et al. (2025)**<br>*(AL-JAMEI Research Journal)* | Sequence tagging (HMM/POS) + supervised classifiers on flattened isnāds | Chain continuity flagging & Direct Hadith Grading (*Ṣaḥīḥ* vs. *Ḍaʿīf*) | Canonical Kutub as-Sittah sample (~several thousand chains) | Precision, Recall, Accuracy | High recall on candidate linking; **75%–82%** accuracy on Hadith grading | **Black-box shortcut:** Bypasses deep individual narrator reliability modeling; jumps straight from raw chain sequences to Hadith grades, making the model vulnerable to dataset grade bias and unable to explain *why* a chain fails. |
| **Ghriss (2025)**<br>*(Muslims in ML / ICML 2025)* | Bipartite graph co-update algorithm modeling multi-path corroboration | Theoretical Information Authentication & Collective Trust Propagation | Curated mathematical toy sub-graphs from Hadith tradition | Convergence rate & ranking correlation | Proven mathematical convergence of trust propagation | **Theoretical / Synthetic constraint:** Algorithmic proof-of-concept; does not parse real classical Arabic biographical corpora (*Taqrīb / Tahdhīb*), and does not leverage deep neural language models (e.g. AraBERT). |
| **This Work (FYP2)**<br>*(Alan Syahmi, 2026)* | **Fused Multi-Modal Framework:**<br>1. Fine-tuned AraBERT v2 on classical *Jarḥ wa Taʿdīl* texts.<br>2. Graph Neural Network (GCN/GAT) over SanadSet transmission topology.<br>3. Chronological generation metadata (*Tabaqa*, death info, net transmission flow).<br>4. Zero target leakage architecture. | **Narrator Reliability Quantification (PO1 & PO2)** and **Downstream Isnād Validation (PO3)** | **SanadSet (650K corpus)** + **3,797 classical biographical profiles** (*Taqrīb at-Tahdhīb* & *Tahdhīb al-Kamāl*) | **Macro-F1 (Primary)**, Accuracy, Cohen’s Kappa, Confusion Matrix | **Benchmark in progress:**<br>• Zero-leakage baseline locked<br>• Macro-F1 evaluated across 5 ordinal classes (*Matrūk, Ḍaʿīf, Maqbūl, Ṣadūq, Thiqah*) | **Key Novelty:** First framework to bridge deep classical biographical Arabic NLP with GNN transmission topology to produce a calibrated, continuous reliability index ($0.0 \dots 1.0$) with zero target leakage. |

---

## 2. In-Depth Scholarly Synthesis of Prior Work

### 2.1 The Narrator Disambiguation Paradigm: Mahmoud et al. & Mosa
A prominent direction in computational Hadith literature focuses on **Narrator Disambiguation** (Entity Resolution). In classical Arabic texts, narrators are frequently cited using partial names, patronymics (*ibn X*), *kunya* (*Abū Y*), or *nisba* (*al-Zuhrī*). 
* **Mahmoud et al. (2022, 2024)** addressed this by creating the **AR-Sanad 280K** dataset, synthesizing artificial isnād variants, and building **NarratorsKG**. They combined graph query embeddings with AraBERT to achieve 94.6% disambiguation accuracy.
* **Mosa (2024/2025)** extended this with a hybrid Knowledge Graph-Transformer model, advancing state-of-the-art accuracy on AR-Sanad 280K-v2 to 97.8%.
* **Distinction & Gap:** While Mahmoud et al. and Mosa solve *who* the narrator is, they do not assess *how reliable* the narrator is. Their models stop at identity resolution. In contrast, our framework takes resolved identities and tackles the higher-order task of **quantifying trustworthiness** based on historical biographical criticism (*ʿIlm ar-Rijāl*).

### 2.2 Unsupervised Sequence Embeddings: Narrator2Vec (Mghari et al., 2024)
* **Mghari et al. (2024)** introduced **Narrator2Vec**, training continuous vector representations (100 dimensions) of narrators based on co-occurrence in over 650,000 SanadSet chains using Word2Vec principles.
* **Distinction & Gap:** Narrator2Vec operates purely at the sequence-token level. While it successfully captures transmission co-occurrence (e.g., that Imām Mālik frequently transmits to Nāfiʿ), it has **no semantic understanding of biographical criticism**. A narrator who frequently co-occurs with trustworthy scholars but was condemned as a fabricator (*kadhdhāb*) by critics cannot be distinguished from a trustworthy narrator by sequence co-occurrence alone. Our framework directly addresses this by incorporating **AraBERT v2 fine-tuned on classical *Jarḥ wa Taʿdīl* judgments**.

### 2.3 Direct Hadith Grading vs. Narrator-Level Modeling: Ahmad et al. (2025)
* **Ahmad et al. (2025)** investigated the feasibility of automatic isnād grading, training machine learning classifiers directly on isnāds from canonical collections to output binary (*Ṣaḥīḥ* vs. *Ḍaʿīf*) labels.
* **Distinction & Gap:** Jumping directly from chain strings to Hadith authenticity verdicts introduces a severe **black-box flaw**. In traditional Hadith methodology, a Hadith is not authentic simply because of its overall text length or token pattern; it is authentic because **each transmitter in the chain possesses integrity (*ʿadālah*) and precision (*ḍabṭ*)**, and the chain is unbroken (*ittiṣāl*). By skipping individual narrator modeling, direct grading models memorize dataset-specific correlations. Our framework adheres to Islamic epistemic principles by **first modeling narrator reliability individually (PO1/PO2)**, and only then aggregating scores downstream to evaluate chain consistency (PO3).

### 2.4 Theoretical Trust Propagation: Ghriss (2025)
* **Ghriss (2025)** presented an elegant mathematical formulation of crowdsourced information authentication inspired by Hadith science at the ICML 2025 Muslims in ML workshop. The work formulated an iterative co-update algorithm for transmission paths, corroboration, and trust scoring.
* **Distinction & Gap:** Ghriss's model is primarily theoretical and evaluated on synthetic/toy graphs. It does not interface with large-scale, messy, real-world classical Arabic biographical dictionaries. Our work operationalizes these conceptual principles on **3,797 parsed biographical entries** from *Taqrīb at-Tahdhīb* and *Tahdhīb al-Kamāl* and **3,541 real transmission edges** from SanadSet.

---

## 3. The Research Gap Addressed by This Project

Across the existing literature, three major research gaps emerge:
1. **The Semantic-Structural Divide:** Existing works either model graph structure without biographical text (Narrator2Vec, Ghriss) or perform entity matching without reliability scoring (Mahmoud et al., Mosa). No prior study fuses **deep biographical Arabic NLP** with **Graph Neural Network transmission topology** to quantify reliability.
2. **Target Leakage & Methodological Fragility:** Early computational attempts often suffered from data leakage (using target grades as input features) or evaluated models on raw Accuracy despite severe class imbalance (>80% authentic/truthful narrators in canonical corpora). This project establishes a **zero-leakage architecture** with fixed **Macro-F1** evaluation.
3. **Epistemic Fidelity:** Machine learning projects often claim to "grade Hadith automatically," ignoring matn defects (*ʿillah*) and irregularities (*shudhūdh*). Our project maintains strict scholarly guardrails: **narrator reliability quantification is the primary objective**, while chain-level evaluation is strictly a secondary downstream validation of narrator estimates against reference classifications.

---

## 4. Citation Key for Thesis Bibliography

```bibtex
@article{mahmoud2024narrator,
  title={Narrator identification by querying Sanad graph and utilizing the NarratorsKG on AR-Sanad 280K-v2 dataset},
  author={Mahmoud, Somaia and Nabil, Emad and Saif, Omar and Torki, Marwan},
  journal={Neural Computing and Applications},
  volume={36},
  pages={23169--23180},
  year={2024},
  publisher={Springer}
}

@article{mghari2024narrator2vec,
  title={Narrator2Vec: An Efficient Narrator Representation in Hadith Literature Using Word Embedding},
  author={Mghari, Mohammed and Bouras, Omar and El Hibaoui, Abdelaaziz},
  journal={Arabian Journal for Science and Engineering},
  volume={49},
  pages={4479--4494},
  year={2024},
  publisher={Springer}
}

@article{mosa2024synergizing,
  title={Synergizing structure and semantics: a knowledge graph-transformer framework for narrator disambiguation in hadith networks},
  author={Mosa, Mohamed Atef},
  journal={Journal of King Saud University - Computer and Information Sciences},
  year={2024}
}

@article{ahmad2025hadith,
  title={Hadith Grading \& Machine Learning: Feasibility of Automatic Isn{\=a}d-Analysis},
  author={Ahmad, Nisar and Ahmed, Shahbaz and Abdullah, Muhammad},
  journal={AL-JAMEI Research Journal},
  volume={3},
  number={2},
  pages={202--215},
  year={2025}
}

@inproceedings{ghriss2025crowdsourced,
  title={Crowdsourced Information Authentication: A Graph-based Model from the Science of Hadith},
  author={Ghriss, Amine},
  booktitle={4th Muslims in ML Workshop Co-Located with ICML 2025},
  year={2025}
}
```
