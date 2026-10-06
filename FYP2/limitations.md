# Limitations of the Study

This document delineates the methodological, computational, and architectural limitations of this research. Transparent acknowledgment of research boundaries is essential for contextualizing the empirical findings, assessing theoretical claims, and guiding future investigations in computational hadith studies and applied graph machine learning.

---

## 1. Dependence on Ibn Hajar's Target Ground Truth

A primary limitation of this research concerns the operationalization of ground truth narrator reliability:
* **Subjectivity of Biographical Synthesis**: Target labels were derived from Ibn Hajar al-Asqalani's monumental 9th-century biographical compendium *Taqrib al-Tahdhib*. While *Taqrib* represents the most authoritative single-author synthesis of biographical evaluations in classical Sunnite scholarship, its verdicts reflect Ibn Hajar's individual harmonization of conflicting earlier opinions. In classical Islamic historiography, other distinguished synthesizers (such as al-Dhahabi in *al-Kashif* and *Mizan al-I'tidal*) occasionally arrived at differing conclusions.
* **Quantification of Consensus and Divergence**: In our verified cross-scholar analysis of 238 shared transmitters, Ibn Hajar and al-Dhahabi exhibited exact agreement on 180 scholars (75.63%) but diverged on 58 scholars (24.37%). Treating Ibn Hajar as the sole definitive ground truth necessarily imposes his specific interpretive framework upon the target distribution.
* **Loss of Granular Nuance via Three-Tier Binning**: Classical *jarh wa-ta'dil* methodology employs a twelve-level hierarchy of narrator credibility (ranging from superlative trust such as *Thiqah Thabt* down to accused fabricators such as *Wadda'*). To ensure viable statistical support across folds, this study compressed these twelve ranks into three discrete ordinal tiers (*Thiqah*, *Saduq*, *Below Saduq*). This compression collapses subtle theological and retentive gradations, treating marginal transmitters with mild memory irregularities identically to severely criticized transmitters.

---

## 2. Upstream Biographical Data Extraction and Entity Alignment

The construction of the empirical corpus relied on the extraction pipelines of the Hadith-Viz and SanadSet datasets:
* **Onomastic Ambiguity and Entity Disambiguation**: Arabic biographical nomenclature frequently features patronymics (*kunya*), lineage markers (*nasab*), and tribal affiliations (*nisbah*). Multiple historical narrators share identical common names (the classical challenge of *al-Mu'talif wa al-Mukhtalif* and *al-Mushtarak*). While rigorous ID mapping was applied, heuristic text alignment between biographical records and transmission chains may carry subtle misattribution risks.
* **Cohort Attrition and Data Sparsity**: The requirement for joint availability across biographical text, structured metadata, and multi-transmitter graph connections resulted in significant cohort attrition. From an initial pool of 4,905 raw narrator entries across the *Kutub al-Sittah*, only 819 satisfied strict classical text requirements, and only 335 possessed complete multimodal graph connectivity. Consequently, the multimodal findings are bounded to well-documented central transmitters rather than obscure or peripheral historical figures.

---

## 3. Transductive Formulation and Static Graph Assumption

The graph neural network architecture was formulated as a transductive node classification task:
* **Transductive Adjacency Availability**: The transmission graph structure (335 nodes, 177 undirected edges, 354 directed message-passing channels) was fixed globally. Although node labels were strictly partitioned into label-disjoint training (80%) and test (20%) folds during repeated cross-validation, the adjacency matrix remained accessible during feature propagation.
* **Absence of Inductive Zero-Shot Generalizability**: A transductive GCN cannot directly classify an unseen narrator who exists entirely outside the pre-computed adjacency matrix without reconstructing the graph topology. If a historical scholar possesses zero transmission links to the existing 335 nodes, the GCN cannot propagate information to that node. In contrast, the tabular early fusion model (P2-A) operates inductively on node features alone.
* **Static Graph Representation**: Historical transmission was modeled as an unweighted static graph. This abstraction neglects temporal succession dynamics, student-teacher lifespan overlaps, and geographical migration patterns between transmission centers.

---

## 4. Frozen Transformer Embeddings and Domain Semantic Mismatch

Textual representation in PO1 and PO2 utilized feature extraction via frozen pre-trained language models:
* **Absence of End-to-End Fine-Tuning**: Due to cohort size constraints ($N = 819$ in PO1; $N = 335$ in PO2) and GPU memory parameters, AraBERTv0.2 was deployed strictly in a frozen feature-extraction mode. Backpropagation was restricted to downstream classification heads (Logistic Regression and Multi-Layer Perceptrons) rather than fine-tuning AraBERT's 110 million contextual parameters.
* **Corpus Domain Disparity**: AraBERT was pre-trained predominantly on Modern Standard Arabic (news corpora, modern Wikipedia, and contemporary literature). In contrast, 2nd-4th century *jarh wa-ta'dil* treatises employ classical technical prose characterized by terse evaluative idioms (such as *la ba'sa bihi*, *yahim*, *salih al-hadith*). This lexical disparity explains why the domain-specific n-gram vocabulary of TF-IDF significantly outperformed generic contextual embeddings ($p = 3.49 \times 10^{-11}$).
* **Context Truncation**: Narrators with exceptionally extensive biographical records were truncated to fit within maximum token sequences (512 tokens), potentially discarding commentary located at the end of lengthy biographical entries.

---

## 5. Severe Class Imbalance in Canonical Hadith Corpora

The empirical distribution of narrator reliability exhibits substantial class skew:
* **Canonical Selection Bias**: The Six Canonical Hadith Compendia (*al-Kutub al-Sittah*) were compiled by master traditionists specifically to preserve authentic and acceptable traditions. Consequently, the vast majority of transmitters preserved in these collections are reliable scholars. In our canonical PO1 cohort, 68.13% (558 narrators) belong to Tier 2 (*Thiqah*), 23.08% (189 narrators) belong to Tier 1 (*Saduq*), and 8.79% (72 narrators) belong to Tier 0 (*Below Saduq*). In the multimodal PO2 cohort, Tier 2 reaches 76.12% (255 narrators), while Tier 0 comprises only 5.97% (20 narrators).
* **Minority Class Variance**: In an 80/20 cross-validation split of the 335-node cohort, a test partition contains on average only 4 instances of Tier 0 (*Below Saduq*). A single misclassification in the minority class shifts the fold-level minority F1 score by 25 percentage points, contributing to higher standard deviations in Macro-F1 across folds.
* **Weighting Trade-Offs**: Inverse-frequency class weighting was implemented to mitigate majority-class dominance. While this penalization restored balanced diagnostic sensitivity, it occasionally forced models to misclassify borderline *Thiqah* transmitters into *Saduq*, depressing raw classification accuracy relative to the majority baseline.

---

## 6. Sequential Chain Sampling and Multi-Path Aggregation in PO3

The chain-level validation in PO3 was subjected to specific sampling and structural boundaries:
* **Sequential Qualifying Sample**: The 1,578 chains evaluated in PO3 were drawn from a total population of 16,754 candidate multi-narrator isnads in the corpus. Chains were selected sequentially based on the criterion of containing multiple verified narrators from the connected multimodal core ($N = 335$). While this sample substantially exceeds the institutional statistical validity threshold ($\ge 1,200$ chains) and achieves statistical significance ($p = 4.29 \times 10^{-58}$), it does not represent an unconditioned uniform random sample of the full 80,731-hadith corpus.
* **Incomplete Chain Node Coverage**: Within the 1,578 evaluated chains, the mean number of multimodal cohort narrators per chain was 2.17. Transmitters in the chain who fell outside the 335-node multimodal core were masked or excluded from the model's node-level aggregation. While the weakest-link operator was evaluated across all covered nodes, unresolved intermediate links introduce potential variance into whole-chain predictions.
* **Single-Path Limitation**: PO3 evaluated individual transmission chains (*turuq*) in isolation. In traditional hadith jurisprudence, the ultimate ruling on a prophetic tradition is established through the synthesis of multiple parallel chains (*jam' al-turuq*). A hadith with a weak individual isnad may achieve the status of *Hasan li-ghayrihi* if supported by independent corroborating chains. The current framework evaluates chain-level integrity rather than cross-chain multi-path corroboration.
