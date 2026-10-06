# Chapter 3: Project Methodology

## 3.1 Overall Methodological Framework and Dual-Cohort Architecture

To address the research questions and objectives established in Chapter 1, this study establishes a multi-scale, leakage-free methodological framework. Figure 3.1 outlines the sequential progression from raw corpus ingestion to downstream chain evaluation.

```
+-----------------------------------------------------------------------------------+
|                           Stage 1: Raw Data Ingestion                             |
|  - SanadSet Corpus (650,000 raw isnāds)                                           |
|  - Hadith-Viz Biographical Encyclopedia (3,047 scholar cards)                     |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                     Stage 2: Five-Stage Data Hygiene Funnel                       |
|  - Remove unrated stubs and unverified entities                                   |
|  - Exclude Companions (Ṣaḥābah) to prevent theological majority bias               |
|  - Apply zero-leakage author filtering (strip Ibn Ḥajar and al-Dhahabī text)      |
|  - Deduplicate entities into 1 identity = 1 graph node                            |
+-----------------------------------------------------------------------------------+
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
+---------------------------------------+ +---------------------------------------+
|  Level 1: Full Biographical Text      | |  Level 2: Multimodal Graph Cohort     |
|  Cohort (N = 844)                     | |  (N = 335 nodes, 354 undirected edges)|
|  - Task: PO1 Text Benchmark           | |  - Task: PO2 Controlled 10-Arm        |
|  - Models: TF-IDF vs AraBERT          | |    Ablation Study                     |
|  - Target: Ibn Ḥajar 3-tier grade     | |  - Modalities: Text, Graph, Metadata  |
+---------------------------------------+ +---------------------------------------+
                                                              │
                                                              ▼
+-----------------------------------------------------------------------------------+
|           Level 3: Downstream Isnād Chain Validation (N = 1,578 chains)           |
|  - Task: PO3 Weakest-Link Error Propagation                                       |
|  - Formulation: S_chain = min_i (y_i)                                             |
|  - Evaluation: Chain Tier Accuracy, Binary Acceptance, Spearman Rank Correlation   |
+-----------------------------------------------------------------------------------+
```

The experimental architecture is organized into three complementary evaluation levels:
* **Level 1 (Full Biographical Text Cohort, $N=844$):** Evaluates linguistic and semantic text classification models across all non-Companion historical scholars possessing verified *Taqrīb* grades and earlier-critic statements.
* **Level 2 (Multimodal Graph Cohort, $N=335$):** Evaluates text, graph, and metadata fusion models on the subset of narrators who actively participate in the canonical transmission network.
* **Level 3 (Downstream Isnād Chain Cohort, $N=1,578$):** Evaluates chain-level authenticity preservation by propagating node-level model predictions through historical transmission paths.

---

## 3.2 Data Ingestion and Source Auditing

The experimental pipeline integrates two primary classical hadith data sources:

### 3.2.1 The SanadSet Transmission Corpus
SanadSet (Mghari et al., 2024) is a comprehensive dataset containing over 650,000 raw isnād chains extracted from classical hadith literature. For this study, the transmission corpus is bounded to the 30,712 chains representing the six canonical collections (*al-Kutub al-Sittah*):
* *Ṣaḥīḥ al-Bukhārī*
* *Ṣaḥīḥ Muslim*
* *Sunan Abī Dāwūd*
* *Jāmiʿ al-Tirmidhī*
* *Sunan al-Nasāʾī*
* *Sunan Ibn Mājah*

### 3.2.2 The Hadith-Viz Biographical Encyclopedia
Biographical narratives, historical metadata, and evaluative opinions were retrieved from the curated Hadith-Viz biographical encyclopedia (`Dataset/hadith_viz/shamela_rijal2.json`). This resource provides structured records for 3,047 classical scholars, containing teacher-student associations, geographical locations, death dates, generation classifications (*ṭabaqāt*), and itemized lists of evaluative statements (*jarḥ wa taʿdīl*).

### 3.2.3 Forensic Audit of Scraped Primary Texts
Prior exploratory workflows attempted to extract biographical records directly from raw digitized volumes of *Tahdhīb al-Tahdhīb* via regular expressions (`output/shamela/`). A forensic audit revealed unacceptable data corruption in the scraped output:
* Text segmentation failure: 51.9% of extracted biographical blocks contained non-person narrative fragments, poetic citations, or merged adjacent biographies.
* Grade assignment errors: Direct regex parsing produced a 53.9% error rate when mapping free-form textual summaries into discrete reliability classes.

Consequently, raw web scraping was abandoned in favor of the curated, structured Hadith-Viz corpus, where scholar boundaries and teacher-student transmission links are verified.

---

## 3.3 The Five-Stage Data Hygiene Funnel and Attrition Analysis

To prevent data leakage, eliminate duplicate identity artifacts, and guarantee reproducible boundaries, raw data records were processed through the 5-stage data hygiene funnel detailed in Table 3.1.

### Table 3.1: Complete Data Attrition Funnel Across Pipeline Stages

| Funnel Stage | Entity / Record Count | Retention Rate | Scientific and Methodological Rationale |
|---|---:|---:|:---|
| **Raw SanadSet Names** | 262,497 tokens | 100.0% | Unfiltered narrator name strings across 650,000 raw isnād chains. |
| **Unique Normalized Names** | 167,047 names | 63.6% | Standardized Arabic orthography by collapsing hamza forms, alif maqṣūrah, and tāʾ marbūṭah. |
| **Hadith-Viz Master Cards** | 3,047 cards | 1.8% | Curated biographical index corresponding to canonical *Kutub al-Sittah* transmitters. |
| **Stage 1: Exclude Unrated Stubs** | 942 cards | 30.9% | Removed 2,105 placeholder cards lacking verified Ibn Ḥajar synthesis grades. |
| **Stage 2: Exclude Companions** | 860 scholars | 91.3% | Removed 82 Companions (*Ṣaḥābah*) due to the theological constant of universal uprightness. |
| **Stage 3: Exclude Empty Text** | **844 scholars** | 98.1% | Removed 16 entries lacking earlier-critic commentary. **Forms the PO1 Text Benchmark Cohort.** |
| **Stage 4: Graph Intersection** | 438 graph nodes | 51.9% | Isolated scholars actively appearing as transmitters within canonical isnād chains. |
| **Stage 5: Identity Deduplication** | **335 identities** | 76.5% | Merged multi-token spelling variants into 1 unique identity = 1 graph node. **Forms the PO2 Multimodal Cohort.** |

The funnel reduced the initial noisy collection of names into two rigorous, frozen evaluation sets:
1. The **Level 1 Text Cohort ($N=844$)**: Contains 579 *Thiqah* (68.6%), 194 *Ṣadūq* (23.0%), and 71 *Below Ṣadūq* (8.4%) scholars.
2. The **Level 2 Multimodal Cohort ($N=335$)**: Contains 258 *Thiqah* (77.0%), 64 *Ṣadūq* (19.1%), and 13 *Below Ṣadūq* (3.9%) scholars, connected by 354 verified transmission edges.

---

## 3.4 Structurally Enforced Zero Target Leakage Protocol

Data leakage represents the most significant threat to validity in applied machine learning. In preliminary investigations, inspection of the raw `jarh_tadil` quote arrays inside `Dataset/hadith_viz/shamela_rijal2.json` revealed severe target leakage:
* Duplicated target verdicts: Ibn Ḥajar's canonical *Taqrīb* grade was duplicated verbatim inside the biographical quote array across 937 cards, totaling 1,659 explicit quotes.
* Trivial memorization: Machine learning models trained on raw quote arrays achieved near-perfect accuracy by simply identifying substrings such as `قال ابن حجر: ثقة` (Ibn Ḥajar said: trustworthy).

### 3.4.1 Algorithmic Author Filtering Rule
To permanently resolve this vulnerability, the data preparation module (`scripts/build_clean_dataset.py`) implemented an algorithmic author filter:

```python
def extract_earlier_critic_text(card):
    snippets = []
    for item in card.get("jarh_tadil", []):
        scholar = item.get("scholar", "").strip()
        # Strictly exclude 9th-century and 8th-century synthesis judgments
        if any(target in scholar for target in ["ابن حجر", "الذهبي"]):
            continue
        for evaluation in item.get("evaluations", []):
            text = evaluation.get("text", "").strip()
            if text and len(text) > 5:
                snippets.append(f"قال {scholar}: {text}")
    return "\n".join(snippets)
```

The operational characteristics of this filtering rule are summarized as follows:
* Excluded authorities: Statements attributed to Ibn Ḥajar al-ʿAsqalānī (d. 852 AH) and al-Dhahabī (d. 748 AH) are 100% purged from input features.
* Retained authorities: Statements from earlier classical critics who lived between the 2nd and 4th centuries AH (such as Yaḥyā ibn Maʿīn, Aḥmad ibn Ḥanbal, ʿAlī ibn al-Madīnī, Abū Ḥātim al-Rāzī, al-Nasāʾī, and al-Dāraquṭnī) are preserved in their entirety.
* Preservation of technical vocabulary: Evaluative words (*alfāẓ al-jarḥ wa al-taʿdīl*) such as *thiqah*, *ṣadūq*, and *ḍaʿīf* are fully retained as legitimate predictive features when uttered by earlier critics. The task is formally defined as: *Predicting Ibn Ḥajar's 9th-century synthesis verdict from 2nd to 4th-century historical critic statements.*

---

## 3.5 Undirected Transmission Graph Formulation and Feature Extraction

Transmission relationships between narrators are modeled as an undirected, unweighted network stored in `output/cleaned_cohort/clean_graph.npz`.

### 3.5.1 Structural Formulation of the Transmission Graph
Let $G = (V, E)$ represent the clean transmission graph:
* $V$ denotes the set of $N = 335$ unique narrator identities. Each identity corresponds to exactly one node.
* $E$ denotes the set of $|E| = 354$ undirected edges. An edge $(u, v)$ exists if and only if narrator $u$ and narrator $v$ appear as adjacent transmission partners (teacher-student) in at least one canonical isnād chain.

Undirected edges are utilized because transmission interactions reflect mutual chronological and intellectual contact; information flow in hadith scholarship involves active interrogation and cross-examination between teachers and students.

### 3.5.2 Node Feature Vector Specification
Each node $v_i \in V$ is assigned an 8-dimensional tabular feature vector $x_i \in \mathbb{R}^8$, partitioned into two distinct feature groups:
1. **Topological Features ($x_{\text{topo}} \in \mathbb{R}^5$):**
   * Total degree centrality: $k_i = \sum_{j} A_{ij}$.
   * In-degree in directed projection: Number of unique teachers from whom the narrator received hadiths.
   * Out-degree in directed projection: Number of unique students to whom the narrator transmitted hadiths.
   * PageRank score: Structural prestige within the global canonical network.
   * Local clustering coefficient: Triangle density among the narrator's transmission partners.
2. **Biographical Metadata Features ($x_{\text{meta}} \in \mathbb{R}^3$):**
   * Generation index (*Ṭabaqa*): Categorical chronological generation mapped to an ordinal scalar (1 to 12).
   * Normalized death year: Recorded death year in the Hijrī calendar, scaled to the interval $[0, 1]$.
   * Active transmission span: Approximate career longevity estimated from earliest and latest transmission links.

---

## 3.6 Cross-Validation Strategy and Identity Isolation Protocol

To guarantee reliable performance estimates on an imbalanced dataset, the evaluation framework applies 5-fold Stratified Cross-Validation repeated across 5 distinct random seeds (42, 123, 456, 789, 2024), yielding 25 paired evaluation folds.

### 3.6.1 Identity Isolation Protocol
In prior flawed graph studies, nodes representing the same historical individual were partitioned across training and testing splits, allowing graph convolution layers to memorize identity-level properties. In this research, identity isolation is structurally enforced:
* Each unique narrator identity is assigned an integer identifier (`identity_id` from 0 to 334).
* The stratified partitioning algorithm partitions unique `identity_id` values rather than raw rows.
* Zero narrator identity overlap exists between the training and testing sets in any fold:

$$V_{\text{train}}^{(k)} \cap V_{\text{test}}^{(k)} = \emptyset \quad \forall k \in \{1, \dots, 25\}$$

All metric calculations (Macro-F1, Accuracy, and Quadratic Weighted Kappa) are computed strictly on out-of-fold test samples, preventing optimistic evaluation bias.
