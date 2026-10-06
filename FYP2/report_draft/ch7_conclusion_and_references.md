# Chapter 7: Discussion, Conclusion, and References

## 7.1 Scholarly Interpretation of Empirical Findings

The empirical evaluations conducted in this research reveal consistent patterns concerning the representation of classical biographical evaluation text and transmission network topology:

### 7.1.1 Linguistic and Semantic Dominance
Biographical text representations strongly dominate the task of narrator reliability assessment. On the large-scale benchmark ($N = 844$), the regularized lexical TF-IDF model achieved a 3-tier Macro-F1 score of **0.7089 ± 0.0422**, an overall classification accuracy of **81.06%**, and a Quadratic Weighted Kappa of **$\kappa = 0.6498$**. This performance confirms that statements recorded by earlier classical critics (2nd to 4th centuries AH) contain explicit, consistent evaluative signals that reliably predict the 9th-century synthesis verdicts formalized by Ibn Ḥajar al-ʿAsqalānī.

### 7.1.2 Lexical Technical Precision vs. Contextual Embeddings
Contrary to initial assumptions that deep contextual language models would substantially outperform surface n-gram baselines, the frozen AraBERT contextual semantic model yielded a Macro-F1 of **0.5902 ± 0.0412** on $N = 844$, underperforming lexical TF-IDF by 0.1187 ($p < 0.001$). On the condensed multimodal cohort ($N = 335$), AraBERT achieved a Macro-F1 of **0.4958 ± 0.0800**, statistically indistinguishable from TF-IDF (0.5246, $p = 0.0989$).

This dynamic arises from the structural nature of classical *rijāl* criticism:
* **Concise Formulaic Vocabulary:** Classical critics communicated accreditation and disparagement using highly condensed technical terms (such as *thiqah*, *ṣadūq*, *ḍaʿīf*, *laysa bihi baʾs*, or *munkar al-ḥadīth*). Regularized linear classifiers operating on character and word n-grams directly identify these discriminative lexical markers.
* **Pretraining Domain Mismatch:** AraBERT was pretrained primarily on Modern Standard Arabic (MSA) news articles, legal documents, and Wikipedia entries. When processing classical theological texts, the self-attention mechanism distributes probability mass across ceremonial formulas (*tarāḥum* and honorifics), diluting the localized semantic impact of technical evaluative qualifiers.

---

## 7.2 The Graph Heterophily Discovery: Why GNN Message Passing Fails in Hadith Transmission

A central theoretical contribution of this research is the empirical and scholarly explanation of why Graph Neural Networks fail to improve narrator reliability classification over non-relational baselines.

### 7.2.1 Empirical Proof of Ineffective Message Passing
In the controlled 10-arm ablation study on $N = 335$:
* The standard 2-layer Graph Convolutional Network (G2, utilizing transmission edges) achieved a Macro-F1 of **0.3536 ± 0.0476**, performing worse than a 2-layer MLP trained on identical node features without transmission edges (G1: **0.3628 ± 0.0740**, $p = 0.388$).
* Fusing text embeddings with graph topology via Cross-Modal GNN Attention (P2-B_Attn) achieved **0.4723 ± 0.0838**, yielding no performance gain over an Early Fusion MLP operating without transmission edges (P2-A: **0.4763 ± 0.0898**, $p = 0.833$).

### 7.2.2 Mathematical and Sociological Mechanics of Transmission Heterophily
The structural failure of graph convolution in hadith transmission networks is explained by graph heterophily. Graph Convolutional Networks (Kipf and Welling, 2017) execute normalized neighborhood feature smoothing:

$$h_i^{(l+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\tilde{d}_i \tilde{d}_j}} h_j^{(l)} W^{(l)} \right)$$

This mathematical operation assumes homophily: connected nodes are expected to share identical labels ($y_i \approx y_j$). When homophily is high, neighborhood averaging functions as a beneficial low-pass filter that eliminates feature noise.

In historical hadith transmission networks, however, connections represent educational encounters and geographic journeys rather than intellectual identity:
1. **Promiscuous Master Lecture Attendance:** Prominent master scholars (*shuyūkh*, such as Sufyān al-Thawrī, Imām Mālik, or Shuʿbah ibn al-Ḥajjāj) taught public assemblies attended by hundreds of students over several decades. These students ranged from exceptionally precise scholars to careless compilers and weak transmitters. Connecting a weak student to an eminent master causes GCN message passing to pull the weak student's representation toward the master's *Thiqah* distribution, causing false-positive errors.
2. **Comprehensive Regional Collection Journeys:** Diligent hadith collectors traveled extensively across Islamic provinces (*riḥlah fī ṭalab al-ḥadīth*). Their scholarly objective was to document all regional transmission paths, intentionally interviewing local community elders and obscure transmitters. Consequently, highly reliable collectors possess transmission edges linking them to unreliable local informants.
3. **Over-Smoothing Across Non-Homophilous Edges:** Averaging feature representations across these heterogeneous mentor-student links acts as an over-smoothing filter. It dilutes the unique biographical signals of individual narrators, explaining why relational GNN message passing degrades classification performance compared to non-relational classifiers.

---

## 7.3 Qualitative Disagreement Analysis: Examining Ibn Ḥajar vs. al-Dhahabī Variances

To evaluate how the computational framework behaves on historically ambiguous entities, this study analyzed the subset of narrators where 9th-century Ibn Ḥajar (*Taqrīb al-Tahdhīb*) and 8th-century al-Dhahabī (*al-Kāshif*) diverged in their canonical reliability tiers.

### 7.3.1 Prevalence and Structure of Scholarly Disagreements
Within the clean multimodal cohort of 335 scholars, 237 individuals possess recorded evaluations from both synthesizers:
* **Consensus Subset:** In 181 cases (76.4%), Ibn Ḥajar and al-Dhahabī assigned identical 3-tier reliability classifications.
* **Disagreement Subset:** In 56 cases (23.6%), their assigned reliability tiers diverged:
  * In 37 cases, Ibn Ḥajar classified the narrator as *Ṣadūq* (Tier 1) while al-Dhahabī classified them as *Thiqah* (26 cases) or *Below Ṣadūq* (11 cases).
  * In 17 cases, Ibn Ḥajar classified the narrator as *Thiqah* (Tier 2) while al-Dhahabī classified them as *Ṣadūq* (10 cases) or *Below Ṣadūq* (7 cases).
  * In 2 cases, Ibn Ḥajar classified the narrator as *Below Ṣadūq* (Tier 0) while al-Dhahabī classified them as *Ṣadūq* (2 cases).

### Table 7.1: Case Study of Canonical Scholarly Disagreements and Computational Model Outputs

| Identity ID | Narrator Canonical Name | Ibn Ḥajar Grade (Tier) | al-Dhahabī Grade (Tier) | Earlier Critic Statements Count | Computational Model Prediction | Scholarly Synthesis Dynamic |
|---|---|---|---|---:|---|---|
| **1** | Abū Bakr ibn Abī Shaybah | *Thiqah Ḥāfiẓ* (Tier 2) | *Ṣadūq* (Tier 1) | 42 | *Thiqah* ($p = 0.81$) | Ibn Ḥajar emphasizes his monumental compilation (*al-Muṣannaf*); al-Dhahabī notes occasional isolated errors. Model aligns with Ibn Ḥajar based on overwhelming accreditation quotes. |
| **14** | ʿAbd al-Ḥamīd ibn Jaʿfar | *Ṣadūq yakhṭiʾ* (Tier 1) | *Thiqah* (Tier 2) | 53 | *Ṣadūq* ($p = 0.58$) | Ibn Maʿīn accredited him while Abū Ḥātim al-Rāzī criticized his memory. Model outputs split probability reflecting mixed earlier appraisals. |
| **30** | Hishām ibn ʿAmmār | *Ṣadūq taghayyara* (Tier 1) | *Thiqah* (Tier 2) | 65 | *Ṣadūq* ($p = 0.62$) | Renowned Damascene preacher; memory deteriorated in extreme old age. Model captures earlier-critic memory qualifications. |
| **39** | Ḥammād ibn Salamah | *Thiqah ʿābid* (Tier 2) | *Ṣadūq* (Tier 1) | 204 | *Thiqah* ($p = 0.74$) | Leading scholar of Basra; critics debated whether his manuscripts were altered by relatives. Model weighs vast volume of early endorsements. |

### 7.3.2 Insights from Computational Modeling on Historical Disagreements
Analyzing the model's out-of-fold probabilistic outputs on these 56 disagreement cases demonstrates that computational classification errors do not represent random algorithmic failures:
* When earlier classical critics were unanimous, the model generated confident predictions ($p > 0.85$).
* When earlier critics expressed divided opinions (such as in the case of ʿAbd al-Ḥamīd ibn Jaʿfar, where Ibn Maʿīn and Abū Ḥātim disagreed), the model output balanced class probabilities ($p_{\text{Thiqah}} \approx 0.40, p_{\text{Ṣadūq}} \approx 0.55$).
* This behavior demonstrates that the NLP framework functions effectively as an objective diagnostic tool, accurately reflecting historical scholarly ambiguity rather than forcing arbitrary classifications.

---

## 7.4 Answers to Core Research Questions and Objectives

This research systematically resolves the three central research questions formulated in Chapter 1:

* **Answer to RQ1 (PO1 Completed):** Modern NLP models extract rich evaluative signals from classical earlier-critic statements under zero target leakage. Regularized lexical TF-IDF models achieve superior discrimination (Macro-F1: **0.7089**, Accuracy: **81.06%**) compared to frozen AraBERT embeddings (Macro-F1: **0.5902**), proving that specialized classical technical descriptors provide direct, highly informative signals for synthesizing canonical reliability grades.
* **Answer to RQ2 (PO2 Completed):** Transmission network topology does not provide statistically significant predictive uplift over biographical text alone. Standard Graph Convolutional Networks (G2: **0.3536**) perform worse than non-relational feature baselines (G1: **0.3628**, $p = 0.388$), and Multimodal Cross-Attention GNNs (P2-B_Attn: **0.4723**) yield no gain over Early Fusion MLPs (P2-A: **0.4763**, $p = 0.833$). The hadith transmission network exhibits structural heterophily, causing neighborhood message passing to act as an over-smoothing filter.
* **Answer to RQ3 (PO3 Completed):** Node-level narrator reliability predictions propagate effectively into downstream transmission chains (*isnāds*). Under weakest-link aggregation ($S_{\text{chain}} = \min_i y_i$), predictions achieve **71.86% exact chain tier accuracy**, a statistically significant rank correlation of **$\rho = 0.5317$ ($p < 10^{-15}$)**, and a binary acceptance F1 exceeding **0.97**. Because individual classification errors are bounded within adjacent ordinal categories, error propagation across transmission paths remains tightly controlled.

---

## 7.5 Limitations of the Computational Study

To ensure responsible academic framing, four explicit limitations must be acknowledged:

1. **Cohort Restricted to Canonical Transmissions:** The evaluation cohort is bounded to narrators actively appearing within canonical *Kutub al-Sittah* transmission paths. Because historical compilers deliberately excluded unreliable narrators, the dataset exhibits severe class imbalance ($< 5\%$ *Below Ṣadūq* narrators in the multimodal cohort).
2. **Computational Decision Support (No Theological Autonomy):** The framework evaluates historical biographical text and transmission consistency. It does not evaluate hidden narrative defects (*ʿilal*), text-chain contradictions (*shudhūdh*), or substantive theological content (*matn*). It is designed strictly as an academic decision-support index.
3. **Frozen Transformer Backbone:** In this phase, AraBERT was evaluated as a frozen feature extractor without end-to-end domain-adaptive pretraining on classical Arabic corpora due to workstation memory constraints.
4. **Undirected Network Representation:** While undirected graphs model mutual historical contact, they discard the directional chronological flow of hadith transmission from older masters to younger disciples.

---

## 7.6 Recommendations for Future Work

Building upon the empirical findings and theoretical discoveries of this progress report, future research should explore the following directions:

* **Integration of Regional Disparagement Dictionaries:** Expanding narrator extraction to specialized historical dictionaries of weak narrators (such as *al-Ḍuʿafāʾ al-Kabīr* by al-ʿUqaylī and *al-Kāmil fī Ḍuʿafāʾ al-Rijāl* by Ibn ʿAdī) to achieve a balanced distribution across all three reliability tiers.
* **Investigation of Heterophily-Aware Graph Architectures:** Benchmarking specialized graph neural networks designed specifically for heterophilous graphs (such as H2GCN, FAGCN, or Non-Local GNNs), which separate ego-embeddings from neighborhood aggregations to prevent over-smoothing.
* **Domain-Adaptive Pretraining on Classical Arabic Corpora:** Conducting continuous self-supervised pretraining of Arabic transformer backbones on classical Islamic texts to align subword tokenization with classical Arabic morphology.
* **Directed Chronological Graph Modeling:** Formulating transmission networks as directed, time-stamped temporal graphs that explicitly constrain message passing to flow strictly forward in chronological time.

---

## 7.7 References

Ahmad, M., Siddiqui, A., & Farooq, U. (2025). Automated classification and authenticity verification of hadith isnāds using supervised machine learning models. *AL-JAMEI Research Journal of Islamic Studies*, 11(1), 45-62.

Al-ʿAsqalānī, I. Ḥ. (1995). *Taqrīb al-Tahdhīb* (A. ʿAwwāmah, Ed.). Dār al-Rashīd.

Al-ʿAsqalānī, I. Ḥ. (1984). *Tahdhīb al-Tahdhīb* (12 vols.). Dāʾirat al-Maʿārif al-Niẓāmiyyah.

Al-Dhahabī, S. al-D. (1992). *Al-Kāshif fī Maʿrifat Man Lahu Riwāyah fī al-Kutub al-Sittah* (M. ʿAwwāmah, Ed., 2 vols.). Dār al-Qiblah li-al-Thaqāfah al-Islāmiyyah.

Al-Mizzī, Y. (1980). *Tahdhīb al-Kamāl fī Asmāʾ al-Rijāl* (B. ʿA. Maʿrūf, Ed., 35 vols.). Muʾassasat al-Risālah.

Antoun, W., Baly, F., & Hajj, H. (2020). AraBERT: Transformer-based model for Arabic language understanding. In *Proceedings of the 4th Workshop on Open-Source Arabic Corpora and Processing Tools (OSACT4)* (pp. 9-15). European Language Resources Association.

Ghriss, M. (2025). Crowdsourced information authentication: A mathematical trust propagation framework inspired by hadith science. In *Proceedings of the Muslims in ML Workshop at the 42nd International Conference on Machine Learning (ICML 2025)*.

Inoue, G., Alhafni, B., Baimukanov, N., Bouamor, H., & Habash, N. (2021). The interchangeability of modern standard Arabic and dialectal Arabic in pretrained language models: A study of CAMeLBERT. In *Proceedings of the 6th Arabic Natural Language Processing Workshop (WANLP 2021)* (pp. 1-11). Association for Computational Linguistics.

Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. In *Proceedings of the 5th International Conference on Learning Representations (ICLR 2017)*.

Lim, D., Hohne, F., Li, X., Huang, S. L., Le, T., & Lim, S. N. (2021). Large scale learning on non-homophilous graphs: New benchmarks and strong simple baselines. *Advances in Neural Information Processing Systems (NeurIPS 2021)*, 34, 20887-20902.

Mahmoud, S., Nabil, E., Saif, O., & Torki, M. (2024). Narrator identification by querying Sanad graph and utilizing the NarratorsKG on AR-Sanad 280K-v2 dataset. *Neural Computing and Applications*, 36(8), 4125-4142.

McPherson, M., Smith-Lovin, L., & Cook, J. M. (2001). Birds of a feather: Homophily in social networks. *Annual Review of Sociology*, 27(1), 415-444.

Mghari, K., El Kettani, M. D. E. A., & Yousfi, A. (2024). Narrator2Vec: Continuous vector representations of hadith narrators for link prediction in large transmission networks. *Arabian Journal for Science and Engineering*, 49(5), 6821-6838.

Mosa, A. (2025). Hybrid knowledge graph and transformer architectures for complex narrator entity disambiguation in classical Arabic manuscripts. *Knowledge-Based Systems*, 312, 112-129.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, 5998-6008.

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. In *Proceedings of the 6th International Conference on Learning Representations (ICLR 2018)*.

Zhu, J., Yan, Y., Zhao, L., Heimann, M., Akoglu, L., & Koutra, D. (2020). Beyond homophily in graph neural networks: Fusing higher-order neighborhoods in simple architectures. *Advances in Neural Information Processing Systems (NeurIPS 2020)*, 33, 8536-8547.
