# Clean Evaluation Cohort Specification (Audited & Recovered Version)

**Artifact Reference:** `output/cleaned_cohort/clean_narrators_cohort.csv`  
**Associated Datasets:** `output/cleaned_cohort/clean_texts.json`, `output/cleaned_cohort/clean_graph.npz`  
**Last Updated:** 2026-10-05 (Week 3 Post-Audit)

---

## 1. Dual-Cohort Architecture Overview

To balance large-scale biographical evaluation with strict graph connectivity constraints, the research implements a two-level cohort design:

1. **Level 1 (Full Biographical Corpus Cohort, N=845):**
   Contains all verified non-Companion historical scholars in Hadith-Viz possessing canonical *Taqrīb* grades and substantive earlier-critic commentary. Used for the PO1 biographical text representation benchmark.
2. **Level 2 (Multimodal Isnād Graph Cohort, N=335):**
   Contains the core subset of historical scholars who are simultaneously present in the biographical corpus and actively interconnected within the Kutub al-Sittah transmission graph. Used for the PO2 multimodal graph ablation study.

---

## 2. Class Distribution Across Tiers

The target ground truth is locked to Ibn Ḥajar al-ʿAsqalānī's canonical *Taqrīb al-Tahdhīb* synthesis verdict mapped onto the three standard operational tiers of *ʿIlm ar-Rijal*:

* **Tier 2 (*Thiqah* / High Reliability):** Scholars with grades of highest praise and established uprightness (*Thiqah Thabt*, *Thiqah Ḥāfiẓ*, *Thiqah*, *Mutqin*, *Imām*).
* **Tier 1 (*Ṣadūq* / Intermediate Reliability):** Scholars with acceptable truthfulness (*Ṣadūq*, *Lā baʾsa bih*, *Ṣadūq yakhṭiʾ*, *Ṣadūq yahim*).
* **Tier 0 (*Below Ṣadūq* / Weak or Rejected):** Scholars with compromised transmission (*Maqbūl*, *Ḍaʿīf*, *Layyin*, *Matrūk*, *Munkar al-Ḥadīth*).

### Comparative Tier Distribution Table

| Reliability Tier | Canonical Label | Level 1: Full Biographical Corpus ($N=845$) | Level 2: Multimodal Graph Cohort ($N=335$) | Dual-Synthesizer Consensus Subset ($N=181$) |
|:---:|:---|---:|---:|---:|
| 2 | *Thiqah* (Reliable) | 579 (68.52%) | 258 (77.01%) | 161 (88.95%) |
| 1 | *Ṣadūq* (Truthful) | 194 (22.96%) | 64 (19.10%) | 14 (7.73%) |
| 0 | *Below Ṣadūq* (Weak / Rejected) | 72 (8.52%) | 13 (3.88%) | 6 (3.31%) |
| **Total** | **All Graded Narrators** | **845 (100.0%)** | **335 (100.0%)** | **181 (100.0%)** |

---

## 3. Structural Zero-Leakage Text Specifications

Input text profiles are restricted exclusively to earlier classical critics (2nd to 4th century AH):
* **Included Critics:** Aḥmad ibn Ḥanbal, Yaḥyā ibn Maʿīn, ʿAlī ibn al-Madīnī, Muḥammad ibn Ismāʿīl al-Bukhārī, Abū Ḥātim al-Rāzī, Abū Zurʿah al-Rāzī, al-Nasāʾī, al-Tirmidhī, Ibn ʿAdī, and al-Dāraquṭnī.
* **Excluded Synthesizers:** All 9th-century statements by Ibn Ḥajar al-ʿAsqalānī and 8th-century statements by Shams al-Dīn al-Dhahabī were strictly removed from the text inputs.
* **Text Length Distribution ($N=335$):**
  * Minimum character length: 31 characters.
  * Median character length: 2,886 characters.
  * Maximum character length: 66,186 characters.
  * Median statements per narrator: 29 individual classical quotes.

---

## 4. Condensed Isnād Graph Specifications

* **Nodes:** 335 unique consolidated historical identities (condensed from 438 raw graph nodes).
* **Edges:** 354 undirected transmission links (representing bidirectional student-teacher interactions).
* **Self-Loops:** 0 (8 historical conflation self-loops were permanently removed).
* **Feature Separation:**
  * `x_topo` (5 dimensions): in-degree, out-degree, total degree, PageRank, clustering coefficient.
  * `x_meta` (3 dimensions): imputed birth year, death year, ṭabaqa generation index.
  * `x_text` (768 dimensions): frozen mean-pooled AraBERT embeddings.

---

## 5. 5-Fold Stratified Cross-Validation Splitting Protocol

Evaluated across 5 random seeds (Seeds: 42, 123, 456, 789, 2024), generating 25 total paired evaluations:
* **Train Partition per Fold:** ~268 narrator identities (80.0%).
* **Test Partition per Fold:** ~67 narrator identities (20.0%).
* **Identity Grouping:** Strict 1 identity = 1 node mapping. Zero narrator identity overlap exists between train and test partitions.