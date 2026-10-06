# Clean Evaluation Cohort Specification (Audited and Verified Version)

**Artifact Reference:** `output/cleaned_cohort/clean_narrators_cohort.csv`  
**Associated Datasets:** `output/cleaned_cohort/clean_texts.json`, `output/cleaned_cohort/clean_graph.npz`  
**Last Updated:** 2026-10-06 (Week 3 Post-Audit)

---

## 1. Dual-Cohort Architecture Overview

To balance large-scale biographical evaluation with strict graph connectivity constraints, the research implements a two-level cohort design:

1. **Level 1 (Full Biographical Corpus Cohort, N=819):**
   Contains all verified non-Companion historical scholars in Hadith-Viz possessing canonical *Taqrīb* grades and substantive earlier critic commentary strictly dated to the 2nd to 4th centuries AH. All later synthesis judgments (including al-Mizzi, d. 742 AH, and 10 other post-classical authorities) are fully purged. Used for the PO1 biographical text representation benchmark.
2. **Level 2 (Multimodal Isnād Graph Cohort, N=335):**
   Contains the core subset of historical scholars who are simultaneously present in the leak-free biographical corpus and actively interconnected within the Kutub al-Sittah transmission graph. Evaluated under transductive node classification with label-disjoint splits across the 13-arm ablation study.

---

## 2. Class Distribution Across Tiers

The target ground truth is locked to Ibn Ḥajar al-ʿAsqalānī's canonical *Taqrīb al-Tahdhīb* synthesis verdict mapped onto the three standard operational tiers of *ʿIlm ar-Rijal*:

* **Tier 2 (*Thiqah* / High Reliability):** Scholars with grades of highest praise and established uprightness (*Thiqah Thabt*, *Thiqah Ḥāfiẓ*, *Thiqah*, *Mutqin*, *Imām*).
* **Tier 1 (*Ṣadūq* / Intermediate Reliability):** Scholars with acceptable truthfulness (*Ṣadūq*, *Lā baʾsa bih*, *Ṣadūq yakhṭiʾ*, *Ṣadūq yahim*).
* **Tier 0 (*Below Ṣadūq* / Weak or Cautionary):** Scholars with compromised transmission or cautionary qualification (*Maqbūl*, *Ḍaʿīf*, *Layyin*, *Matrūk*, *Munkar al-Ḥadīth*, *Laysa bi-Thiqah*).

### Comparative Tier Distribution Table

| Reliability Tier | Canonical Label | Level 1: Canonical Text Corpus ($N=819$) | Level 2: Multimodal Graph Cohort ($N=335$) | Dual-Synthesizer Consensus Subset ($N=180$) |
|:---:|:---|---:|---:|---:|
| 2 | *Thiqah* (Reliable) | 558 (68.13%) | 255 (76.12%) | 160 (88.89%) |
| 1 | *Ṣadūq* (Truthful) | 189 (23.08%) | 60 (17.91%) | 14 (7.78%) |
| 0 | *Below Ṣadūq* (Weak / Cautionary) | 72 (8.79%) | 20 (5.97%) | 6 (3.33%) |
| **Total** | **All Graded Narrators** | **819 (100.0%)** | **335 (100.0%)** | **180 (100.0%)** |

---

## 3. Structural Zero-Leakage Text Specifications

Input text profiles are restricted exclusively to earlier classical critics (2nd to 4th century AH):
* **Included Earlier Critics:** Aḥmad ibn Ḥanbal (d. 241), Yaḥyā ibn Maʿīn (d. 233), ʿAlī ibn al-Madīnī (d. 234), Muḥammad ibn Ismāʿīl al-Bukhārī (d. 256), Abū Ḥātim al-Rāzī (d. 277), Abū Zurʿah al-Rāzī (d. 264), al-Nasāʾī (d. 303), al-Tirmidhī (d. 279), Ibn ʿAdī (d. 365), and al-Dāraquṭnī (d. 385).
* **Excluded Later Synthesizers:** All 7th to 9th century AH synthesizing authorities are systematically purged:
  1. Ibn Ḥajar al-ʿAsqalānī (d. 852 AH)
  2. Shams al-Dīn al-Dhahabī (d. 748 AH)
  3. Jamāl al-Dīn al-Mizzī (d. 742 AH)
  4. ʿAlāʾ al-Dīn Mughulṭāy (d. 762 AH)
  5. Zayn al-Dīn al-ʿIrāqī (d. 806 AH)
  6. Khalīl ibn Kaykaldī al-ʿAlāʾī (d. 761 AH)
  7. Yaḥyā ibn Sharaf al-Nawawī (d. 676 AH)
  8. Burhān al-Dīn al-Ḥalabī / Sibṭ Ibn al-ʿAjamī (d. 841 AH)
  9. Shams al-Dīn Ibn ʿAbd al-Hādī (d. 744 AH)
  10. Shams al-Dīn al-Sakhāwī (d. 902 AH)
  11. Jalāl al-Dīn al-Suyūṭī (d. 911 AH)
* **Text Coverage ($N=335$):**
  * Minimum character length: 31 characters.
  * Median character length: 2,886 characters.
  * Maximum character length: 66,186 characters.
  * Total earlier critic quotes retained across corpus: 34,984 quotes.

---

## 4. Condensed Isnād Graph Specifications

* **Nodes:** 335 unique consolidated historical identities (condensed from 438 raw graph nodes).
* **Edges:** 177 unique undirected teacher-student transmission pairs, represented as 354 directed message-passing edges in PyTorch Geometric (`edge_index` shape: 2 x 354).
* **Self-Loops:** 0 (all historical conflation self-loops were permanently removed).
* **Graph Learning Formulation:** Transductive node classification with label-disjoint train/test splits.
* **Feature Separation:**
  * `x_topo` (5 dimensions): appearances in corpus, in-degree, out-degree, net-flow (in-degree minus out-degree), and mean chain position.
  * `x_meta` (3 dimensions): death year (AH), normalized tabaqa generation index, and earlier critic statement count.
  * `x_text` (768 dimensions): frozen attention-weighted mean-pooled AraBERT embeddings (`aubmindlab/bert-base-arabertv2`).

---

## 5. 5-Fold Stratified Cross-Validation Splitting Protocol

Evaluated across 5 random seeds (Seeds: 42, 123, 456, 789, 2024), generating 25 total paired evaluations:
* **Train Partition per Fold:** 268 narrator identities (80.0%).
* **Test Partition per Fold:** 67 narrator identities (20.0%).
* **Identity Grouping:** Strict 1 identity = 1 node mapping. Zero narrator identity overlap exists between train and test partitions.
* **Historical Disagreements:** Exactly 58 scholars exhibit divergent tier grades between Ibn Ḥajar and al-Dhahabī out of 238 shared evaluated scholars (with 180 reaching identical consensus).