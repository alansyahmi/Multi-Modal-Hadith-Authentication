# Dataset Attrition Table: Data Hygiene and Entity Resolution

**Artifact Reference:** `output/cleaned_cohort/clean_narrators_cohort.csv`  
**Associated Scripts:** `scripts/match_sanadset_to_hadithviz.py`, `scripts/build_clean_dataset.py`  
**Last Updated:** 2026-10-05 (Week 3 Post-Audit)

---

## 1. The Complete Data Attrition Funnel

To ensure total zero target leakage, eliminate phantom entities, and avoid circular reasoning, raw databases were filtered through a multi-stage data hygiene funnel:

| Funnel Stage | Entity / Record Count | Retention Rate | Scientific Rationale |
|---|---:|---:|:---|
| **Raw SanadSet Names** | 262,497 tokens | 100.0% | Unfiltered narrator name strings across 650,000 isnād chains. |
| **Unique Normalized Names** | 167,047 names | 63.6% | Standardized Arabic orthography (collapsing hamza, alif maqsura, ta marbuta). |
| **Hadith-Viz Master Cards** | 3,047 cards | 1.8% | Specialized visualizer index for canonical *Kutub al-Sittah* transmissions. |
| **Step 1: Exclude Unrated Stubs** | 942 cards | 30.9% | Removed 2,105 placeholder cards lacking verified Ibn Ḥajar synthesis grades. |
| **Step 2: Exclude Companions** | 860 scholars | 91.3% | Removed 82 Companions (*Ṣaḥābah*) due to theological constant (*ʿAdālat al-Ṣaḥābah*). |
| **Step 3: Exclude Empty Text** | **845 scholars** | 98.3% | Removed 15 entries lacking earlier-critic commentary. **Forms the PO1 Text Cohort.** |
| **Step 4: Graph Intersection** | 438 graph nodes | 51.8% | Isolated scholars actively appearing in canonical Kutub al-Sittah transmission chains. |
| **Step 5: Identity Deduplication** | **335 identities** | 76.5% | Merged spelling variants and split nodes into 1 unique identity = 1 graph node. **Forms the PO2 Multimodal GNN Cohort.** |

---

## 2. Downstream Chain Attrition (PO3 Cohort)

| Stage | Chain Count | Selection & Coverage Criteria |
|---|---:|:---|
| **SanadSet Total Chains** | 650,000 chains | Full raw dataset spanning all classical hadith literature. |
| **Kutub al-Sittah Chains** | 30,712 chains | Verified chains from Bukhārī, Muslim, Abū Dāwūd, Tirmidhī, Nasāʾī, and Ibn Mājah. |
| **Cohort-Covered Chains** | 9,840 chains | Chains containing at least 1 narrator from the clean cohort. |
| **Multi-Narrator Test Chains** | **1,578 chains** | Chains containing $\ge 2$ cohort narrators (mean cohort depth: 2.17 narrators per chain). Used for weakest-link chain validation. |

---

## 3. Audited Entity Matching Precision (Stratified Audit N=87)

To verify that SanadSet graph nodes were correctly matched to Hadith-Viz biographical cards, a stratified sample of 87 match pairs was manually audited:

| Stratum | Population ($N_h$) | Stratum Weight ($w_h$) | Sample ($n_h$) | Correct ($c_h$) | Stratum Precision ($P_h$) |
|---|---:|---:|---:|---:|---:|
| `multi_token_curated` | 1,098 | 0.6638 | 25 | 25 | 100.00% |
| `multi_token_v2` | 289 | 0.1747 | 25 | 25 | 100.00% |
| `single_token_curated` | 255 | 0.1542 | 25 | 9 | 36.00% |
| `single_token_v2` | 12 | 0.0073 | 12 | 6 | 50.00% |
| **Overall (Population-Weighted)** | **1,654** | **1.0000** | **87** | **65** | **89.77% (±2.81%)** |

> [!NOTE]
> Multi-token patronymics (e.g. `مُحَمَّدِ بْنِ إِسْحَاق`) achieved 100.0% precision. Single-token names (e.g. `عمر`) exhibited lower precision, which led to their complete exclusion from the final clean evaluation cohort.

---

## 4. Zero-Leakage Author Filtering Summary

* **Total Raw Appraisal Quotes in Dataset:** 36,643 quotes.
* **Statements Attributed to Ibn Ḥajar:** 1,659 quotes (4.53%). **100% stripped from model input text.**
* **Statements Attributed to al-Dhahabī:** 0 quotes in text (isolated strictly as external validation label).
* **Earlier Classical Critic Statements Retained:** 34,984 quotes from 2nd to 4th-century scholars.
