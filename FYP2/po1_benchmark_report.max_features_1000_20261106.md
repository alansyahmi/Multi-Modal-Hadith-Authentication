# PO1 Canonical Benchmark Report: Configured Author-Filtered Biographical Corpus (N=819)

## 1. Experimental Overview

- **Cohort Size:** 819 unique historical non-Companion scholars.
- **Data Source:** Complete Hadith-Viz biographical encyclopedia (`Dataset/hadith_viz/shamela_rijal2.json`).
- **Target Ground Truth:** Canonical Ibn Ḥajar *Taqrīb* 3-tier synthesis grade (*Thiqah*: 558, *Ṣadūq*: 189, *Below Ṣadūq*: 72).
- **Temporal Author Filter:** Retains only the configured date-verified core critics from the 2nd to 4th centuries AH; excludes the 11 configured later authorities and snippets containing compiler references.
- **Evaluation Protocol:** 5-fold stratified cross-validation repeated across 5 random seeds (25 paired runs); the 819 evaluated source names are unique across rows.

- **TF-IDF Configuration:** Word n-grams (1, 2), `min_df=1`, `max_features=1000`, and sublinear term frequency; the vocabulary is fitted within each training fold.

---

## 2. Benchmark Results Across 25 Paired Evaluations

| Arm ID | Model Architecture | Macro-F1 (Mean ± SD) | Macro-Precision (Mean ± SD) | Macro-Recall (Mean ± SD) | 95% CI for Macro-F1 | Accuracy (%) | Quadratic Weighted $\kappa$ |
|---|---|---|---|---|---|---|---|
| **B0** | B0 (Majority Baseline) | **0.2702 ± 0.0006** | 0.2271 ± 0.0008 | 0.3333 ± 0.0000 | [0.2695, 0.2708] | 68.13% | 0.0000 |
| **B1** | B1 (Stratified Random) | **0.3261 ± 0.0398** | 0.3272 ± 0.0403 | 0.3282 ± 0.0400 | [0.2819, 0.3704] | 50.84% | -0.0195 |
| **P1-A** | P1-A (Lexical TF-IDF LR) | **0.6730 ± 0.0428** | 0.6666 ± 0.0416 | 0.6883 ± 0.0507 | [0.6255, 0.7205] | 79.41% | 0.6751 |
| **P1-B** | P1-B (AraBERT Semantic LR) | **0.5679 ± 0.0388** | 0.5574 ± 0.0355 | 0.5936 ± 0.0415 | [0.5248, 0.6110] | 70.55% | 0.5037 |

Macro-Precision and Macro-Recall are computed per fold as unweighted averages across the three reliability tiers, then summarized across the 25 folds as mean ± SD.

---

## 3. Formal Hypothesis Testing Decision for PO1

- **Mean Difference (AraBERT minus TF-IDF):** $\bar{\Delta} = -0.1051$
- **Corrected repeated-CV $t$-test ($df=24$):** $t = -3.7798$, $p = 9.17384e-04$
- **Test Method:** Nadeau-Bengio corrected repeated 5-fold CV paired t-test, accounting for overlap between folds.
- **Scientific Decision:** Reject $H_0$; the empirical evidence significantly favors the lexical baseline over AraBERT, contrary to $H_1$.
- **Interpretation:** Classical earlier-critic appraisals contain strong, consistent predictive signals for synthesizing narrator reliability grades. Regularized lexical n-gram representations capture specialized technical descriptors (*thiqah, laysa bihi ba's, da'if*) with higher fidelity than general-domain contextual language models under frozen feature extraction.
