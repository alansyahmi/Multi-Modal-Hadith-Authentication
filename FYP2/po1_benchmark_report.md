# PO1 Canonical Benchmark Report: Strict Pre-Ibn-Ḥajar Biographical Corpus (N=819)

## 1. Experimental Overview

- **Cohort Size:** 819 unique historical non-Companion scholars.
- **Data Source:** Complete Hadith-Viz biographical encyclopedia (`Dataset/hadith_viz/shamela_rijal2.json`).
- **Target Ground Truth:** Canonical Ibn Ḥajar *Taqrīb* 3-tier synthesis grade (*Thiqah*: 558, *Ṣadūq*: 189, *Below Ṣadūq*: 72).
- **Strict Author Filtering:** Strictly excludes all 11 later synthesizing authorities: Ibn Ḥajar (d. 852), al-Dhahabī (d. 748), al-Mizzī (d. 742), Mughulṭāy (d. 762), al-ʿIrāqī (d. 806), al-ʿAlāʾī (d. 761), al-Nawawī (d. 676), Burhān al-Dīn al-Ḥalabī (d. 841), Ibn ʿAbd al-Hādī (d. 744), al-Sakhāwī (d. 902), and al-Suyūṭī (d. 911).
- **Evaluation Protocol:** 5-fold Stratified Cross-Validation repeated across 5 random seeds (25 paired runs), zero train/test identity overlap.

---

## 2. Benchmark Results Across 25 Paired Evaluations

| Arm ID | Model Architecture | Macro-F1 (Mean ± SD) | 95% Confidence Interval | Accuracy (%) | Quadratic Weighted $\kappa$ |
|---|---|---|---|---|---|
| **B0** | B0 (Majority Baseline) | **0.2702 ± 0.0006** | [0.2699, 0.2704] | 68.13% | 0.0000 |
| **B1** | B1 (Stratified Random) | **0.3261 ± 0.0398** | [0.3105, 0.3417] | 50.84% | -0.0195 |
| **P1-A** | P1-A (Lexical TF-IDF LR) | **0.6852 ± 0.0514** | [0.6651, 0.7054] | 80.59% | 0.6697 |
| **P1-B** | P1-B (AraBERT Semantic LR) | **0.5635 ± 0.0373** | [0.5488, 0.5781] | 69.99% | 0.5008 |

---

## 3. Formal Hypothesis Testing Decision for PO1

- **Mean Difference (AraBERT minus TF-IDF):** $\bar{\Delta} = -0.1218$
- **Paired $t$-test ($df=24$):** $t = -11.4153$, $p = 3.48986e-11$
- **Wilcoxon Signed-Rank Test:** $W = 0.0$, $p = 5.96046e-08$
- **Scientific Decision:** Reject $H_0$; the empirical evidence significantly favors the lexical baseline over AraBERT, contrary to $H_1$.
- **Interpretation:** Classical earlier-critic appraisals contain strong, consistent predictive signals for synthesizing narrator reliability grades. Regularized lexical n-gram representations capture specialized technical descriptors (*thiqah, laysa bihi ba's, da'if*) with higher fidelity than general-domain contextual language models under frozen feature extraction.
