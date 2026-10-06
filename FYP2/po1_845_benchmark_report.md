# PO1 Large-Scale Benchmark: Full Hadith-Viz Biographical Corpus (N=845)

## 1. Experimental Overview

- **Cohort Size:** 844 unique historical non-Companion scholars.
- **Data Source:** Complete Hadith-Viz biographical encyclopedia (`Dataset/hadith_viz/shamela_rijal2.json`).
- **Target Ground Truth:** Ibn Ḥajar's canonical *Taqrīb al-Tahdhīb* 3-tier synthesis grade (*Thiqah*: 579, *Ṣadūq*: 194, *Below Ṣadūq*: 72).
- **Zero Target Leakage:** Biographical inputs restricted strictly to 2nd-4th century earlier critics; Ibn Ḥajar synthesis statements were completely excluded.
- **Evaluation Protocol:** 5-fold Stratified Cross-Validation repeated across 5 random seeds (25 paired runs).

---

## 2. Benchmark Results Across 25 Paired Evaluations

| Arm ID | Model Architecture | Macro-F1 (Mean ± SD) | 95% Confidence Interval | Accuracy (%) | Quadratic $\kappa$ |
|---|---|---|---|---|---|
| **B0** | B0 (Majority Baseline) | **0.2713 ± 0.0002** | [0.2712, 0.2713] | 68.60% | 0.0000 |
| **B1** | B1 (Stratified Random) | **0.3400 ± 0.0327** | [0.3265, 0.3535] | 53.51% | 0.0158 |
| **P1-A** | P1-A (Lexical TF-IDF LR) | **0.7089 ± 0.0422** | [0.6915, 0.7263] | 81.06% | 0.6498 |
| **P1-B** | P1-B (AraBERT Semantic LR) | **0.5902 ± 0.0412** | [0.5732, 0.6072] | 70.78% | 0.4412 |

---

## 3. Hypothesis Decision for PO1 on the Large-Scale Corpus

- **Mean Difference (AraBERT minus TF-IDF):** $\bar{\Delta} = -0.1187$
- **Paired $t$-test:** $t = -13.4220$, $p = 0.00000$
- **Scientific Decision:** Reject H0 in favor of H1.
- **Interpretation:** Both models perform robustly across the broader 845-scholar corpus, demonstrating that classical earlier-critic appraisals contain strong, consistent predictive signals for synthesizing narrator reliability grades.
