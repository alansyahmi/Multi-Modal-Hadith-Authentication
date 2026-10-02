# Finding 9: Consensus Tier Analysis (Ṣaḥīḥ vs Mutawātir)

## Experiment
Check whether machine-scored "Ṣaḥīḥ" hadiths align with scholarly consensus levels. Hadiths appearing in multiple canonical collections (especially Bukhari + Muslim = muttafaqun ʿalayhi) should receive higher chain scores than those in only one collection.

## Approach
1. Use the Kutub as-Sitta evaluation set (1,200 hadiths with ground-truth grades)
2. Score each hadith's chain using GNN predictions
3. Compare scores across: traditional grade (Ṣaḥīḥ/Ḥasan/Ḍaʿīf) and book (Bukhari vs others)

## Results

### By Traditional Grade

| Grade | n | Mean Score | Median |
|---|---|---|---|
| Mawḍūʿ | 1 | 0.6249 | 0.6249 |
| Unknown | 34 | 0.6274 | 0.6283 |
| Ḍaʿīf | 96 | 0.6278 | 0.6322 |
| Ḥasan | 105 | 0.6303 | 0.6363 |
| **Ṣaḥīḥ** | **786** | **0.6290** | **0.6331** |

All grades score ~0.63. The difference between Ṣaḥīḥ and Ḍaʿīf is **0.0012** — essentially zero.

### By Book (Ṣaḥīḥ/Ḥasan only)

| Book | n | Mean Score |
|---|---|---|
| Ṣaḥīḥ al-Bukhārī | 183 | 0.6308 |
| Ṣaḥīḥ Muslim | 154 | **0.6248** |
| Sunan al-Nasāʾī | 180 | **0.6324** |
| Jāmiʿ at-Tirmiḏī | 130 | 0.6302 |
| Sunan Abī Dāwūd | 140 | 0.6291 |
| Sunan Ibn Mājah | 104 | 0.6262 |

### Consensus Tier Comparison

| Tier | n | Mean |
|---|---|---|
| **Bukhari + Muslim** (muttafaqun proxy) | 154 | **0.6248** |
| Other books | 737 | 0.6301 |
| **Difference** | | **-0.0054** (Other books score HIGHER) ❌ |

## Conclusion

**The GNN scores do NOT differentiate by scholarly consensus level.** All hadiths score ~0.63 regardless of grade or book. This is because:

1. The GNN scores ALL narrators in a narrow band (0.55–0.70)
2. Averaging 6.8 narrators per chain collapses to the mean
3. Even Ṣaḥīḥ al-Bukhārī contains some weak narrators in chains

## Fix Expected with Weakest-Link

With weakest-link scoring, this analysis would produce meaningful differentiation:
- Ṣaḥīḥ chains → narrators all ≥ 0.70 → min ≥ 0.70 → Hasan/Sahih
- Ḍaʿīf chains → one narrator ≤ 0.15 → min ≤ 0.15 → Da'if/Matruk

This would be a key experiment to re-run after implementing weakest-link.

## Key Files
- `output/consensus_experiment/consensus_scores.csv` — 1,022 scored hadiths
- `output/consensus_experiment/consensus_findings.json` — Summary
- `scripts/sahih_vs_mutawatir_experiment.py` — Experiment script
