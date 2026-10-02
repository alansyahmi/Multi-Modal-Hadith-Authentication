# Experimental Findings Index

This directory contains all documented findings from the FYP project's experiments. Each file covers a specific experiment or analysis.

## Index

| # | File | Experiment | Status |
|---|------|------------|--------|
| 1 | [01_encyclopedia_extraction.md](./01_encyclopedia_extraction.md) | Narrator Encyclopedia Extraction | ✅ Complete |
| 2 | [02_alias_index.md](./02_alias_index.md) | Alias Index & Name Variant Coverage | ✅ Complete |
| 3 | [03_family_graph.md](./03_family_graph.md) | Family & Teacher-Student Graph | ✅ Complete |
| 4 | [04_track_a_arabert.md](./04_track_a_arabert.md) | Track A: AraBERT v2 Regression Training | ✅ Complete |
| 5 | [05_gnn_training.md](./05_gnn_training.md) | GNN Training (Full-Scale KG) | ✅ Complete |
| 6 | [06_chain_evaluation_mean.md](./06_chain_evaluation_mean.md) | Chain Evaluation: Mean-Based Scoring | ✅ Complete |
| 7 | [07_weakest_link_evaluation.md](./07_weakest_link_evaluation.md) | Chain Evaluation: Weakest-Link Scoring | ✅ Complete |
| 8 | [08_multi_sanad_convergence.md](./08_multi_sanad_convergence.md) | Multi-Sanad Convergence Analysis | ✅ Complete |
| 9 | [09_consensus_tier_analysis.md](./09_consensus_tier_analysis.md) | Consensus Tier (Ṣaḥīḥ vs Mutawātir) Analysis | ✅ Complete |
| 10 | [10_recommendations.md](./10_recommendations.md) | Recommendations & Next Steps | ✅ Complete |
| 11 | [11_critical_fixes_applied.md](./11_critical_fixes_applied.md) | Critical Fixes: Weakest-Link + Risk + Geo | ✅ Complete |
| 12 | [12_classifier_5_class.md](./12_classifier_5_class.md) | 5-Class Classifier (Track A v2) | ✅ Complete |
| 13 | [13_classifier_vs_canonical.md](./13_classifier_vs_canonical.md) | Classifier vs Kutub as-Sitta Grades | ✅ Complete |
| 14 | [14_turuq_corroboration.md](./14_turuq_corroboration.md) | Ṭuruq Multi-Chain Corroboration | ✅ Complete |
| 15 | [15_fyp2_feasibility.md](./15_fyp2_feasibility.md) | FYP2 Feasibility & Roadmap | ✅ Complete |
| **16** | [**16_fyp1_breakthrough.md**](./16_fyp1_breakthrough.md) | **FYP1 Breakthrough: 57% Chain Accuracy** 🚀 | **✅ Complete** |
| **17** | [**17_fyp1_final_results.md**](./17_fyp1_final_results.md) | **FYP1 Final: 73.6% Accuracy, 91.8% Ceiling** 🏁 | **✅ Final** |
| **18** | [**18_chapter4_5_mapping.md**](./18_chapter4_5_mapping.md) | **Chapter 4 & 5: 7 Experiments Mapping** 📋 | **✅ Complete** |
| **19** | [**19_objective_progress.md**](./19_objective_progress.md) | **Objective Progress Report** 🎯 | **✅ Complete** |
| **20** | [**20_geospatial_verification.md**](./20_geospatial_verification.md) | **Geospatial Verification (PO2)** 🌍 | **✅ Complete** |
| **21** | [**21_sanad_only_ceiling.md**](./21_sanad_only_ceiling.md) | **The Sanad-Only Ceiling: 35% Balanced Acc** 🔬 | **✅ Final** |

## Quick Reference: Key Metrics

| Experiment | Best Result | Compared To |
|---|---|---|
| AraBERT Spearman ρ | **0.8534** | 0.7584 (old encyclopedia) |
| AraBERT Cohen's κ | **0.4061** | 0.3321 (old encyclopedia) |
| GNN R² (full KG) | **+0.014** | -0.093 (old KG) |
| Weakest-link: Daif chains | **15.9%** | 0.1% (mean-based) |
| **Final pipeline: Maqbul** | **55.0%** | — |
| **Final pipeline: Daif** | **33.1%** | — |
| **Final pipeline: Matruk** | **11.8%** | — |
| Encyclopedia entries | **15,448** | 3,797 (old) |
| KG narrator coverage | **84%+** | ~10% (old) |
