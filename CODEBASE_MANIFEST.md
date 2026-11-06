# Frozen v4 Codebase Manifest

Created: 2026-11-06
Source bundle: C:/Projects/FYP Project v4
Source branch: v4
Base commit: 48dbfeaf033cefa6f39a19f2ae45a134d42f8a71
Selected report: FYP2/report_draft/fyp2_report_v4_final.pdf
Recorded snapshot SHA-256: 783D64523795CA3C29D6F8C837DC4DA25E8E282E1856A2172D7F1BAF630D8B96

This folder was refreshed from the report-centered v4 bundle so its code, data, saved outputs, and walkthrough correspond to the frozen presentation version. The separate v4 bundle's provenance record is preserved at FYP2/V4_BUNDLE_MANIFEST.md.

The bundle is not a bit-for-bit source release from the report submission date. It was reconstructed using the recorded snapshot and timestamp rule documented in FYP2/V4_BUNDLE_MANIFEST.md. The original forensic snapshot remains PARTIALLY VERIFIED - EXCEPTION DOCUMENTED. This folder does not upgrade that status.

The root codebase contains the v4 source tree, selected benchmark scripts, dataset sources, saved outputs, report, figures, and references. No experiment scripts were run during this refresh, and hashes were not recomputed.

The Stage 1 and Stage 2 default alignment filenames differ in the retained v4 code. The report benchmarks also run through separate scripts. Consult docs/CODE_WALKTHROUGH.md and docs/CODE_MAP.md for the boundaries between these workflows.

## Report-era firewall source recovery (2026-11-06)

The original report-centered v4 bundle's `src/canonical_data_pipeline.py` lacked `has_late_compiler_leak()`, although `scripts/build_clean_dataset.py` imported and called it. The walkthrough copy now starts from the v4 bundle's baseline pipeline and restores the two-filter behavior described by the submitted report: the existing `is_earlier_critic()` exclusion list and a snippet-level name-substring check integrated into `extract_clean_critic_snippets()`.

Evidence: the forensic snapshot at `C:\FYP_Forensic_Archive\FYP_Project_Snapshot_2026-11-06\PROJECT\src\canonical_data_pipeline.py` contains the helper, and the October 17, 2026 change-log entry plus the October 30, 2026 report describe its addition before submission. The snapshot copy also contains later edits (including a date-verified critic allowlist and a label-mapping change); those later edits were not copied here. This is a source-backed reconstruction of the report-described behavior, not proof of the exact source bytes that generated the report results. The missing `scripts/match_sanadset_to_hadithviz.py` was also restored from the forensic snapshot; it is referenced by the attrition table and dated October 3, 2026. Its inputs are present in this walkthrough copy, but it was not run because it overwrites `FYP2/attrition_table.md`. The forensic v4 bundle remains unchanged, and saved results were not regenerated.

## Post-rerun filter refinement (2026-11-06)

The broad substring implementation was found to remove Iraqi-demographic references and literal-gold mentions. The walkthrough copy now uses whole-token author-name checks, exact `الذهبي` spelling, and citation-context matching for ambiguous `العراقي`. This post-report refinement does not change the submitted PDF or prove the report's stated count of 58. The pre-refinement code/docs snapshot is pushed as commit `c5eaf60` on `codex/progress-report-rerun-backup-20261106`. The broad-filter comparison and audit records remain, but its generated outputs were overwritten by the refined rerun and were not saved separately. The pre-rerun frozen output backups remain under `output/cleaned_cohort/` with the `pre_rerun_frozen_20261106` suffix. The refined comparison is in `FYP2/implementation_walkthrough/REFINED_FILTER_COMPARISON_2026-11-06.md`.

## TF-IDF configuration alignment (2026-11-06)

The text-classification vectorizers in the PO1 benchmark, PO2 experiment suite, and PO3 chain-validation helper now use the Chapter 5 settings: word n-grams `(1, 2)`, `min_df=2`, `max_features=1000`, and sublinear term frequency. The character n-gram TF-IDF used separately for hadith text matching is unchanged. This aligns the current code with the report's documented configuration; saved benchmark outputs were not regenerated, and their producing configuration is not independently recorded in the result JSON. Do not claim this edit proves those historical metrics were produced under the aligned settings.
