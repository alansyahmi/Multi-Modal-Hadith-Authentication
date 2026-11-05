# Pre-rerun versus regenerated preparation outputs

Date: 2026-11-06

Historical note: this comparison describes the initial broad-substring rerun. Its generated cleaned-cohort files were later overwritten by the refined-filter rerun and are not retained separately. The current generated files are compared with the frozen backups in [`REFINED_FILTER_COMPARISON_2026-11-06.md`](REFINED_FILTER_COMPARISON_2026-11-06.md).

## Scope

Only the report-era narrator matcher and cleaned-cohort preparation scripts were run in this progress-report copy. No PO1, PO2, PO3, or model experiments were run. Python UTF-8 mode was enabled because the matcher's Arabic final console print failed under the default Windows CP1252 console encoding after its first output write.

Runtime used: Python 3.13, NumPy 2.3.3, pandas 2.3.3. These are the current rerun environment versions, not evidence of the experiment-time environment in Chapter 5.

## Output comparison

| Artifact | Comparison result |
|---|---|
| `output/hadith_viz_match/matches_cohort.csv` | Exact byte match (300,977 bytes). |
| `output/hadith_viz_match/attrition.json` | Parsed JSON values match (375 bytes). |
| `output/hadith_viz_match/audit_sample.csv` | Same 87 rows and other fields; all 87 `CORRECT(1/0)` manual annotations in the saved sample are blank in the fresh script output. The frozen annotations remain in the renamed backup. |
| `FYP2/attrition_table.md` | Differs. The matcher writes a short entity-resolution summary; the saved file was the five-stage attrition document. Both versions are retained: the saved copy has the `pre_rerun_frozen_20261106` suffix, and the regenerated file is at this path. |
| `output/cleaned_cohort/clean_narrators_cohort.csv` | Both contain 335 identities. Only `earlier_critic_statements_count` differs, for 92 identities; the regenerated cohort has 169 fewer statements in total. |
| `output/cleaned_cohort/clean_texts.json` | Both contain 335 narrator records; 92 records differ after snippet filtering. |
| `output/cleaned_cohort/cohort_summary.json` | Parsed JSON values match. |
| `output/cleaned_cohort/clean_graph.npz` | All logical arrays match, including 335 nodes, 8 feature channels, and 354 directed edge entries. Edge columns were compared as edge sets to ignore ordering. |

The original outputs were renamed in place with `.pre_rerun_frozen_20261106` before running the scripts. The input `output/hadith_viz_match/frozen_evaluation_cohort.csv` was left untouched.

## Interpretation and unresolved issue

The identity resolution and graph outputs reproduce the saved values. The regenerated text does not: the snippet filter removes 169 statements across 92 narrators, whereas the report describes 58 detected secondary quote fragments. This is not evidence that the 169 are all false positives or all true leaks. The removed snippets need a focused audit before claiming that the implementation reproduces the report's stated 58-quote audit.

The fresh matcher sample intentionally starts with blank manual-correctness labels. Do not replace the preserved 87 annotated rows with the blank regenerated copy when using the manual audit evidence.

The attrition Markdown is rewritten by `match_sanadset_to_hadithviz.py`, so the saved five-stage table remains recoverable under its renamed backup. No frozen backup was deleted or overwritten.
