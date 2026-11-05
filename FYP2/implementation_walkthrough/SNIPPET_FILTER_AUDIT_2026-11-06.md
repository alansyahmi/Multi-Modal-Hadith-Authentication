# Initial broad substring-filter discrepancy audit

Date: 2026-11-06. This records the first, broad-substring rerun and is historical; the current walkthrough helper and generated cohort outputs use the later refined filter.

## Result

The output difference of 169 statements decomposes exactly into 152 nonempty snippets rejected by `has_late_compiler_leak()` and 17 nonempty snippets of at most five characters rejected by the builder's length rule. The frozen output contains 13,478 statements; the rerun retains 13,309. The rerun's 335 narrator identities are unchanged.

The 152 substring-filtered records contain 110 unique text values and 114 unique narrator/text pairs. Trigger counts (a record can trigger one configured term) are: al-Mizzi 79, al-Ala'i 30, al-Dhahabi substring 24, al-Iraqi substring 18, and al-Nawawi 1. The report's 58 is not reproduced by this code/output comparison. Its denominator or meaning may differ, so these values should not be treated as the same statistic without tracing the original audit.

## False-positive evidence

Manual inspection of the al-Iraqi substring hits found all 18 use `العراقي`/`العراقيين` to refer to Iraqis or Iraqi transmitters, rather than the later compiler al-Iraqi. In the 24 al-Dhahabi substring hits, 6 use `الذهب` in literal-gold phrases such as `مشبك بالذهب`, rather than naming al-Dhahabi. These are confirmed false positives for a compiler-name filter. Other trigger groups still need review.

Therefore the implemented filter is broader than the report's description of removing secondary compiler prefixes. It can delete non-leakage text. The 100% zero-leakage assertion is not established by this substring check alone.

## Audit file and follow-up

[`SNIPPET_FILTER_AUDIT_2026-11-06.csv`](SNIPPET_FILTER_AUDIT_2026-11-06.csv) lists all 169 removed statement records, marks the 152 name hits versus 17 short-text removals, and leaves manual-decision fields blank for review. The two obvious false-positive classes are flagged in `initial_review_note`.

The broad-filter generated outputs were subsequently overwritten by the refined rerun and are not retained as a separate set. The broad-filter code can be recovered from parent commit `c5eaf60`; this report and CSV preserve its recorded comparison and audit. The pre-rerun outputs remain preserved with the `pre_rerun_frozen_20261106` suffix. The later refined outputs and their comparison are documented in [`REFINED_FILTER_COMPARISON_2026-11-06.md`](REFINED_FILTER_COMPARISON_2026-11-06.md).
