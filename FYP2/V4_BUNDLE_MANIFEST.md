# V4 Report-Centered Bundle Manifest

Created: 2026-11-06
Branch: `v4`
Base commit: `48dbfeaf033cefa6f39a19f2ae45a134d42f8a71` (2026-10-06)
Selected report: `FYP2/report_draft/fyp2_report_v4_final.pdf`
Recorded snapshot SHA-256: `783D64523795CA3C29D6F8C837DC4DA25E8E282E1856A2172D7F1BAF630D8B96`

## What this bundle represents

This is a report-centered reconstruction of the selected v4 report and its implementation evidence. It is not a bit-for-bit source release from the report submission date. The selected PDF was identified by the user; the archive does not contain an external submission receipt.

The Git base commit predates the report. For retained source files under `src/`, `scripts/`, and the root stage commands, the snapshot's working-tree copy was overlaid only when its recorded modification time was at or before the selected PDF's recorded time, 2026-10-30 17:00:01 +08:00. Later source changes remain at the base commit version. This timestamp rule is a reconstruction aid, not proof that those exact bytes generated the report.

## Included materials

- Core package under `src/`, root stage commands, and the report-linked benchmark scripts.
- The selected report PDF and the baseline FYP2 documentation retained from the base commit.
- Raw files under `Dataset/`, `References/`, and `references_shamela/` from the forensic snapshot.
- Saved inputs and outputs under `output/stage1/`, `output/stage2/`, `output/hadith_viz_match/`, and the report-relevant portion of `output/cleaned_cohort/`.
- The PO3 exploratory producer `scripts/evaluate_independent_po3_candidate.py` and its supporting candidate cohort, predictions, and saved result under `output/corrected_candidate_20261029/`. This candidate evidence is explicitly cross-version support for the report's 181-record exploratory result.
- Snapshot-only `src/data/` loader modules are included because the retained Stage 1 entry point imports them. Their recorded snapshot modification times predate the selected report, but they are absent from the Git base commit; their inclusion is snapshot evidence, not base-commit content.
- `FYP2/V4_PRUNE_MANIFEST.csv`, which lists the 412 committed paths removed from this worktree and why they were scoped out.

The post-report review-gated PO3 artifacts, v5B work, alternative v4 PDF candidates, broad unrelated output folders, and temporary material are not included in this worktree. They remain in the original project or the forensic snapshot.

## Reproduction limits

- The v4 report's PO1/PO2 results do not have complete source-to-output hashes or preserved run wrappers.
- The PO3 number relies on an exploratory candidate workflow; it is not confirmatory validation.
- The experiment-time Python/package environment is not pinned by a contemporaneous lock file.
- Dataset inclusion and exact source editions are not fully linked to each reported run.
- The original forensic snapshot remains `PARTIALLY VERIFIED - EXCEPTION DOCUMENTED`; this branch does not change that status.

Treat this as a local, report-centered working bundle. Do not describe it as a verified exact reproduction of the submitted software environment.

- The pre-report snapshot-only src/data/ loaders are included because the retained Stage 1 entry point imports them. Their recorded local modification times predate the selected report; they are not present in the Git base commit, so this inclusion is disclosed as snapshot evidence rather than base-commit content.
