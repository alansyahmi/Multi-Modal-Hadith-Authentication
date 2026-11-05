# FYP2 V4 Report-Centered Bundle

This local `v4` worktree collects the selected submitted report, the core implementation, report-related data, and saved results in one navigable place.

## Start here

1. Read the [implementation walkthrough](FYP2/implementation_walkthrough/README.md).
2. Use its [code map](FYP2/implementation_walkthrough/CODE_MAP.md) to follow entrypoints into `src/`.
3. Review the [bundle manifest](FYP2/V4_BUNDLE_MANIFEST.md) for version and reproduction limits.
4. Open the selected [v4 report](FYP2/report_draft/fyp2_report_v4_final.pdf).

## Main paths

- `run_stage1.py`, `run_stage2.py`, `run_stage3.py`: staged prototype entrypoints.
- `src/`: reusable data, preprocessing, alignment, graph, model, and training modules.
- `scripts/`: the report-linked benchmark and embedding scripts retained for this bundle.
- `Dataset/`: input data copied from the forensic snapshot.
- `output/`: selected prepared inputs and saved results for the report.
- `FYP2/`: report records, traceability, figures, and the implementation walkthrough.

This is a report-centered reconstruction, not a verified bit-for-bit copy of the project as it existed at submission. The manifest explains the source cutoff and the known lineage gaps.
