# Implementation Walkthrough

This guide maps the implementation retained in the local `v4` worktree. It links to the actual source files; it does not duplicate them.

## Two execution tracks

The code has a staged prototype and a separate research-evaluation track. They share modules and data, but the benchmark scripts are not just aliases for the three root stage commands.

### Staged prototype

1. **Ingest and align:** [`run_stage1.py`](../../run_stage1.py) connects the dataset loaders, chain parsing, Arabic normalization, hybrid retrieval/alignment, and quarantine output.
2. **Build the graph:** [`run_stage2.py`](../../run_stage2.py) calls [`src/stage2/pipeline.py`](../../src/stage2/pipeline.py), which builds narrator nodes and transmission edges and exports graph files.
3. **Score narrators:** [`run_stage3.py`](../../run_stage3.py) calls [`src/stage3/pipeline.py`](../../src/stage3/pipeline.py), then the GNN model and trainer modules, and writes predictions and evaluation artifacts.

### Report benchmark track

- PO1 text benchmark: [`scripts/run_po1_benchmark.py`](../../scripts/run_po1_benchmark.py).
- PO2 graph and multimodal benchmark: [`scripts/build_clean_dataset.py`](../../scripts/build_clean_dataset.py) and [`scripts/run_clean_experiments.py`](../../scripts/run_clean_experiments.py).
- PO3 exploratory result in the selected report: [`scripts/evaluate_independent_po3_candidate.py`](../../scripts/evaluate_independent_po3_candidate.py), with its saved candidate inputs and output under `output/corrected_candidate_20261029/`.
- Shared label/text preparation and statistics: [`src/canonical_data_pipeline.py`](../../src/canonical_data_pipeline.py) and [`src/evaluation_stats.py`](../../src/evaluation_stats.py).

## Read next

See the [code map](CODE_MAP.md) for module responsibilities and output locations. Use the [bundle manifest](../V4_BUNDLE_MANIFEST.md) to understand which files were drawn from the snapshot and why this is not certified as the exact submitted source state.

## Explanation boundary

The implementation estimates narrator reliability or predicts narrator tiers. It does not issue an independent hadith-authenticity verdict. Explain PO3 as exploratory and keep the separate staged prototype distinct from the report benchmark scripts.

## Slide data preparation and temporal firewall

The slide's data-preparation box is represented across the matching and cohort-building steps, with the documented attrition counts in [`FYP2/attrition_table.md`](../attrition_table.md). The report-era matching step is [`scripts/match_sanadset_to_hadithviz.py`](../../scripts/match_sanadset_to_hadithviz.py): it normalizes names, links graph nodes to Hadith-Viz candidates, uses tabaqa/death-year and graph overlap to resolve some ambiguous matches, and writes match/audit outputs. The cohort builder is [`scripts/build_clean_dataset.py`](../../scripts/build_clean_dataset.py): it loads the matched cohort, Hadith-Viz records, and graph nodes/edges; removes ineligible records; keeps records with qualifying earlier-critic text; maps identities to graph nodes; and writes the cleaned narrator cohort. Together, the matcher and cohort builder cover the entity matching, eligibility filtering, graph intersection, and identity consolidation steps represented in the funnel. The attrition table records the counts; this code copy has not been rerun to regenerate them. **Do not run the matcher casually:** it rewrites `FYP2/attrition_table.md` as part of its outputs.

The report-described firewall behavior is represented in [`src/canonical_data_pipeline.py`](../../src/canonical_data_pipeline.py): `is_earlier_critic()` rejects names matching the configured later-author exclusion list, and `has_late_compiler_leak()` rejects snippets containing those configured names. The snippet check is a simple name-substring screen; it does not establish date eligibility for every retained author or detect every possible indirect citation or paraphrase. Both `scripts/build_clean_dataset.py` and the PO1 text benchmark use these checks through the shared pipeline.

**Version boundary:** the helper was recovered from the forensic snapshot and restored to the walkthrough copy in report-described form. The October 17 change log and October 30 report support that this behavior predated submission, but the exact submitted source bytes are not proven. See [`CODEBASE_MANIFEST.md`](../../CODEBASE_MANIFEST.md).
