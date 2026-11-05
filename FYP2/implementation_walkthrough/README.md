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
