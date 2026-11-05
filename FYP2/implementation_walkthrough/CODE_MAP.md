# Code Map

| Task | Entry point | Main implementation | Inputs and outputs |
| --- | --- | --- | --- |
| Stage 1 ingestion/alignment | [`run_stage1.py`](../../run_stage1.py) | [`src/data/`](../../src/data/), [`src/preprocessing/`](../../src/preprocessing/), [`src/alignment/hybrid_aligner.py`](../../src/alignment/hybrid_aligner.py) | Reads SanadSet and hadith data; writes aligned records and quarantine under `output/stage1/` |
| Stage 2 graph | [`run_stage2.py`](../../run_stage2.py) | [`src/stage2/pipeline.py`](../../src/stage2/pipeline.py), [`graph_builder.py`](../../src/stage2/graph_builder.py), [`graph_exporter.py`](../../src/stage2/graph_exporter.py) | Reads alignment and narrator-grade inputs; writes node/edge tables and graph files under `output/stage2/` |
| Stage 3 GNN | [`run_stage3.py`](../../run_stage3.py) | [`src/stage3/gnn_model.py`](../../src/stage3/gnn_model.py), [`src/stage3/trainer.py`](../../src/stage3/trainer.py) | Loads graph tensors; writes checkpoint, predictions, and evaluation metrics under `output/stage3/` |
| PO1 text benchmark | [`scripts/run_po1_benchmark.py`](../../scripts/run_po1_benchmark.py) | [`src/canonical_data_pipeline.py`](../../src/canonical_data_pipeline.py) | Uses Hadith-Viz critic text and saved AraBERT embeddings; writes PO1 folds and summary in `output/cleaned_cohort/` |
| Report Stage 1 data preparation | [`scripts/match_sanadset_to_hadithviz.py`](../../scripts/match_sanadset_to_hadithviz.py), [`scripts/build_clean_dataset.py`](../../scripts/build_clean_dataset.py) | [`src/canonical_data_pipeline.py`](../../src/canonical_data_pipeline.py) | Matches graph narrators to Hadith-Viz, applies the documented eligibility/firewall checks, consolidates identities, and prepares the PO1/PO2 cohorts. The matcher rewrites `FYP2/attrition_table.md` if run. |
| PO2 graph/multimodal benchmark | [`scripts/build_clean_dataset.py`](../../scripts/build_clean_dataset.py), [`scripts/run_clean_experiments.py`](../../scripts/run_clean_experiments.py) | [`src/evaluation_stats.py`](../../src/evaluation_stats.py) | Uses cleaned cohort, graph, text, and embedding inputs; writes per-fold and summary JSON in `output/cleaned_cohort/` |
| PO3 exploratory analysis | [`scripts/evaluate_independent_po3_candidate.py`](../../scripts/evaluate_independent_po3_candidate.py) | Candidate evaluation logic in that script | Uses SanadSet, alignment, graph nodes, candidate cohort, and inductive predictions; writes `po3_independent_candidate.json` |

## Version notes

- The report-linked PO3 candidate is a separately dated candidate workflow. Its inclusion supports the report's exploratory statistic and does not make PO3 validated.
- The v4 report and current code are linked by the bundle manifest's timestamp rule, not by a preserved source hash from the actual experiment run.
- The root stage CLIs have a default alignment filename mismatch between Stage 1 and Stage 2. Inspect configured filenames before describing them as a verified consecutive run.

