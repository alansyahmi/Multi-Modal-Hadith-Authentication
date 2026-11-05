# Code Map

| Task | Entry point | Main implementation | Main outputs |
| --- | --- | --- | --- |
| Stage 1 ingestion and alignment | [run_stage1.py](../run_stage1.py) | [src/data/](../src/data/), [src/preprocessing/](../src/preprocessing/), [hybrid_aligner.py](../src/alignment/hybrid_aligner.py) | Alignment, evaluation sample, and quarantine under output/stage1/. |
| Stage 2 graph construction | [run_stage2.py](../run_stage2.py) | [pipeline.py](../src/stage2/pipeline.py), [graph_builder.py](../src/stage2/graph_builder.py), [graph_exporter.py](../src/stage2/graph_exporter.py) | Node and edge tables, metadata, and graph tensors under output/stage2/. |
| Stage 3 GNN | [run_stage3.py](../run_stage3.py) | [gnn_model.py](../src/stage3/gnn_model.py), [trainer.py](../src/stage3/trainer.py) | Checkpoint, predictions, and evaluation under output/stage3/. |
| PO1 | [run_po1_benchmark.py](../scripts/run_po1_benchmark.py) | [canonical_data_pipeline.py](../src/canonical_data_pipeline.py) | PO1 folds and summary under output/cleaned_cohort/. |
| PO2 | [build_clean_dataset.py](../scripts/build_clean_dataset.py), [run_clean_experiments.py](../scripts/run_clean_experiments.py) | [evaluation_stats.py](../src/evaluation_stats.py) | Clean cohort and controlled experiment results under output/cleaned_cohort/. |
| PO3 exploratory analysis | [evaluate_independent_po3_candidate.py](../scripts/evaluate_independent_po3_candidate.py) | Candidate evaluation logic in that script | Exploratory result under output/corrected_candidate_20261029/. |

The Stage 1 and Stage 2 default alignment filenames differ in the retained v4 code. See the walkthrough and the v4 bundle manifest before attempting reproduction.
