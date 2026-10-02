"""
Stage 2 Pipeline Orchestrator.
"""
from pathlib import Path

import pandas as pd

from .config import (
    STAGE2_DIR,
    ALIGNMENT_RESULTS,
    SANADSET_CLEANED,
    NARRATOR_GRADES,
    GRAPH_NODES_CSV,
    GRAPH_EDGES_CSV,
)
from .graph_builder import build_knowledge_graph
from .graph_exporter import export_to_pyg, write_metadata


def run_stage2(
    alignment_path: Path = ALIGNMENT_RESULTS,
    sanadset_path: Path = SANADSET_CLEANED,
    grades_path: Path = NARRATOR_GRADES,
    skip_graph_export: bool = False,
) -> dict:
    """
    Execute the full Stage 2 pipeline.

    Returns a summary dict.
    """
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Build Knowledge Graph
    kg = build_knowledge_graph(
        alignment_path=alignment_path,
        sanadset_path=sanadset_path,
        grades_path=grades_path,
    )

    # Step 2: Export to PyG format
    if not skip_graph_export:
        nodes_df = pd.read_csv(GRAPH_NODES_CSV)
        edges_df = pd.read_csv(GRAPH_EDGES_CSV)
        export_to_pyg()
        write_metadata(nodes_df, edges_df)

    stats = kg.get_stats()

    print(f"\n{'='*50}")
    print("STAGE 2 COMPLETE")
    print(f"{'='*50}")
    print(f"  Nodes:     {stats['num_nodes']:,}")
    print(f"  Edges:     {stats['num_edges']:,}")
    print(f"  Avg grade: {stats['avg_grade_score']}")
    print(f"  With bio:  {stats['nodes_with_grade']}")

    return stats
