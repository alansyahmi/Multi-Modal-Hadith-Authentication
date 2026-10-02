#!/usr/bin/env python3
"""
Stage 2: Named Entity Recognition and Knowledge Graph Construction.

Builds a directed knowledge graph of hadith narrators from the 1,030
aligned Sanadset chains, enriched with Ibn Hajar reliability grades
from Taqrib al-Tahdhib and biographical data from Tahdhib al-Kamal.

Output:
  output/stage2/
    graph_nodes.csv          Node table with features and bio data
    graph_edges.csv          Edge table with transmission attributes
    graph_metadata.json      Schema and statistics
    pyg_graph.pt             PyTorch Geometric Data object (optional)

Usage:
  python run_stage2.py                   # Full pipeline
  python run_stage2.py --skip-export     # Skip PyG export
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.stage2.pipeline import run_stage2
from src.stage2.config import STAGE2_DIR
from src.config import STAGE1_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: NER and Knowledge Graph Construction"
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip PyTorch Geometric export (output CSV only)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        choices=[1200, 5000],
        default=1200,
        help="Sample size used in Stage 1 (determines which alignment file to read)",
    )
    args = parser.parse_args()

    STAGE2_DIR.mkdir(parents=True, exist_ok=True)

    alignment_path = STAGE1_DIR / f"alignment_results_{args.sample_size}.json"
    stats = run_stage2(
        skip_graph_export=args.skip_export,
        alignment_path=alignment_path,
    )

    if stats:
        print(f"\n  Output directory: {STAGE2_DIR}")


if __name__ == "__main__":
    main()
