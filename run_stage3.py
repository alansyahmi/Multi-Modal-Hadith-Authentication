#!/usr/bin/env python3
"""
Stage 3: Probabilistic Trust Scoring Using Graph Neural Networks.

Trains a Graph Convolutional Network (GCN) to predict narrator reliability
scores from graph structure and node features. Evaluates against Ibn Hajar
ground-truth grades.

Usage:
  python run_stage3.py                  # Train GCN, evaluate
  python run_stage3.py --model gat      # Use Graph Attention Network
  python run_stage3.py --epochs 300     # More training epochs
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.stage3.pipeline import run_stage3


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3: GNN Reliability Scoring"
    )
    parser.add_argument(
        "--model", choices=["gcn", "gat"], default="gcn",
        help="GNN model type (default: gcn)",
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Maximum training epochs (default: 200)",
    )
    parser.add_argument(
        "--patience", type=int, default=30,
        help="Early stopping patience (default: 30)",
    )
    args = parser.parse_args()

    run_stage3(
        model_type=args.model,
        num_epochs=args.epochs,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
