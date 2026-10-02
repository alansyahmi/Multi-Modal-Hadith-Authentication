"""
Stage 3 Pipeline Orchestrator — GNN Training and Evaluation.
"""
from pathlib import Path

from .config import STAGE3_DIR, GRAPH_NPZ
from .trainer import load_graph, train, save_results


def run_stage3(
    model_type: str = "gcn",
    num_epochs: int = 200,
    patience: int = 30,
) -> dict:
    """
    Execute the full Stage 3 pipeline: GNN training + evaluation.

    Args:
        model_type: "gcn" or "gat"
        num_epochs: Maximum training epochs
        patience: Early stopping patience

    Returns:
        Evaluation results dict.
    """
    STAGE3_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("STAGE 3: GNN RELIABILITY SCORING")
    print("=" * 50)

    # Load graph
    if not GRAPH_NPZ.exists():
        raise FileNotFoundError(
            f"Stage 2 graph not found: {GRAPH_NPZ}. Run Stage 2 first."
        )
    data = load_graph(GRAPH_NPZ)

    # Train
    model, results = train(
        data,
        model_type=model_type,
        num_epochs=num_epochs,
        patience=patience,
        verbose=True,
    )

    # Save
    save_results(model, results)

    print(f"\n{'='*50}")
    print("STAGE 3 COMPLETE")
    print(f"{'='*50}")

    return results
