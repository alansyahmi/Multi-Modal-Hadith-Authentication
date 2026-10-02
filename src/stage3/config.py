"""
Stage 3: GNN hyperparameters and configuration.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
STAGE3_DIR = OUTPUT_DIR / "stage3"

# ── Input from Stage 2 ──
GRAPH_NPZ = OUTPUT_DIR / "stage2" / "pyg_graph_large_graded.npz"
GRAPH_NODES_CSV = OUTPUT_DIR / "stage2" / "graph_nodes.csv"
GRAPH_EDGES_CSV = OUTPUT_DIR / "stage2" / "graph_edges.csv"

# ── Matn text source ──
SANADSET_CLEANED = OUTPUT_DIR / "stage1" / "sanadset_cleaned.csv"
ALIGNMENT_RESULTS = OUTPUT_DIR / "stage1" / "alignment_results.json"

# ── GNN hyperparameters ──
HIDDEN_CHANNELS = 64
NUM_LAYERS = 3
DROPOUT = 0.3
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 200
PATIENCE = 30  # Early stopping

# ── Data split ──
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15   # test = 1 - train - val

# ── Random seed ──
RANDOM_SEED = 42

# ── Output ──
MODEL_CHECKPOINT = STAGE3_DIR / "gnn_model.pt"
PREDICTIONS_CSV = STAGE3_DIR / "narrator_predictions.csv"
EVALUATION_JSON = STAGE3_DIR / "evaluation_results.json"
