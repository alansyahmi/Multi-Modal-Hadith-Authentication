"""
Stage 2 configuration constants.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
STAGE2_DIR = OUTPUT_DIR / "stage2"

# ── Input paths ──
ALIGNMENT_RESULTS = OUTPUT_DIR / "stage1" / "alignment_results_1200.json"
EVALUATION_CSV = OUTPUT_DIR / "stage1" / "kutub_sitta_evaluation_1200.csv"
SANADSET_CLEANED = OUTPUT_DIR / "stage1" / "sanadset_cleaned.csv"
NARRATOR_GRADES = STAGE2_DIR / "narrator_grades.json"

# ── Output paths ──
GRAPH_NODES_CSV = STAGE2_DIR / "graph_nodes.csv"
GRAPH_EDGES_CSV = STAGE2_DIR / "graph_edges.csv"
GRAPH_METADATA_JSON = STAGE2_DIR / "graph_metadata.json"
GRAPH_PYG = STAGE2_DIR / "pyg_graph.pt"

# ── Ibn Hajar 12-tier grade → numeric score ──
# Standardized Parameter-Free Min-Max Rank Normalization:
# S(r) = (12 - r) / 11, where r in {1, 2, ..., 12}
# Defensibility: Parameter-free, objective, bounds Rank 1 (Companions) to 1.000
# and Rank 12 (Fabricators) strictly to 0.000.
GRADE_SCORE_MAP: dict[str, float] = {
    # Rank 1: Companions (11/11 = 1.000)
    "صحابي": 1.0,
    # Rank 2: Highest praise (10/11 = 0.909)
    "ثقة ثبت": 0.909,
    "ثقة حافظ": 0.909,
    "ثقة متقن": 0.909,
    "إمام": 0.909,
    "حافظ": 0.909,
    "ثبت": 0.909,
    "متقن": 0.909,
    # Rank 3: Trustworthy (9/11 = 0.818)
    "ثقة": 0.818,
    "ثقات": 0.818,
    "مأمون": 0.818,
    # Rank 4: Truthful (8/11 = 0.727)
    "صدوق": 0.727,
    "لا بأس به": 0.727,
    "ليس به بأس": 0.727,
    "محله الصدق": 0.727,
    # Rank 5: Truthful but flawed (7/11 = 0.636)
    "صدوق يهم": 0.636,
    "صدوق سيء الحفظ": 0.636,
    "سيء الحفظ": 0.636,
    # Rank 6: Acceptable only with corroboration (6/11 = 0.545)
    "مقبول": 0.545,
    # Rank 7: Hidden / unknown state (5/11 = 0.455)
    "مستور": 0.455,
    "مجهول الحال": 0.455,
    # Rank 8: Weak (4/11 = 0.364)
    "ضعيف": 0.364,
    "ضعيف الحديث": 0.364,
    "لين": 0.364,
    "لين الحديث": 0.364,
    "ليس بالقوي": 0.364,
    "فيه ضعف": 0.364,
    "منكر الحديث": 0.364,
    # Rank 9: Completely unknown (3/11 = 0.273)
    "مجهول": 0.273,
    "مجهول العين": 0.273,
    # Rank 10: Abandoned (2/11 = 0.182)
    "متروك": 0.182,
    "متروك الحديث": 0.182,
    "ساقط": 0.182,
    "مختلط": 0.182,
    # Rank 11: Accused of lying (1/11 = 0.091)
    "متهم": 0.091,
    # Rank 12: Fabricator (0/11 = 0.000)
    "كذاب": 0.0,
    "وضاع": 0.0,
    "دجال": 0.0,
}

# Unknown/missing grade fallback
DEFAULT_GRADE_SCORE = 0.30

# ── Edge feature encoding ──
# Transmission method → one-hot index
METHOD_ENCODING: dict[str, int] = {
    "direct": 0,
    "ambiguous": 1,
    "unknown": 2,
}

# ── Chain position features ──
MAX_CHAIN_LENGTH = 30  # Truncate longer chains

# ── Chunk size for processing ──
CHUNK_SIZE = 10_000
