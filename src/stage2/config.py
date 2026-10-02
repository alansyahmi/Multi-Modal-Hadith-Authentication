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

# ── Ibn Hajar 12-tier grade → numeric score (1.0 = perfect, 0.0 = rejected) ──
GRADE_SCORE_MAP: dict[str, float] = {
    # Rank 1: Companions (inherently trustworthy)
    "صحابي": 1.0,
    # Rank 2: Highest praise
    "ثقة ثبت": 0.95,
    "ثقة حافظ": 0.95,
    "ثقة متقن": 0.95,
    "إمام": 0.95,
    "حافظ": 0.90,
    "ثبت": 0.90,
    "متقن": 0.90,
    # Rank 3: Trustworthy
    "ثقة": 0.85,
    "ثقات": 0.85,
    "مأمون": 0.85,
    # Rank 4: Truthful
    "صدوق": 0.70,
    "لا بأس به": 0.65,
    "ليس به بأس": 0.65,
    "محله الصدق": 0.65,
    # Rank 5: Truthful but flawed
    "صدوق يهم": 0.55,
    "صدوق سيء الحفظ": 0.50,
    "سيء الحفظ": 0.40,
    # Rank 6: Acceptable only with corroboration
    "مقبول": 0.40,
    # Rank 7: Hidden / unknown state
    "مستور": 0.25,
    "مجهول الحال": 0.20,
    # Rank 8: Weak
    "ضعيف": 0.15,
    "ضعيف الحديث": 0.15,
    "لين": 0.15,
    "لين الحديث": 0.15,
    "ليس بالقوي": 0.15,
    "فيه ضعف": 0.15,
    "منكر الحديث": 0.10,
    # Rank 9: Completely unknown
    "مجهول": 0.10,
    "مجهول العين": 0.05,
    # Rank 10: Abandoned
    "متروك": 0.05,
    "متروك الحديث": 0.05,
    "ساقط": 0.05,
    "مختلط": 0.05,
    # Rank 11: Accused of lying
    "متهم": 0.0,
    # Rank 12: Fabricator
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
