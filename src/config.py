"""
Project configuration: paths, constants, and parameters for Stage 1-3 pipelines.
All paths are relative to the project root.
"""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Dataset paths ─────────────────────────────────────────────
DATASET_DIR = PROJECT_ROOT / "Dataset"

# Sanadset 650K
SANADSET_DIR = DATASET_DIR / "Sanadset 650K Data on Hadith Narrators"
# Note: nested subdirectory with same name
SANADSET_DATA_DIR = SANADSET_DIR / "Sanadset 650K Data on Hadith Narrators"
SANADSET_CSV = SANADSET_DATA_DIR / "sanadset.csv"
BOOKS_CSV = SANADSET_DATA_DIR / "books.csv"
SANADSET_SAMPLES_CSV = SANADSET_DATA_DIR / "hadith_samples.csv"

# HuggingFace Kutub as-Sitta (ground truth with Grade column)
HF_DIR = DATASET_DIR / "hadith_datasets HuggingFace"
HF_FILES = {
    "bukhari": HF_DIR / "Sahih%20al-Bukhari.csv",
    "muslim": HF_DIR / "Sahih%20Muslim.csv",
    "abudawud": HF_DIR / "Sunan20Dawud.csv",
    "tirmidhi": HF_DIR / "Jami20at-Tirmidhi.csv",
    "nasai": HF_DIR / "Sunan_an-Nasai.csv",
    "ibnmajah": HF_DIR / "Sunan20Majah.csv",
}

# ── Output paths ──────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "output"
STAGE1_DIR = OUTPUT_DIR / "stage1"
STAGE1_DIR.mkdir(parents=True, exist_ok=True)

# ── Stage 1: Alignment parameters ─────────────────────────────
# Tier 1: TF-IDF retrieval
TFIDF_NGRAM_RANGE = (3, 5)          # character n-grams
TFIDF_MAX_FEATURES = 50_000         # vocabulary cap
TOP_K_CANDIDATES = 10                # candidates retrieved per query

# Tier 2: Fuzzy matching
FUZZY_THRESHOLD = 95.0               # minimum similarity ratio (%)

# Stratified sampling per sample size (overshoot for alignment filtering)
# 1,200 Book-Balanced: 300 × 6 = 1,800 queries → target 1,200 aligned
# 6,000 Book-Balanced: 1,166 × 6 = 6,996 queries → target 6,000 aligned
STRATIFIED_SAMPLE_PER_BOOK = {1200: 300, 6000: 1166}

# Grade-balanced sampling per sample size (capped by available graded records)
# 1,200 Grade-Balanced: target 450 × 4 = 1,800 → actual ~1,378 (limited by Hasan)
# 6,000 Grade-Balanced: target 1750 × 4 = 7,000 → actual ~5,278 (limited by Hasan)
BALANCED_SAMPLE_PER_GRADE = {1200: 450, 6000: 1750}

# Reproducibility
RANDOM_SEED = 42

# ── Arabic normalisation constants ────────────────────────────
# Characters to strip during normalisation
DIACRITICS = str.maketrans("", "", "ًٌٍَُِّْ")  # Arabic diacritics (tashkeel)

# ── Shamela extraction paths ───────────────────────────────────
SHAMELA_ISO_PATH: Path = PROJECT_ROOT / "shamela-win-1447.11.iso"
SHAMELA_OUTPUT_DIR: Path = OUTPUT_DIR  # Per-book dirs created at runtime

# Book registry: known book IDs → slug, title, author
SHAMELA_BOOKS: dict[int, dict[str, str]] = {
    3722: {
        "slug": "tahdhib_al_kamal",
        "title": "تهذيب الكمال في أسماء الرجال",
        "author": "المزي",
    },
    1299: {
        "slug": "taqrib_al_tahdhib",
        "title": "تقريب التهذيب",
        "author": "ابن حجر العسقلاني",
    },
    1222: {
        "slug": "tahdhib_al_tahdhib",
        "title": "تهذيب التهذيب",
        "author": "ابن حجر العسقلاني",
    },
}

# ── Data columns ──────────────────────────────────────────────
SANADSET_COLUMNS = ["Hadith", "Book", "Num_hadith", "Matn", "Sanad", "Sanad_Length"]
HF_COLUMNS = [
    "Book", "Chapter_Number", "Chapter_Title_Arabic", "Chapter_Title_English",
    "Arabic_Text", "English_Text", "Grade", "Reference", "In-book reference",
]
