"""
Arabic text normalisation pipeline for classical Hadith texts.

Provides consistent text cleaning used across the data ingestion pipeline:
diacritic removal, character normalisation, and consonantal skeleton extraction.

References
----------
- CAMeL Tools (https://github.com/CAMeL-Lab/camel_tools)
- PyArabic (https://github.com/linuxscout/pyArabic)
"""

import re
from typing import Optional

from src.config import DIACRITICS


# ── Character normalisation maps ──────────────────────────────
# Normalise alif variants to bare alif
ALIF_MAP = str.maketrans({
    "آ": "ا",  # alif with madd
    "أ": "ا",  # alif with hamza above
    "إ": "ا",  # alif with hamza below
    "ٱ": "ا",  # alif wasl
})

# Normalise ya and ha variants
YAH_MAP = str.maketrans({
    "ى": "ي",  # alif maqsura -> ya
    "ة": "ه",  # taa marbouta -> ha
})

# Spaces and punctuation to collapse/remove
PUNCTUATION_PATTERN = re.compile(r"[؀-؅،؍؛؟٪-٭۔]+")


def strip_diacritics(text: str) -> str:
    """
    Remove Arabic diacritics (tashkeel).

    Removes: fatha, damma, kasra, sukun, tanwin, shadda.
    Shadda is also removed (the gemination mark is a diacritic).

    Parameters
    ----------
    text : Input Arabic string.

    Returns
    -------
    String with diacritics removed.
    """
    if not isinstance(text, str):
        return ""
    return text.translate(DIACRITICS)


def normalise_alif(text: str) -> str:
    """
    Normalise alif variants (آ, أ, إ, ٱ) to bare alif (ا).

    Classical Hadith texts use multiple alif forms interchangeably;
    this normalises them for consistent matching.
    """
    return text.translate(ALIF_MAP)


def normalise_ya_ha(text: str) -> str:
    """
    Normalise ya and ha variants:
      - alif maqsura (ى) -> ya (ي)
      - taa marbouta (ة) -> ha (ه)
    """
    return text.translate(YAH_MAP)


def remove_non_arabic(text: str) -> str:
    """
    Remove non-Arabic characters, keeping only Arabic letters and whitespace.

    Arabic Unicode block: ؀-ۿ
    Arabic Supplement:   ݐ-ݿ
    Arabic Extended-A:   ࢠ-ࣿ
    """
    arabic_pattern = re.compile(r"[^؀-ۿݐ-ݿࢠ-ࣿ\s]")
    return arabic_pattern.sub("", text)


def collapse_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def get_consonantal_skeleton(text: str) -> str:
    """
    Compute the consonantal skeleton of an Arabic string.

    This is the *rasm* (the bare consonantal outline without diacritics):
    1. Strip diacritics
    2. Normalise alif variants
    3. Normalise ya/ha variants
    4. Remove non-Arabic characters and punctuation
    5. Collapse whitespace

    The skeleton is used as the primary key for fuzzy matching and entity
    disambiguation, as it is robust to orthographic variation across
    manuscript traditions (see Section 3.2.3).
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    result = text.strip()
    result = strip_diacritics(result)
    result = normalise_alif(result)
    result = normalise_ya_ha(result)
    result = remove_non_arabic(result)
    result = collapse_whitespace(result)
    return result


def normalise_hadith_text(text: str, for_matching: bool = False) -> str:
    """
    Full normalisation pipeline for Hadith/matn text.

    Parameters
    ----------
    text         : Input Arabic string.
    for_matching : If True, also remove punctuation for TF-IDF matching.

    Returns
    -------
    Normalised text string.
    """
    result = get_consonantal_skeleton(text)
    if for_matching:
        result = PUNCTUATION_PATTERN.sub("", result)
    return result


# ── Quick-start (when run directly) ───────────────────────────
if __name__ == "__main__":
    test_cases = [
        "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
        "حَدَّثَنَا أَبُو عَبْدِ اللَّهِ الْحُمَيْدِيُّ",
        "آمَنَ الرَّسُولُ بِمَا أُنْزِلَ إِلَيْهِ",
        "سَعِيدِ بْنِ زَيْدٍ",
    ]

    print("=== Arabic Normalisation Test ===")
    for text in test_cases:
        print(f"\n  Original:   {text}")
        print(f"  Skeleton:   {get_consonantal_skeleton(text)}")
        print(f"  For match:  {normalise_hadith_text(text, for_matching=True)}")
