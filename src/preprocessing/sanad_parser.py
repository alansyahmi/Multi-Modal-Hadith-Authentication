"""
Sanad (chain of transmission) parser.

Extracts structured narrator chains from both the pre-extracted Sanad column
and the raw <NAR> tags in the Hadith text. Provides cross-validation between
both sources and generates cleaned, structured chain representations.

Two extraction methods are supported:
  1. From the pre-extracted Sanad column (Python-list string)
  2. From <NAR> tags embedded in the raw Hadith text
"""

import re
import ast
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from src.data.sanadset_loader import parse_sanad_column, parse_nar_tags


# ── Transmission term patterns ────────────────────────────────
# Terms that indicate direct (explicit) transmission
DIRECT_TRANSMISSION = re.compile(
    r"\b(حَدَّثَنَا|حَدَّثَنِي|أَخْبَرَنَا|أَخْبَرَنِي|سَمِعْتُ|سَمِعَ|أَنْبَأَنَا)\b"
)

# Terms that indicate ambiguous transmission
AMBIGUOUS_TRANSMISSION = re.compile(r"\b(عَنْ|أَنَّ|أَنَّهُ|عَن)\b")


def extract_transmission_method(sanad_text: str) -> List[str]:
    """
    Extract the transmission method between consecutive narrators.

    Parses the raw sanad text to determine whether each link is:
      - 'direct' (sami'a, haddathana, akhbarana)
      - 'ambiguous' (an, anna)
      - 'unknown' (none of the above)

    Parameters
    ----------
    sanad_text : The raw sanad portion of the Hadith (including <NAR> tags).

    Returns
    -------
    List of transmission method strings, one per edge (len = narrators - 1).
    """
    if pd.isna(sanad_text) or not sanad_text.strip():
        return []

    # Remove NAR tags for cleaner matching
    cleaned = re.sub(r"</?NAR[^>]*>", "", sanad_text)

    # Find all transmission terms in order
    methods = []
    # We'll look at the text between consecutive narrator mentions
    nar_positions = [(m.start(), m.end()) for m in re.finditer(r"</NAR>", sanad_text)]

    for i in range(len(nar_positions) - 1):
        between = sanad_text[nar_positions[i][1]:nar_positions[i + 1][0]]

        if DIRECT_TRANSMISSION.search(between):
            methods.append("direct")
        elif AMBIGUOUS_TRANSMISSION.search(between):
            methods.append("ambiguous")
        else:
            methods.append("unknown")

    return methods


def cross_validate_narrators(
    row: pd.Series,
) -> Tuple[List[str], List[str], bool]:
    """
    Cross-validate narrators from the Sanad column vs <NAR> tag extraction.

    Parameters
    ----------
    row : A row from the Sanadset DataFrame.

    Returns
    -------
    (sanad_col_narrators, nar_tag_narrators, match)
      - sanad_col_narrators: narrators parsed from the 'Sanad' column.
      - nar_tag_narrators:   narrators extracted from <NAR> tags.
      - match:               True if both lists are identical.
    """
    col_narrators = parse_sanad_column(row.get("Sanad", ""))
    tag_narrators = parse_nar_tags(row.get("Hadith", ""))
    match = col_narrators == tag_narrators
    return col_narrators, tag_narrators, match


def validate_dataset_narrators(
    df: pd.DataFrame,
    sample_frac: float = 0.01,
) -> dict:
    """
    Validate narrator extraction consistency on a sample of the dataset.

    Parameters
    ----------
    df           : Sanadset DataFrame.
    sample_frac  : Fraction of rows to validate (default 1%).

    Returns
    -------
    stats dict with match rate, total checked, etc.
    """
    sample = df.sample(frac=sample_frac, random_state=42)
    matches = 0
    total = len(sample)
    mismatches = []

    for idx, row in tqdm(sample.iterrows(), total=total, desc="Validating"):
        _, _, match = cross_validate_narrators(row)
        if match:
            matches += 1
        else:
            mismatches.append(idx)
            if len(mismatches) >= 5:  # Collect only first 5 examples
                pass

    stats = {
        "total_checked": total,
        "match_count": matches,
        "match_rate": matches / total if total > 0 else 0,
        "mismatch_examples": len(mismatches),
    }

    print(f"\n=== Narrator Cross-Validation ===")
    print(f"  Rows checked:    {total:,}")
    print(f"  Match rate:      {stats['match_rate']:.2%}")
    print(f"  Mismatches:      {total - matches:,}")

    if mismatches:
        print(f"\n  First mismatch example (row {mismatches[0]}):")
        row = df.loc[mismatches[0]]
        col_n, tag_n, _ = cross_validate_narrators(row)
        print(f"    Sanad column: {col_n}")
        print(f"    <NAR> tags:   {tag_n}")

    return stats


def build_chain_record(row: pd.Series) -> dict:
    """
    Build a structured chain record from a Sanadset row.

    The output dict can be serialised to JSON for the Knowledge Graph (Stage 2).
    """
    col_narrators, tag_narrators, _ = cross_validate_narrators(row)

    # Prefer the Sanad column, fall back to NAR tags
    narrators = col_narrators if col_narrators else tag_narrators
    method = extract_transmission_method(row.get("Hadith", ""))

    return {
        "hadith_id": row.get("Num_hadith"),
        "book": row.get("Book"),
        "narrator_chain": narrators,
        "chain_length": len(narrators),
        "transmission_methods": method,
        "matn": row.get("Matn"),
    }
