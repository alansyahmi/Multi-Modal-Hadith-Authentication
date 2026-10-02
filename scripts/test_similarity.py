"""Test Levenshtein similarity between matched matn texts."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"c:\Projects\FYP Project")

import pandas as pd
from src.alignment.hybrid_aligner import extract_matn_portion, verify_match
from src.preprocessing.arabic_normalizer import normalise_hadith_text
from rapidfuzz.distance import Levenshtein

sanad = pd.read_csv(
    r"c:\Projects\FYP Project\Dataset\Sanadset 650K Data on Hadith Narrators\Sanadset 650K Data on Hadith Narrators\hadith_samples.csv"
)
hf = pd.read_csv(
    r"c:\Projects\FYP Project\Dataset\hadith_datasets HuggingFace\Sahih%20al-Bukhari.csv"
)

# Compare matn extracts for the first few
print("=== Matn Similarity Comparison ===")
for i in range(min(5, len(hf))):
    hf_matn = extract_matn_portion(hf.iloc[i]["Arabic_Text"])
    sanad_matn = extract_matn_portion(sanad.iloc[0]["Matn"])  # Compare against 1st hadith

    # Normalize
    hf_norm = normalise_hadith_text(hf_matn, for_matching=True)
    sanad_norm = normalise_hadith_text(sanad_matn, for_matching=True)

    dist = Levenshtein.distance(hf_norm, sanad_norm)
    total = len(hf_norm) + len(sanad_norm)
    sim = ((total - dist) / total * 100) if total > 0 else 0

    match, _ = verify_match(hf_matn, sanad_matn)

    print(f"Hadaf {i+1}: len={len(hf_norm):>4}, len2={len(sanad_norm):>4}, dist={dist:>4}, sim={sim:.2f}%, match={match}")
    print(f"  HF:     {hf_norm[:80]}")
    print(f"  Sanad:  {sanad_norm[:80]}")
    print()
