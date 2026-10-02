"""Precise partial_ratio test on a known-matching hadith."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
from rapidfuzz.fuzz import partial_ratio, ratio, token_set_ratio, token_sort_ratio
sys.path.insert(0, r"c:\Projects\FYP Project")
from src.preprocessing.arabic_normalizer import normalise_hadith_text

# Load actual texts from the datasets
sanad = pd.read_csv(
    r"c:\Projects\FYP Project\Dataset\Sanadset 650K Data on Hadith Narrators\Sanadset 650K Data on Hadith Narrators\hadith_samples.csv"
)
hf = pd.read_csv(r"c:\Projects\FYP Project\Dataset\hadith_datasets HuggingFace\Sahih%20al-Bukhari.csv")

# Use first row of each
sanad_matn = str(sanad.iloc[0]["Matn"])
hf_full = str(hf.iloc[0]["Arabic_Text"])

# Normalize
sanad_norm = normalise_hadith_text(sanad_matn, for_matching=True)
hf_norm = normalise_hadith_text(hf_full, for_matching=True)

print(f"Sanad Matn ({len(sanad_matn)} raw, {len(sanad_norm)} norm):")
print(f"  {sanad_norm[:200]}")
print(f"\nHF full ({len(hf_full)} raw, {len(hf_norm)} norm):")
print(f"  {hf_norm[:200]}")

# Compare: is sanad_norm a substring of hf_norm?
print(f"\n--- Substring check ---")
print(f"Sanad norm IN hf_norm: {sanad_norm in hf_norm}")

# Try different similarity methods
print(f"\n--- Similarity Methods (Sanad Matn vs HF full) ---")
print(f"partial_ratio:        {partial_ratio(sanad_norm, hf_norm):.1f}%")
print(f"ratio:                {ratio(sanad_norm, hf_norm):.1f}%")
print(f"token_set_ratio:      {token_set_ratio(sanad_norm, hf_norm):.1f}%")
print(f"token_sort_ratio:     {token_sort_ratio(sanad_norm, hf_norm):.1f}%")

# Sanad norm vs HF extracted matn (current approach)
from src.alignment.hybrid_aligner import extract_matn_portion
hf_matn = extract_matn_portion(hf_full)
hf_matn_norm = normalise_hadith_text(hf_matn, for_matching=True)

print(f"\n--- Current approach (Sanad Matn vs HF extracted matn) ---")
print(f"HF extracted matn ({len(hf_matn_norm)} norm): {hf_matn_norm[:150]}")
print(f"partial_ratio:        {partial_ratio(sanad_norm, hf_matn_norm):.1f}%")
print(f"ratio:                {ratio(sanad_norm, hf_matn_norm):.1f}%")

# Where does sanad_norm appear within hf_norm? Find offset
offset = hf_norm.find(sanad_norm[:50])
print(f"\n--- Position analysis ---")
print(f"First 50 chars of sanad_norm at HF offset: {offset}")
if offset >= 0:
    print(f"HF text at offset: ...{hf_norm[offset:offset+100]}...")
    print(f"Sanad text:         {sanad_norm[:100]}")
    # Check what comes before in HF
    print(f"HF pre-offset:  {hf_norm[max(0,offset-30):offset]}")

# Also try with diacritics stripped differently
print(f"\n--- Try treating SAW as a boundary (inverse: take text BEFORE) ---")
# What if we extract everything up to and including the SAW from HF?
# The matn is BETWEEN the chain's last narrator and the SAW or after it
for pattern, label in [(r"صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ", "SAW-full"),
                        (r"صلى الله عليه وسلم", "SAW-simple")]:
    import re
    match = re.search(pattern, hf_full)
    if match:
        before_saw = hf_full[:match.end()]
        before_saw_norm = normalise_hadith_text(before_saw, for_matching=True)
        print(f"\n  {label}: text before+including SAW ({len(before_saw_norm)} norm)")
        print(f"  ratio vs sanad: {ratio(before_saw_norm, sanad_norm):.1f}%")
        partial_vs_before = partial_ratio(sanad_norm, before_saw_norm)
        print(f"  partial_ratio vs sanad: {partial_vs_before:.1f}%")
