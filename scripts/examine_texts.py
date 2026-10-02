"""Examine text structure for matn extraction."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

sanad = pd.read_csv(
    r"c:\Projects\FYP Project\Dataset\Sanadset 650K Data on Hadith Narrators\Sanadset 650K Data on Hadith Narrators\hadith_samples.csv"
)
hf = pd.read_csv(
    r"c:\Projects\FYP Project\Dataset\hadith_datasets HuggingFace\Jami20at-Tirmidhi.csv"
)

print("=== Sanadset Matn examples ===")
for i in range(3):
    print(f"  [{i}] {sanad.iloc[i]['Matn'][:200]}")
    print()

print("=== HuggingFace Arabic_Text examples ===")
for i in range(3):
    txt = hf.iloc[i]["Arabic_Text"]
    print(f"  [{i}] full: {txt[:200]}...{txt[-150:]}")
    print()

# Also check the first few from Sanadset Hadith column (tag-stripped)
print("=== Sanadset Hadith (tag-stripped) ===")
import re
for i in range(3):
    raw = sanad.iloc[i]["Hadith"]
    stripped = re.sub(r"</?(?:SANAD|NAR|MATN)[^>]*>", "", raw).strip()
    print(f"  [{i}] {stripped[:200]}...{stripped[-150:]}")
    print()
