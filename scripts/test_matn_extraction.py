"""Test matn extraction."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import re
sys.path.insert(0, r"c:\Projects\FYP Project")
from src.alignment.hybrid_aligner import extract_matn_portion
from src.data.sanadset_loader import load_samples
from src.data.huggingface_loader import load_single_book

# Test on Sanadset Matn
corpus = load_samples()
for i in range(3):
    raw = corpus.iloc[i]["Matn"]
    matn = extract_matn_portion(raw)
    print(f"Sanadset Matn [{i}]:")
    print(f"  Raw:   {raw[:120]}...")
    print(f"  Matn:  {matn[:200]}")
    print()

# Test on HuggingFace text
hf = load_single_book("bukhari")
for i in range(3):
    raw = hf.iloc[i]["Arabic_Text"]
    matn = extract_matn_portion(raw)
    print(f"HF Bukhari [{i}]:")
    print(f"  Raw:   {raw[:120]}...")
    print(f"  Matn:  {matn[:200]}")
    print()
