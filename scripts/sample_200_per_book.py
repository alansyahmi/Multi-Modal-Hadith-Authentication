"""Resample the 6,450 aligned hadiths to exactly 200 per canonical book."""
import sys, re, ast, json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"c:\Projects\FYP Project")

# ── Load the full aligned set ──
aligned = pd.read_csv("output/final_evaluation/stage1_aligned_balanced_4k.csv")
print(f"Aligned hadiths loaded: {len(aligned):,}")

# Map book names to canonical books
book_map = {
    "Sahih al-Bukhari": "Ṣaḥīḥ al-Bukhārī",
    "Sahih Muslim": "Ṣaḥīḥ Muslim",
    "Sunan Abi Dawud": "Sunan Abī Dāwūd",
    "Jami at-Tirmidhi": "Jāmiʿ at-Tirmiḏī",
    "Sunan an-Nasai": "Sunan al-Nasāʾī (al-Mujtabā)",
    "Sunan Ibn Majah": "Sunan Ibn Mājah",
}

# The aligned set has book_key column - map to standardized names
# First check what book info we have
if "book" in aligned.columns:
    print(f"Sample books: {aligned['book'].value_counts().head(10)}")
elif "book_key" in aligned.columns:
    print(f"Sample books: {aligned['book_key'].value_counts().head(10)}")

# The alignment returns query_df columns. Load query_df to get book info
import os
all_hf = []
for f in sorted(os.listdir("Dataset/hadith_datasets HuggingFace")):
    if not f.endswith(".csv"): continue
    df = pd.read_csv(f"Dataset/hadith_datasets HuggingFace/{f}")
    for _, row in df.iterrows():
        all_hf.append({
            "book_key": f.replace(".csv","").replace("%20"," "),
            "grade_raw": str(row.get("Grade","")),
        })

# Check: does the aligned set have a query_idx we can use?
if "query_idx" in aligned.columns:
    print(f"Has query_idx — can map back to original book info")
    sample_q = aligned["query_idx"].iloc[0]
    print(f"Sample query_idx: {sample_q}")
else:
    # Try to map by corpus_idx to Sanadset Book column
    print("No query_idx — checking other columns")
    print(f"Columns: {list(aligned.columns)}")
