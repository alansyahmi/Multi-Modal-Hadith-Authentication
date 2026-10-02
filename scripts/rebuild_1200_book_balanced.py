"""Rebuild exactly 1,200 evaluation set: 200 per book, aligned first."""
import sys, os, re, ast
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"c:\Projects\FYP Project")

# ── Build query_idx → book mapping from HF data ──
# The alignment script sampled 2,500 per grade from HF data.
# HF data is loaded in book-order. We need to reconstruct which
# indices went to which book.
print("Rebuilding query-to-book mapping...")
all_hf = []
for f in sorted(os.listdir("Dataset/hadith_datasets HuggingFace")):
    if not f.endswith(".csv"): continue
    df = pd.read_csv(f"Dataset/hadith_datasets HuggingFace/{f}")
    for _, row in df.iterrows():
        all_hf.append({"book_file": f, "grade_raw": str(row.get("Grade",""))})

# Apply same grade normalization as alignment script
gm = {"sahih":"Sahih","hasan":"Hasan","daif":"Daif","da'if":"Daif","maudu":"Matruk","mawdu":"Matruk"}
graded_indices = []
for i, h in enumerate(all_hf):
    grade = "Unknown"
    for k, v in gm.items():
        if k in h["grade_raw"].lower(): grade = v; break
    if grade != "Unknown":
        graded_indices.append((i, grade, h["book_file"]))

# Reconstruct the same sampling the alignment script did (2,500 per grade, random state 42)
rng = np.random.RandomState(42)
query_lookup = {}  # query_idx → {"grade": str, "book": str}

for grade in ["Sahih", "Hasan", "Daif", "Matruk"]:
    pool = [idx for idx, g, b in graded_indices if g == grade]
    n = min(2500, len(pool))
    chosen = rng.choice(pool, n, replace=False)
    for i, orig_idx in enumerate(chosen):
        query_idx = len(query_lookup)
        query_lookup[query_idx] = {"grade": grade, "book": all_hf[orig_idx]["book_file"]}

# Now load the alignment results (saved balanced file has query_idx)
aligned = pd.read_csv("output/final_evaluation/stage1_aligned_balanced_4k.csv")

# But wait — this file only has 3,022 (the balanced sample), not all 6,450.
# We need the FULL alignment. The full 6,450 wasn't saved.
# Let me check: the alignment script saved AFTER sampling.
# The ful aligned results were in the `alignment` variable.

# Alternative: use the original 1,022 book-stratified set from Stage 1
# which already has 200 per book
orig = pd.read_csv("output/stage1/kutub_sitta_evaluation.csv")
print(f"Original eval set: {len(orig)}")
print(f"Books: {orig['book_name'].value_counts().to_dict()}")

# Use the alignment_results.json to get the matched subset
import json
align = json.load(open("output/stage1/alignment_results.json", encoding="utf-8"))

# Filter to matched only
matched_indices = []
for i, a in enumerate(align):
    if a["is_match"]:
        matched_indices.append(i)

print(f"Matched queries: {len(matched_indices)}")

# Take exactly 200 per book
rng = np.random.RandomState(42)
selected = []
for book in orig["book_name"].unique():
    book_indices = [i for i in matched_indices if orig.iloc[i]["book_name"] == book]
    take = min(200, len(book_indices))
    chosen = rng.choice(book_indices, take, replace=False)
    selected.extend(chosen)
    print(f"  {book[:30]:30s}: {take}/{len(book_indices)} available")

selected.sort()
final = orig.iloc[selected].copy()
print(f"\nFinal: {len(final)} hadiths ({len(final)//6} per book × 6 books)")
final.to_csv("output/final_evaluation/book_balanced_1200.csv", index=False)
print("Saved: output/final_evaluation/book_balanced_1200.csv")
