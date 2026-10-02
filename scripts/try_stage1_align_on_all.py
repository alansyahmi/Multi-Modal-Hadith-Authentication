"""
Use the EXISTING Stage 1 alignment pipeline on ALL graded HF hadiths.
This is the approach you suggested — reuse what already works.
"""
import sys, os
sys.path.insert(0, r"c:\Projects\FYP Project")
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from collections import Counter
import numpy as np
from src.alignment.hybrid_aligner import align_datasets
from src.data.huggingface_loader import normalize_grade, load_all_books

# ── 1. Load ALL HF hadiths using existing loader ──
print("Loading all HF hadiths...")
all_books = load_all_books()
print(f"  Total: {len(all_books):,}")

# Normalize grades using existing function
all_books["grade_normalised"] = all_books["Grade"].apply(normalize_grade)

# Filter to graded only
graded = all_books[all_books["grade_normalised"].notna()].copy()
print(f"  Graded: {len(graded):,}")
for g, c in Counter(graded["grade_normalised"]).most_common():
    print(f"    {g}: {c:,}")

# Map to 4 classes
grade_4 = {"Ṣaḥīḥ": "Sahih", "Ḥasan": "Hasan", "Ḍaʿīf": "Daif", "Mawḍūʿ": "Matruk"}
graded["grade_4"] = graded["grade_normalised"].map(grade_4).fillna("Daif")

# ── 2. Load Sanadset ──
print("\nLoading Sanadset...")
sanadset = pd.read_csv(
    "Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv",
    dtype_backend="pyarrow",
)
print(f"  Sanadset: {len(sanadset):,} rows")

# ── 3. Run alignment on ALL graded (using existing aligner) ──
print(f"\nAligning {len(graded):,} queries against {len(sanadset):,} corpus...")
# Use a random sample of N per grade for manageable size
rng = np.random.RandomState(42)
SAMPLE_PER_GRADE = 2500  # Total would be 4 × 2500 = 10K

sampled = []
for g in ["Sahih", "Hasan", "Daif", "Matruk"]:
    pool = graded[graded["grade_4"] == g]
    n = min(SAMPLE_PER_GRADE, len(pool))
    chosen = pool.sample(n, random_state=rng)
    sampled.append(chosen)
    print(f"  {g}: sampling {n}/{len(pool)}")
query_df = pd.concat(sampled).reset_index(drop=True)
print(f"  Total queries: {len(query_df):,}")

# Align
alignment = align_datasets(
    query_df, sanadset,
    query_text_col="Arabic_Text",
    corpus_text_col="Matn",
    verbose=True,
)

# ── 4. Filter matched only ──
matched = alignment[alignment["is_match"] == True].copy()
print(f"\n  Matched: {len(matched)}/{len(alignment)}")

# ── 5. Grade-balanced sample (1K per grade) ──
matched["grade_4"] = query_df.loc[matched["query_idx"], "grade_4"].values
eval_set = []
for g in ["Sahih", "Hasan", "Daif", "Matruk"]:
    pool = matched[matched["grade_4"] == g]
    n = min(1000, len(pool))
    chosen = pool.sample(n, random_state=rng)
    eval_set.append(chosen)
    print(f"  {g}: taking {n}/{len(pool)}")
eval_df = pd.concat(eval_set).reset_index(drop=True)
print(f"\n  Final eval set: {len(eval_df):,}")

# ── 6. Save ──
eval_df.to_csv("output/final_evaluation/stage1_aligned_balanced_4k.csv", index=False)
print(f"\nSaved: output/final_evaluation/stage1_aligned_balanced_4k.csv")
