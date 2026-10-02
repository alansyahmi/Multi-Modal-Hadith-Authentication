"""
Sample exactly 1,200 from Kutub as-Sitta in Sanadset ONLY.
Bukhari + Muslim: grade known (all Sahih).
Other 4 books: carry over aligned grades from Stage 1 (already done).
"""
import sys, json, re, ast
import pandas as pd
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

print("Loading Sanadset...")
s = pd.read_csv(
    "Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv",
    usecols=["Book", "Sanad"], dtype_backend="pyarrow",
)

rng = np.random.RandomState(42)
eval_rows = []
found_books = set(s["Book"].dropna().unique())

# ── Identify canonical book patterns ──
patterns = {
    "Bukhari": "صحيح البخاري",
    "Muslim": "صحيح مسلم",
    "Abu Dawud": "سنن أبي داود",
    "Tirmidhi": "جامع الترمذي",
    "Nasai": "السنن الكبرى للنسائي",
    "Ibn Majah": "سنن ابن ماجه",
}

# Count available
for name, pat in patterns.items():
    matching = [b for b in found_books if pat[:6] in b]
    count = sum(1 for _, row in s[s["Book"].str.contains(pat[:6], na=False) & s["Sanad"].notna()].iterrows())
    print(f"  {name:15s}: {count:,} rows with sanad")

# ── Sample 200 from each ──
for name, pat in patterns.items():
    pool = s[s["Book"].str.contains(pat[:6], na=False) & s["Sanad"].notna()]
    n = min(200, len(pool))
    chosen = pool.sample(n, random_state=rng)
    for _, row in chosen.iterrows():
        eval_rows.append({"book": name, "sanad": row["Sanad"]})
    print(f"  Sampled {n}/{len(pool)} from {name}")

df = pd.DataFrame(eval_rows)
print(f"\nTotal: {len(df)} (200 × 6 books)")
print(df["book"].value_counts().to_dict())

# ── Assign grades ──
# Bukhari + Muslim: all Sahih
df.loc[df["book"].isin(["Bukhari", "Muslim"]), "grade"] = "Sahih"

# Other 4 books: use aligned grades from Stage 1
align = json.load(open("output/stage1/alignment_results.json", encoding="utf-8"))
eval_csv = pd.read_csv("output/stage1/kutub_sitta_evaluation.csv")
gl = {"Ṣaḥīḥ": "Sahih", "Ḥasan": "Hasan", "Ḍaʿīf": "Daif", "Mawḍūʿ": "Matruk"}
eval_csv["gsimple"] = eval_csv["grade_normalised"].map(gl).fillna("Daif")

# Book name mapping
book_map = {
    "Abu Dawud": "Sunan Abī Dāwūd",
    "Tirmidhi": "Jāmiʿ at-Tirmiḏī",
    "Nasai": "Sunan al-Nasāʾī (al-Mujtabā)",
    "Ibn Majah": "Sunan Ibn Mājah",
}

# For each sampled hadith from mixed books, find its grade
# by matching through the alignment
sanad_list = []
for chunk in pd.read_csv("Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv",
                         usecols=["Sanad"], chunksize=100_000, dtype_backend="pyarrow"):
    sanad_list.extend(chunk["Sanad"].tolist())

# Build lookup: corpus_idx → grade from alignment
corpus_grade = {}
for i, a in enumerate(align):
    if a["is_match"]:
        ci = int(a["corpus_idx"])
        corpus_grade[ci] = eval_csv.iloc[i]["gsimple"]

# For each mixed-book row, try to find its grade
for idx, row in df.iterrows():
    if row["book"] in book_map:
        # Search for this chain in the Sanadset by index
        # We sampled from the Sanadset, but we need to find the original index
        # Direct approach: scan Sanadset for matching chain
        pass  # Too expensive — skip

# Alternative: just assign grade where alignment exists
# We know which corpus_idx maps to which grade
# But we sampled randomly from Sanadset — we don't know corpus_idx

# Simplest approach: use the aligned set directly for mixed books
# The aligned set already has grades from Stage 1

# Load the aligned set
aligned_mixed = []
for i, a in enumerate(align):
    if a["is_match"]:
        book = eval_csv.iloc[i]["book_name"]
        ci = int(a["corpus_idx"])
        if ci < len(sanad_list) and pd.notna(sanad_list[ci]):
            aligned_mixed.append({
                "book": book, "grade": eval_csv.iloc[i]["gsimple"],
                "sanad": str(sanad_list[ci])
            })

aligned_df = pd.DataFrame(aligned_mixed)
print(f"\nAligned mixed books: {len(aligned_df)}")

# Replace mixed-book rows with aligned versions
mask = df["book"].isin(book_map)
df = df[~mask]  # Remove ungraded mixed-book rows

# Add aligned versions
book_rev = {v: k for k, v in book_map.items()}
for _, row in aligned_df.iterrows():
    short = book_rev.get(row["book"], row["book"])
    df = pd.concat([df, pd.DataFrame([{"book": short, "sanad": row["sanad"],
                                        "grade": row["grade"]}])], ignore_index=True)

# Sample 200 per mixed book from aligned
final = []
for b in ["Bukhari", "Muslim", "Abu Dawud", "Tirmidhi", "Nasai", "Ibn Majah"]:
    pool = df[df["book"] == b]
    n = min(200, len(pool))
    chosen = pool.sample(n, random_state=rng)
    final.append(chosen)

final_df = pd.concat(final).reset_index(drop=True)
print(f"\nFinal: {len(final_df)} hadiths")
for b in ["Bukhari", "Muslim", "Abu Dawud", "Tirmidhi", "Nasai", "Ibn Majah"]:
    sub = final_df[final_df["book"] == b]
    print(f"  {b:15s}: {len(sub)}")
for g, c in final_df["grade"].value_counts().items():
    print(f"  {g:10s}: {c}")

final_df.to_csv("output/final_evaluation/sanadset_only_1200.csv", index=False)
print("\nSaved: output/final_evaluation/sanadset_only_1200.csv")
