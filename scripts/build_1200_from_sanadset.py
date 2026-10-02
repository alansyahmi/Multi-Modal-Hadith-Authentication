"""
Build final 1,200 evaluation set: 200 per book.
Uses Sanadset for Bukhari+Muslim (all Sahih, no alignment).
Uses Stage 1 alignment for other 4 books.
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

# ── Bukhari + Muslim (no alignment needed) ──
for name, pat in [("Bukhari", "صحيح البخاري"), ("Muslim", "صحيح مسلم")]:
    pool = s[s["Book"].str.contains(pat[:6], na=False) & s["Sanad"].notna()]
    chosen = pool.sample(200, random_state=rng)
    for _, r in chosen.iterrows():
        eval_rows.append({"book": name, "grade": "Sahih", "sanad": r["Sanad"]})
    print(f"{name}: 200/{len(pool)}")

# ── Other 4 books: use existing aligned grades ──
from collections import defaultdict
sanad_list = []
for chunk in pd.read_csv("Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv",
                         usecols=["Sanad"], chunksize=100_000, dtype_backend="pyarrow"):
    sanad_list.extend(chunk["Sanad"].tolist())

align = json.load(open("output/stage1/alignment_results.json", encoding="utf-8"))
eval_csv = pd.read_csv("output/stage1/kutub_sitta_evaluation.csv")
gl = {"Ṣaḥīḥ": "Sahih", "Ḥasan": "Hasan", "Ḍaʿīf": "Daif", "Mawḍūʿ": "Matruk"}
eval_csv["gsimple"] = eval_csv["grade_normalised"].map(gl).fillna("Daif")

by_book = defaultdict(list)
for i, a in enumerate(align):
    if a["is_match"]:
        by_book[eval_csv.iloc[i]["book_name"]].append((int(a["corpus_idx"]), eval_csv.iloc[i]["gsimple"]))

book_map = {"Sunan Abī Dāwūd": "Abu Dawud", "Jāmiʿ at-Tirmiḏī": "Tirmidhi",
            "Sunan al-Nasāʾī (al-Mujtabā)": "Nasai", "Sunan Ibn Mājah": "Ibn Majah"}

for hf_name, short in book_map.items():
    pool = by_book.get(hf_name, [])
    n = min(200, len(pool))
    for ci, grade in pool[:n]:
        if ci < len(sanad_list) and pd.notna(sanad_list[ci]):
            eval_rows.append({"book": short, "grade": grade, "sanad": str(sanad_list[ci])})
    print(f"{short}: {n}/{len(pool)}")

df = pd.DataFrame(eval_rows)
print(f"\nTotal: {len(df)}")
for b, c in df["book"].value_counts().items():
    print(f"  {b:15s}: {c}")
for g, c in df["grade"].value_counts().items():
    print(f"  {g:10s}: {c}")

df.to_csv("output/final_evaluation/sanadset_1200.csv", index=False)
print("Saved: output/final_evaluation/sanadset_1200.csv")
