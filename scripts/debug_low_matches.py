"""Debug low match-rate books."""
import json, sys, re
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
sys.path.insert(0, r"c:\Projects\FYP Project")
from src.alignment.hybrid_aligner import extract_matn_portion
from src.preprocessing.arabic_normalizer import normalise_hadith_text
from rapidfuzz.distance import Levenshtein

with open(r"c:\Projects\FYP Project\output\stage1\alignment_results.json") as f:
    data = json.load(f)

eval_df = pd.read_csv(r"c:\Projects\FYP Project\output\stage1\kutub_sitta_evaluation.csv")

# Match query_idx to book_key
for d in data:
    d["book_key"] = eval_df.iloc[d["query_idx"]]["book_key"]

# Check a few Bukhari quarantine entries
print("=== Bukhari Quarantine Deep Dive ===")
bukhari_q = [d for d in data if d["book_key"] == "bukhari" and d.get("quarantine")]
print(f"Total Bukhari quarantined: {len(bukhari_q)}")

# Show top candidates with their TF-IDF scores
for d in bukhari_q[:5]:
    print(f"\nQuery idx={d['query_idx']}: tfidf_best={d.get('tfidf_score', 0):.3f}")
    row = eval_df.iloc[d["query_idx"]]
    qmatn = extract_matn_portion(str(row["Arabic_Text"]))
    qnorm = normalise_hadith_text(qmatn, for_matching=True)
    print(f"  Query matn: {qnorm[:150]}")
    print(f"  Matn len: {len(qnorm)}")

