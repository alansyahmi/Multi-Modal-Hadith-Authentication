"""Analyse quarantined entries — clip-show of text + scores."""
import json, sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
sys.path.insert(0, r"c:\Projects\FYP Project")
from src.preprocessing.arabic_normalizer import normalise_hadith_text

with open(r"c:\Projects\FYP Project\output\stage1\alignment_results.json") as f:
    data = json.load(f)
eval_df = pd.read_csv(r"c:\Projects\FYP Project\output\stage1\kutub_sitta_evaluation.csv")

# Sanadset for spot-check
sanad = pd.read_csv(r"c:\Projects\FYP Project\output\stage1\sanadset_cleaned.csv", nrows=100)

print("=== Quarantine Analysis ===")
print(f"Total:             {len(data):,}")
print(f"Matched:           {sum(1 for d in data if d.get('is_match')):,}")
print(f"Quarantined:       {sum(1 for d in data if d.get('quarantine')):,}")
print()

# Per-book quarantine rate
from collections import Counter
book_q = Counter()
book_t = Counter()
for d in data:
    bk = eval_df.iloc[d["query_idx"]]["book_key"]
    book_t[bk] += 1
    if d.get("quarantine"):
        book_q[bk] += 1

print("=== Per-book Quarantine Rate ===")
for bk in sorted(book_t):
    q = book_q[bk]
    t = book_t[bk]
    print(f"  {bk:15s}: {q}/{t} quarantined ({q/t*100:.1f}%)")
print()

# Histogram of best_any_sim for quarantined
qs = [d for d in data if d.get("quarantine")]
sims = [d.get("similarity_score", 0) for d in qs]
buckets = {"0-50": 0, "50-70": 0, "70-80": 0, "80-90": 0, "90-94.9": 0, "95+": 0}
for s in sims:
    if s >= 95: buckets["95+"] += 1
    elif s >= 90: buckets["90-94.9"] += 1
    elif s >= 80: buckets["80-90"] += 1
    elif s >= 70: buckets["70-80"] += 1
    elif s >= 50: buckets["50-70"] += 1
    else: buckets["0-50"] += 1

print("=== Similarity Score Distribution (quarantined) ===")
for k, v in buckets.items():
    bar = "#" * (v // 5)
    print(f"  {k:10s}: {v:>4}  {bar}")

# Spot-check 5 quarantined with highest similarity
high_q = sorted(qs, key=lambda d: d.get("similarity_score", 0), reverse=True)[:5]
print(f"\n=== Top 5 Near-Misses (highest similarity, but below threshold) ===")
for d in high_q:
    row = eval_df.iloc[d["query_idx"]]
    bk = row["book_key"]
    sim = d.get("similarity_score", 0)
    tfidf = d.get("tfidf_score", 0)
    query_norm = normalise_hadith_text(str(row["Arabic_Text"]), for_matching=True)[:120]

    # try to get the candidate matn (if corpus_idx available)
    cand_idx = d.get("corpus_idx")
    if cand_idx is not None and cand_idx < len(sanad):
        cand_matn = normalise_hadith_text(str(sanad.iloc[cand_idx]["Matn"]), for_matching=True)[:120]
    else:
        cand_matn = "(not in sample)"

    print(f"\n  [{bk}] qidx={d['query_idx']}  sim={sim:.2f}%  tfidf={tfidf:.3f}")
    print(f"    Query matn:  {query_norm}")
    print(f"    Cand matn:   {cand_matn}")
