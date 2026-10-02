"""Analyze alignment per book."""
import json, sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

with open(r"c:\Projects\FYP Project\output\stage1\alignment_results.json") as f:
    data = json.load(f)

eval_df = pd.read_csv(r"c:\Projects\FYP Project\output\stage1\kutub_sitta_evaluation.csv")

# Match query_idx to book
for i, d in enumerate(data):
    d["book_key"] = eval_df.iloc[d["query_idx"]]["book_key"]

# Per-book stats
from collections import Counter
book_matched = Counter()
book_total = Counter()
for d in data:
    bk = d["book_key"]
    book_total[bk] += 1
    if d.get("is_match"):
        book_matched[bk] += 1

print("=== Per-Book Alignment Rate ===")
for bk in sorted(book_total):
    m = book_matched.get(bk, 0)
    t = book_total[bk]
    print(f"  {bk:15s}: {m}/{t} ({m/t*100:.1f}%)")

print(f"\nTotal: {sum(book_matched.values())}/{sum(book_total.values())} ({(sum(book_matched.values())/sum(book_total.values()))*100:.1f}%)")
