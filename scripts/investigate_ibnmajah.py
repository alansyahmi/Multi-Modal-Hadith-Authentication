"""Investigate Ibn Majah regression (82.5% → 79.5%)."""
import json, sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
sys.path.insert(0, r"c:\Projects\FYP Project")
from src.preprocessing.arabic_normalizer import normalise_hadith_text
from rapidfuzz.fuzz import partial_ratio, ratio

# Load current results
with open(r"c:\Projects\FYP Project\output\stage1\alignment_results.json") as f:
    data = json.load(f)
eval_df = pd.read_csv(r"c:\Projects\FYP Project\output\stage1\kutub_sitta_evaluation.csv")

sanad = pd.read_csv(r"c:\Projects\FYP Project\output\stage1\sanadset_cleaned.csv", nrows=1000)

# Ibn Majah specific
ibn = [d for d in data if eval_df.iloc[d["query_idx"]]["book_key"] == "ibnmajah"]
matched = sum(1 for d in ibn if d.get("is_match"))
quarantined = sum(1 for d in ibn if d.get("quarantine"))
print(f"Ibn Majah: {matched} matched / {quarantined} quarantined")

# Show Ibn Majah quarantined with near-miss scores
q_ibn = [d for d in ibn if d.get("quarantine")]
print(f"\n=== Ibn Majah quarantined with sim > 80% ===")
near = [d for d in q_ibn if d.get("similarity_score", 0) > 80]
near.sort(key=lambda d: d.get("similarity_score", 0), reverse=True)
for d in near[:8]:
    row = eval_df.iloc[d["query_idx"]]
    sim = d.get("similarity_score", 0)
    tfidf = d.get("tfidf_score", 0)
    qtext = normalise_hadith_text(str(row["Arabic_Text"]), for_matching=True)
    print(f"  qidx={d['query_idx']}  sim={sim:.2f}%  tfidf={tfidf:.3f}")
    print(f"    Query: {qtext[:150]}")

# Compare: what if we use token_set_ratio instead of partial_ratio?
from rapidfuzz.fuzz import token_set_ratio
print(f"\n=== Sample: token_set_ratio vs partial_ratio ===")
for i in range(min(5, len(q_ibn))):
    d = q_ibn[i]
    row = eval_df.iloc[d["query_idx"]]
    qtext = str(row["Arabic_Text"])
    # Find the best TF-IDF candidate
    cand_idx = d.get("corpus_idx")
    if cand_idx is not None and cand_idx < len(sanad):
        ctext = str(sanad.iloc[cand_idx]["Matn"])
        qnorm = normalise_hadith_text(qtext, for_matching=True)
        cnorm = normalise_hadith_text(ctext, for_matching=True)
        p = partial_ratio(qnorm, cnorm)
        t = token_set_ratio(qnorm, cnorm)
        r = ratio(qnorm, cnorm)
        print(f"  [{i}] partial={p:.1f}%  token_set={t:.1f}%  full_ratio={r:.1f}%")
        print(f"    Q: {qnorm[:120]}")
        print(f"    C: {cnorm[:120]}")
