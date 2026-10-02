"""Spot-check partial_ratio on quarantined entries (pick top TF-IDF candidates directly)."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import json, pandas as pd
sys.path.insert(0, r"c:\Projects\FYP Project")
from src.preprocessing.arabic_normalizer import normalise_hadith_text
from src.alignment.hybrid_aligner import build_tfidf_index, retrieve_candidates
from rapidfuzz.fuzz import partial_ratio

with open(r"c:\Projects\FYP Project\output\stage1\alignment_results.json") as f:
    results = json.load(f)
eval_df = pd.read_csv(r"c:\Projects\FYP Project\output\stage1\kutub_sitta_evaluation.csv")

# Load a decent sample of Sanadset for TF-IDF
sanadset = pd.read_csv(r"c:\Projects\FYP Project\output\stage1\sanadset_cleaned.csv", nrows=50000)
corpus = sanadset["Matn"].fillna("")

vec, tfidf_mtx = build_tfidf_index(corpus, max_features=5000)

# Pick 10 quarantined entries from various books
qs = [r for r in results if r.get("quarantine")]
import random
random.seed(42)
sample = random.sample(qs, min(10, len(qs)))

print("=== Spot-check: partial_ratio for quarantined entries ===")
for d in sample:
    row = eval_df.iloc[d["query_idx"]]
    bk = row["book_key"]
    qtext = str(row["Arabic_Text"])
    qnorm = normalise_hadith_text(qtext, for_matching=True)

    # Get TF-IDF candidate
    cands = retrieve_candidates(qtext, vec, tfidf_mtx, corpus.index, top_k=3)

    print(f"\n[{bk}] qidx={d['query_idx']} (TF-IDF sim reported: {d.get('tfidf_score',0):.3f})")
    print(f"  Query ({len(qnorm)}c): {qnorm[:120]}")

    for j, (_, cand) in enumerate(cands.iterrows()):
        cidx = int(cand["corpus_idx"])
        ctext = corpus.iloc[corpus.index.get_loc(cidx)]
        cnorm = normalise_hadith_text(str(ctext), for_matching=True)
        p = partial_ratio(qnorm, cnorm)
        print(f"  Cand{j+1} tfidf={cand['tfidf_score']:.3f}  partial={p:.1f}%")
        print(f"    Matn ({len(cnorm)}c): {cnorm[:120]}")
