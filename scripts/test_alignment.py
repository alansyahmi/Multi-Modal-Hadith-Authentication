"""Quick alignment test with tag-stripped Sanadset text."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from src.data.sanadset_loader import load_samples
from src.data.huggingface_loader import load_single_book, stratified_sample
from src.alignment.hybrid_aligner import (
    strip_sanadset_tags, build_tfidf_index, retrieve_candidates, verify_match
)

corpus = load_samples()
corpus_texts = corpus["Hadith"].apply(strip_sanadset_tags)
print(f"Corpus: {len(corpus)} samples")
print(f"First corpus (first 120 chars): {corpus_texts.iloc[0][:120]}")

query = stratified_sample(load_single_book("bukhari"), n_per_book=10)
vec, mat = build_tfidf_index(corpus_texts)

for i, (idx, row) in enumerate(query.iterrows()):
    qtext = row["Arabic_Text"]
    candidates = retrieve_candidates(qtext, vec, mat, corpus.index, top_k=3)
    print(f"\nQuery {i+1}: {qtext[:100]}...")
    if len(candidates) > 0:
        for _, c in candidates.iterrows():
            ct = corpus_texts.loc[c["corpus_idx"]]
            is_match, sim = verify_match(qtext, ct)
            print(f"  TF-IDF={c['tfidf_score']:.3f} | Lev={sim:.2f} | Match={is_match}")
            print(f"  Corpus: {ct[:100]}...")
    else:
        print(f"  No candidates found")
    if i >= 2:
        break
