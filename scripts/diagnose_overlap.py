"""Diagnose: do Sanadset and HuggingFace texts overlap at all for low-match books?

Compares raw Sanadset 'Matn' against HuggingFace 'Arabic_Text' (with and without matn extraction)
using TF-IDF to determine whether low match rates are due to:
  (A) bad matn extraction, or
  (B) genuine lack of overlap between datasets.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import numpy as np
sys.path.insert(0, r"c:\Projects\FYP Project")
from src.alignment.hybrid_aligner import extract_matn_portion, build_tfidf_index, retrieve_candidates, verify_match
from src.preprocessing.arabic_normalizer import normalise_hadith_text

# Load Sanadset corpus
sanadset = pd.read_csv(r"c:\Projects\FYP Project\output\stage1\sanadset_cleaned.csv", nrows=50000)
print(f"Sanadset loaded: {len(sanadset):,} rows (sample)")

for book in ["bukhari", "tirmidhi"]:
    print(f"\n{'='*60}")
    print(f"  BOOK: {book.upper()}")
    print(f"{'='*60}")

    # Load HuggingFace book
    hf_path = {
        "bukhari": r"c:\Projects\FYP Project\Dataset\hadith_datasets HuggingFace\Sahih%20al-Bukhari.csv",
        "tirmidhi": r"c:\Projects\FYP Project\Dataset\hadith_datasets HuggingFace\Jami20at-Tirmidhi.csv",
    }[book]
    hf = pd.read_csv(hf_path)
    print(f"HuggingFace: {len(hf):,} hadiths")

    # --- Test A: Raw Sanadset Matn vs HF Arabic_Text (no extraction on either side) ---
    print(f"\n  Test A: Raw Sanadset Matn (no extraction)")
    corpus_texts = sanadset["Matn"].fillna("").head(10000)
    # Normalise for matching
    corpus_norm = corpus_texts.apply(lambda t: normalise_hadith_text(t, for_matching=True))
    vectorizer, tfidf_matrix = build_tfidf_index(corpus_norm, max_features=5000)

    hf_texts = hf["Arabic_Text"].fillna("").head(200)
    high_score = 0
    for i in range(min(200, len(hf))):
        query = normalise_hadith_text(str(hf_texts.iloc[i]), for_matching=True)
        candidates = retrieve_candidates(query, vectorizer, tfidf_matrix, corpus_texts.index, top_k=1)
        if len(candidates) > 0 and candidates.iloc[0]["tfidf_score"] > high_score:
            high_score = candidates.iloc[0]["tfidf_score"]
    print(f"    Best TF-IDF score (raw, top-200): {high_score:.3f}")
    print(f"    Threshold for potential match: >0.7 = good candidate")

    # --- Test B: Extracted matn vs Sanadset Matn (current approach) ---
    print(f"\n  Test B: Extracted matn vs Sanadset Matn (current pipeline)")
    hf_matns = hf["Arabic_Text"].fillna("").head(200).apply(extract_matn_portion)
    hf_norm = hf_matns.apply(lambda t: normalise_hadith_text(t, for_matching=True))

    vectorizer2, tfidf_matrix2 = build_tfidf_index(corpus_norm, max_features=5000)

    matches = 0
    avg_score = 0
    for i in range(min(200, len(hf))):
        query = hf_norm.iloc[i]
        candidates = retrieve_candidates(query, vectorizer2, tfidf_matrix2, corpus_texts.index, top_k=1)
        if len(candidates) > 0:
            cs = candidates.iloc[0]["tfidf_score"]
            avg_score += cs
            # Check Levenshtein
            cand_text = corpus_texts.iloc[candidates.iloc[0]["corpus_idx"]]
            is_match, sim = verify_match(hf_matns.iloc[i], cand_text)
            if is_match:
                matches += 1

    avg_score = avg_score / max(1, len(hf))
    print(f"    Top-1 avg TF-IDF: {avg_score:.3f}")
    print(f"    Levenshtein matches (>=95%): {matches}/200 ({matches/2:.1f}%)")

    # --- Test C: Compare first 10 texts side by side ---
    print(f"\n  Test C: First 5 texts — side by side (abbreviated)")
    for i in range(5):
        hf_raw = str(hf.iloc[i]["Arabic_Text"])
        hf_matn = extract_matn_portion(hf_raw)
        sanad_first = str(sanadset.iloc[i]["Matn"]) if i < len(sanadset) else "(no data)"

        print(f"\n  [{i}]")
        print(f"    HF raw     ({len(hf_raw):4d}c): {normalise_hadith_text(hf_raw, for_matching=True)[:100]}")
        print(f"    HF matn    ({len(hf_matn):4d}c): {normalise_hadith_text(hf_matn, for_matching=True)[:100]}")
        print(f"    Sanad Matn ({len(sanad_first):4d}c): {normalise_hadith_text(sanad_first, for_matching=True)[:100]}")
