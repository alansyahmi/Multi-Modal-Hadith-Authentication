"""Test partial_ratio approach: compare Sanadset Matn against HF full text."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import numpy as np
from rapidfuzz.distance import Levenshtein
from rapidfuzz.fuzz import partial_ratio, ratio
sys.path.insert(0, r"c:\Projects\FYP Project")
from src.alignment.hybrid_aligner import extract_matn_portion, build_tfidf_index, retrieve_candidates, verify_match
from src.preprocessing.arabic_normalizer import normalise_hadith_text

# Load Sanadset sample
sanadset = pd.read_csv(r"c:\Projects\FYP Project\output\stage1\sanadset_cleaned.csv", nrows=10000)
sanad_matns = sanadset["Matn"].fillna("")

# Build TF-IDF index on Sanadset Matn (NO extraction)
vectorizer, tfidf_matrix = build_tfidf_index(sanad_matns, max_features=5000)

for book in ["bukhari", "tirmidhi", "muslim", "ibnmajah"]:
    hf_path = {
        "bukhari": r"c:\Projects\FYP Project\Dataset\hadith_datasets HuggingFace\Sahih%20al-Bukhari.csv",
        "tirmidhi": r"c:\Projects\FYP Project\Dataset\hadith_datasets HuggingFace\Jami20at-Tirmidhi.csv",
        "muslim": r"c:\Projects\FYP Project\Dataset\hadith_datasets HuggingFace\Sahih%20Muslim.csv",
        "ibnmajah": r"c:\Projects\FYP Project\Dataset\hadith_datasets HuggingFace\Sunan20Majah.csv",
    }[book]
    hf = pd.read_csv(hf_path)

    print(f"\n{'='*60}")
    print(f"  BOOK: {book.upper()} — testing partial_ratio")
    print(f"{'='*60}")

    results_full = []
    results_partial = []

    for i in range(min(100, len(hf))):
        hf_text = str(hf.iloc[i]["Arabic_Text"])
        hf_norm = normalise_hadith_text(hf_text, for_matching=True)

        # Current approach: extract_matn_portion + full ratio
        hf_matn = extract_matn_portion(hf_text)
        hf_matn_norm = normalise_hadith_text(hf_matn, for_matching=True)

        # TF-IDF retrieve
        candidates = retrieve_candidates(hf_text, vectorizer, tfidf_matrix, sanad_matns.index, top_k=1)
        if len(candidates) == 0:
            continue

        best_idx = int(candidates.iloc[0]["corpus_idx"])
        # corpus_idx is the actual DataFrame index label, use positional via get
        sanad_matn = sanad_matns.iloc[sanad_matns.index.get_loc(best_idx)]
        sanad_norm = normalise_hadith_text(sanad_matn, for_matching=True)

        # Method A: full_ratio between extracted HF matn and Sanadset Matn (CURRENT)
        full_lev = Levenshtein.distance(hf_matn_norm, sanad_norm)
        total_len = len(hf_matn_norm) + len(sanad_norm)
        full_sim = ((total_len - full_lev) / total_len * 100) if total_len > 0 else 0

        # Method B: partial_ratio between FULL HF text and Sanadset Matn (PROPOSED)
        partial_sim = partial_ratio(sanad_norm, hf_norm)

        # Method C: full_ratio between full HF text and Sanadset Matn
        full2_lev = Levenshtein.distance(hf_norm, sanad_norm)
        total2_len = len(hf_norm) + len(sanad_norm)
        full2_sim = ((total2_len - full2_lev) / total2_len * 100) if total2_len > 0 else 0

        results_full.append(full_sim)
        results_partial.append(partial_sim)

    # Stats
    arr_full = np.array(results_full)
    arr_partial = np.array(results_partial)

    print(f"  Current (extract+full_ratio): mean={arr_full.mean():.1f}%, >=95%: {(arr_full>=95).sum()}/{len(arr_full)} ({(arr_full>=95).sum()/len(arr_full)*100:.1f}%)")
    print(f"  Proposed (partial_ratio):     mean={arr_partial.mean():.1f}%, >=95%: {(arr_partial>=95).sum()}/{len(arr_partial)} ({(arr_partial>=95).sum()/len(arr_partial)*100:.1f}%)")

    # Show first 3 examples
    for i in range(min(3, len(results_full))):
        hf_text = str(hf.iloc[i]["Arabic_Text"])
        hf_matn = extract_matn_portion(hf_text)
        cand = retrieve_candidates(hf_text, vectorizer, tfidf_matrix, sanad_matns.index, top_k=1)
        if len(cand) == 0:
            continue
        cand_idx = int(cand.iloc[0]["corpus_idx"])
        sanad_m = sanad_matns.iloc[sanad_matns.index.get_loc(cand_idx)]
        sanad_n = normalise_hadith_text(sanad_m, for_matching=True)
        hf_n = normalise_hadith_text(hf_matn, for_matching=True)
        hf_full_n = normalise_hadith_text(hf_text, for_matching=True)

        print(f"\n  [{i}] TF-IDF={cand.iloc[0]['tfidf_score']:.3f}")
        print(f"    Sanad Matn ({len(sanad_n)}c): {sanad_n[:100]}")
        print(f"    HF matn    ({len(hf_n)}c): {hf_n[:100]}")
        print(f"    HF full    ({len(hf_full_n)}c): {hf_full_n[:100]}")
        print(f"    Current={results_full[i]:.1f}%  Partial={results_partial[i]:.1f}%")
