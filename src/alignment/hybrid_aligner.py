"""
Two-tiered Hybrid Alignment Framework.

Implements the alignment pipeline described in Section 3.2.3 of the methodology:

  Tier 1: Sparse TF-IDF retrieval (search space reduction: O(M×N) → O(K))
  Tier 2: Substring-level partial_ratio verification (≥95% threshold)

Aligns Sanadset 650K Matn (clean content text) against Kutub as-Sitta
full Arabic_Text (which includes the isnad chain). Uses partial_ratio to
find the best matching window of the Sanadset Matn within the longer HF text,
avoiding brittle SAW-based matn extraction.
"""

import ast
import json
import hashlib
import logging
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz.fuzz import partial_ratio
from tqdm import tqdm

from src.config import (
    TFIDF_NGRAM_RANGE,
    TFIDF_MAX_FEATURES,
    TOP_K_CANDIDATES,
    FUZZY_THRESHOLD,
    RANDOM_SEED,
)
from src.preprocessing.arabic_normalizer import normalise_hadith_text


def _similarity_ratio(x: str, y: str) -> float:
    """
    Compute the similarity ratio using partial_ratio substring matching.

    Uses rapidfuzz's partial_ratio which finds the best matching substring
    of the longer text against the shorter text. This is essential because
    Sanadset Matn is typically a substring (or near-substring) within the
    longer HuggingFace Arabic_Text that includes the isnad chain.

    Returns a score from 0-100 (%). """
    if not x or not y:
        return 0.0
    return float(partial_ratio(x, y))


def _generate_id(text: str) -> str:
    """
    Generate an immutable SHA-256 hash as a primary key.
    Uses the normalised consonantal skeleton. (Section 3.2.3)
    """
    normalised = normalise_hadith_text(text, for_matching=True)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]


def strip_sanadset_tags(text: str) -> str:
    """Strip <SANAD>, <NAR>, <MATN> tags from Sanadset Hadith text."""
    if pd.isna(text):
        return ""
    return re.sub(r"</?(?:SANAD|NAR|MATN)[^>]*>", "", text).strip()


def build_tfidf_index(
    corpus: pd.Series,
    ngram_range: tuple = TFIDF_NGRAM_RANGE,
    max_features: int = TFIDF_MAX_FEATURES,
) -> tuple:
    """
    Build a TF-IDF index over the corpus of normalised matn strings.

    Parameters
    ----------
    corpus       : Series of Arabic matn strings to index.
    ngram_range  : Character n-gram range (default 3-5 for Arabic).
    max_features : Vocabulary size cap.

    Returns
    -------
    (vectorizer, tfidf_matrix)
    """
    print(f"[aligner] Building TF-IDF index over {len(corpus):,} documents ...")
    print(f"  n-gram range: {ngram_range}, max_features: {max_features}")

    normalised = corpus.apply(lambda t: normalise_hadith_text(t, for_matching=True))

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        max_features=max_features,
        dtype=np.float32,
    )
    tfidf_matrix = vectorizer.fit_transform(normalised)

    print(f"  Vocabulary size: {len(vectorizer.get_feature_names_out()):,}")
    print(f"  Matrix shape: {tfidf_matrix.shape}")
    return vectorizer, tfidf_matrix


def retrieve_candidates(
    query: str,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    corpus_indices: pd.Index,
    top_k: int = TOP_K_CANDIDATES,
) -> pd.DataFrame:
    """
    Tier 1: Retrieve top-K candidate matches via TF-IDF cosine similarity.

    Parameters
    ----------
    query           : Query matn string.
    vectorizer      : Fitted TfidfVectorizer.
    tfidf_matrix    : Pre-computed TF-IDF matrix of the corpus.
    corpus_indices  : Index of the corpus DataFrame (for mapping back).
    top_k           : Number of candidates to retrieve.

    Returns
    -------
    DataFrame of top-K candidates with scores and corpus indices.
    """
    query_normalised = normalise_hadith_text(query, for_matching=True)
    query_vec = vectorizer.transform([query_normalised])
    scores = (tfidf_matrix @ query_vec.T).toarray().flatten()

    # Get top-K indices
    top_indices = np.argsort(scores)[-top_k:][::-1]
    top_scores = scores[top_indices]

    results = pd.DataFrame({
        "corpus_idx": corpus_indices.take(top_indices),
        "tfidf_score": top_scores,
    })
    return results[results["tfidf_score"] > 0]


def verify_match(
    query: str,
    candidate_text: str,
    threshold: float = FUZZY_THRESHOLD,
) -> tuple:
    """
    Tier 2: Verify a candidate match using partial_ratio similarity.

    Uses rapidfuzz's partial_ratio, which finds the best matching
    substring of the shorter text within the longer text. This is
    essential because the Sanadset Matn is typically a near-substring
    of the longer HuggingFace Arabic_Text that includes the isnad chain.

    Parameters
    ----------
    query           : Query matn string.
    candidate_text  : Candidate matn string from the corpus.
    threshold       : Minimum similarity ratio (default 95%).

    Returns
    -------
    (is_match, similarity_score)
    """
    q_norm = normalise_hadith_text(query, for_matching=True)
    c_norm = normalise_hadith_text(candidate_text, for_matching=True)
    sim = _similarity_ratio(q_norm, c_norm)
    return sim >= threshold, sim


def secondary_validation(
    query_row: pd.Series,
    candidate_row: pd.Series,
) -> bool:
    """
    Secondary deterministic validation: compare the first two narrators
    in each chain to break ties when multiple candidates have identical
    similarity scores (Section 3.2.3).

    If the first two ancestral nodes (closest to the collector) match,
    the candidate is accepted.
    """

    def get_first_two(sanad_str):
        try:
            narrators = ast.literal_eval(sanad_str) if isinstance(sanad_str, str) else []
            return narrators[:2] if len(narrators) >= 2 else narrators
        except (ValueError, SyntaxError, TypeError):
            return []

    q_nodes = get_first_two(query_row.get("Sanad", ""))
    c_nodes = get_first_two(candidate_row.get("Sanad", ""))

    if not q_nodes or not c_nodes:
        return False
    return q_nodes == c_nodes


def align_datasets(
    query_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    query_text_col: str = "Arabic_Text",
    corpus_text_col: str = "Matn",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full two-tiered alignment pipeline.

    Aligns by comparing Sanadset Matn (which is already clean content)
    against the full HuggingFace Arabic_Text using partial_ratio. This
    avoids brittle matn extraction — the Sanadset Matn is a near-substring
    within the longer HF text that includes the isnad chain.

    Parameters
    ----------
    query_df         : Smaller query dataset (Kutub as-Sitta samples).
    corpus_df        : Larger corpus dataset (Sanadset 650K).
    query_text_col   : Column in query_df with full hadith text.
    corpus_text_col  : Column in corpus_df with matn text.
    verbose          : Print progress.

    Returns
    -------
    DataFrame of matched records with columns:
      query_idx, corpus_idx, tfidf_score, similarity_score, is_match,
      id (SHA-256), quarantine (bool)
    """
    # Prepare corpus texts: use Matn column directly (already clean, no extraction)
    if corpus_text_col in corpus_df.columns:
        corpus_raw = corpus_df[corpus_text_col].fillna("")
    else:
        corpus_raw = corpus_df["Hadith"].fillna("").apply(strip_sanadset_tags)
    corpus_texts = corpus_raw  # Use as-is — no extract_matn_portion

    # Build TF-IDF index over corpus
    vectorizer, tfidf_matrix = build_tfidf_index(corpus_texts)

    results = []
    quarantined = 0
    matched = 0

    queries = tqdm(query_df.iterrows(), total=len(query_df), desc="Aligning") if verbose else query_df.iterrows()

    for idx, query_row in queries:
        # Use full Arabic_Text as query — no extraction needed,
        # partial_ratio finds the best window automatically
        query_text = str(query_row[query_text_col])

        # Tier 1: Retrieve candidates
        candidates = retrieve_candidates(
            query_text, vectorizer, tfidf_matrix, corpus_df.index
        )

        if len(candidates) == 0:
            results.append({
                "query_idx": idx,
                "corpus_idx": None,
                "tfidf_score": 0.0,
                "similarity_score": 0.0,
                "is_match": False,
                "id": _generate_id(query_text),
                "quarantine": True,
            })
            quarantined += 1
            continue

        # Tier 2: Verify each candidate using partial_ratio
        best_match = None
        best_sim = 0.0
        best_tfidf = candidates.iloc[0]["tfidf_score"]
        best_any_sim = 0.0
        for _, cand in candidates.iterrows():
            cand_text = corpus_raw.loc[cand["corpus_idx"]]
            is_match, sim = verify_match(query_text, cand_text)
            best_any_sim = max(best_any_sim, sim)

            if is_match and sim > best_sim:
                best_sim = sim
                best_match = cand["corpus_idx"]
                best_tfidf = cand["tfidf_score"]
            elif sim == best_sim and is_match and best_match is not None:
                # Tie-break via narrator comparison
                if not secondary_validation(query_row, corpus_df.loc[cand["corpus_idx"]]):
                    continue
                best_sim = sim
                best_match = cand["corpus_idx"]
                best_tfidf = cand["tfidf_score"]

        if best_match is not None:
            results.append({
                "query_idx": idx,
                "corpus_idx": best_match,
                "tfidf_score": best_tfidf,
                "similarity_score": best_sim,
                "is_match": True,
                "id": _generate_id(query_text),
                "quarantine": False,
            })
            matched += 1
        else:
            results.append({
                "query_idx": idx,
                "corpus_idx": candidates.iloc[0]["corpus_idx"],
                "tfidf_score": candidates.iloc[0]["tfidf_score"],
                "similarity_score": best_any_sim,
                "is_match": False,
                "id": _generate_id(query_text),
                "quarantine": True,
            })
            quarantined += 1

    result_df = pd.DataFrame(results)
    match_rate = matched / len(query_df) * 100

    if verbose:
        print(f"\n=== Alignment Results ===")
        print(f"  Total queries:    {len(query_df)}")
        print(f"  Matched:          {matched} ({match_rate:.1f}%)")
        print(f"  Quarantined:      {quarantined} ({quarantined/len(query_df)*100:.1f}%)")

    return result_df


def export_quarantine(
    query_df: pd.DataFrame,
    alignment: pd.DataFrame,
    output_path: str,
):
    """
    Export unmatched records to a JSON quarantine log for manual review
    (Section 3.2.3 — Quarantine Routing Vector).
    """
    quarantined = alignment[alignment["quarantine"]]
    if len(quarantined) == 0:
        print("[aligner] No quarantined records to export.")
        return

    records = []
    for _, row in quarantined.iterrows():
        query_row = query_df.loc[row["query_idx"]]
        records.append({
            "query_idx": int(row["query_idx"]),
            "book": str(query_row.get("Book", "")),
            "grade": str(query_row.get("Grade", "")),
            "similarity_score": float(row["similarity_score"]),
            "hadith_text_snippet": str(query_row.get("Arabic_Text", ""))[:200],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[aligner] Quarantine log saved: {output_path} ({len(records)} records)")


# ── Quick-start (when run directly) ───────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Two-Tiered Alignment — Standalone Test")
    print("=" * 60)

    # Quick test with sample data
    from src.data.sanadset_loader import load_samples
    from src.data.huggingface_loader import load_single_book, stratified_sample

    corpus = load_samples()
    query = stratified_sample(load_single_book("bukhari"), n_per_book=3)

    print(f"\nCorpus: {len(corpus)} samples")
    print(f"Query:  {len(query)} samples")

    result = align_datasets(query, corpus, verbose=True)
    print(f"\nFirst 3 alignment results:")
    print(result.head(3))
