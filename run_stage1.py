#!/usr/bin/env python3
"""
Stage 1: Data Ingestion and Preprocessing

Executes the full Stage 1 pipeline as described in Section 3.2:
  1. Load Sanadset 650K (~650K hadiths with pre-extracted narrators)
  2. Load Kutub as-Sitta ground-truth data (6 books, HuggingFace)
  3. Normalise Arabic text and parse sanad chains
  4. Stratified sampling: 200 Hadiths × 6 books = 1,200 evaluation set
  5. Two-tiered alignment (TF-IDF + Levenshtein) to cross-reference datasets
  6. Export cleaned data and quarantine log

Usage:
  python run_stage1.py              # Full pipeline
  python run_stage1.py --sample     # Run on sample data (for testing)

Output:
  output/stage1/
    sanadset_cleaned.csv       Cleaned Sanadset data
    kutub_sitta_evaluation.csv Stratified 1,200-hadith evaluation set
    alignment_results.json     Cross-reference results
    quarantine.json            Unmatched records for manual review
"""

import argparse
import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from src.config import (
    SANADSET_CSV,
    SANADSET_SAMPLES_CSV,
    STRATIFIED_SAMPLE_PER_BOOK,
    BALANCED_SAMPLE_PER_GRADE,
    STAGE1_DIR,
)
from src.data.sanadset_loader import load_sanadset_csv, explore_dataset
from src.data.huggingface_loader import load_all_books, stratified_sample, balanced_sample, normalize_grade
from src.preprocessing.sanad_parser import build_chain_record, validate_dataset_narrators
from src.alignment.hybrid_aligner import align_datasets, export_quarantine
from src.utils.helpers import timer, save_csv, save_json, dataframe_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Data Ingestion and Preprocessing")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run on sample data only (load hadith_samples.csv instead of full 650K)",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Number of rows to load from sanadset.csv (for testing)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        choices=[1200, 6000],
        default=1200,
        help="Target aligned hadiths: 1200 or 6000",
    )
    parser.add_argument(
        "--balance",
        type=str,
        choices=["book", "grade"],
        default="book",
        help="Balancing strategy: book-balanced (per-book sampling) or grade-balanced (per-grade sampling)",
    )
    parser.add_argument(
        "--prealigned",
        type=str,
        default=None,
        help="Path to pre-aligned JSON (e.g. output/final_evaluation/5k_stratified.json). "
             "Skips sampling and alignment steps; builds output files from pre-computed alignments.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("STAGE 1: DATA INGESTION AND PREPROCESSING")
    print("=" * 60)

    suffix = f"{args.sample_size}_{args.balance}_balanced"

    # ── Pre-aligned mode: skip sampling + alignment ─────────────
    if args.prealigned:
        print(f"\n[PRE-ALIGNED MODE] Reading pre-computed alignments from {args.prealigned} ...")
        import json as _json
        with open(args.prealigned, encoding="utf-8") as f:
            pre_data = _json.load(f)
        results = pre_data["results"] if isinstance(pre_data, dict) and "results" in pre_data else pre_data

        # Build reverse mapping: book filename → book_key (e.g. "Sunan_an-Nasai.csv" → "nasai")
        from src.config import HF_FILES
        file_to_key = {path.name: key for key, path in HF_FILES.items()}

        # Grade → tier mapping
        grade_tiers = {"Sahih": 4, "Hasan": 3, "Daif": 2, "Matruk": 1}
        grade_labels = {"Sahih": "Sahih", "Hasan": "Hasan", "Daif": "Daif", "Matruk": "Matruk"}

        align_records = []
        eval_rows = []
        for i, r in enumerate(results):
            grade = r["grade"]
            ci = r["ci"]
            book_file = r.get("book", "")
            book_key = file_to_key.get(book_file, "unknown")

            align_records.append({
                "query_idx": i,
                "corpus_idx": float(ci),
                "tfidf_score": r.get("sim", r.get("similarity", 0.0)),
                "similarity_score": r.get("sim", r.get("similarity", 0.0)) * 100,
                "is_match": True,
                "id": f"prealigned_{i}",
                "quarantine": False,
            })
            eval_rows.append({
                "Arabic_Text": "",
                "Grade": grade,
                "grade_normalised": grade,
                "book_key": book_key,
                "label": grade,
                "tier": grade_tiers.get(grade, 2),
            })

        alignment = pd.DataFrame(align_records)
        eval_df = pd.DataFrame(eval_rows)
        sanadset_df = pd.DataFrame()  # dummy, not used in export

        print(f"  Loaded {len(results):,} pre-aligned hadiths")
        from collections import Counter
        gc = Counter(r["grade"] for r in results)
        for g, n in gc.most_common():
            print(f"    {g:10s}: {n}")
        print(f"  (skipped Sanadset loading, chain parsing, sampling, and TF-IDF alignment)")

    # ── Normal pipeline (load Sanadset, sample, align) ──────────
    else:
        # ── Step 1: Load Sanadset 650K ──────────────────────────
        print("\n--- Step 1: Loading Sanadset 650K ---")
        with timer("Load Sanadset"):
            if args.sample:
                print("  [SAMPLE MODE] Loading hadith_samples.csv ...")
                sanadset_df = pd.read_csv(SANADSET_SAMPLES_CSV)
            else:
                sanadset_df = load_sanadset_csv(SANADSET_CSV, nrows=args.nrows)

        explore_dataset(sanadset_df)
        dataframe_stats(sanadset_df, "sanadset")

        # ── Step 2: Parse and validate sanad chains ─────────────
        print("\n--- Step 2: Parsing Sanad Chains ---")
        with timer("Parse chains"):
            validate_dataset_narrators(sanadset_df, sample_frac=0.01 if not args.sample else 1.0)
            chain_records = sanadset_df.head(10000).apply(build_chain_record, axis=1, result_type="expand")
            print(f"  Built {len(chain_records)} chain records (preview)")

        # ── Step 3: Load Kutub as-Sitta ground truth ────────────
        print("\n--- Step 3: Loading Kutub as-Sitta Ground Truth ---")
        with timer("Load Kutub as-Sitta"):
            kuttub_df = load_all_books()
        kuttub_df["grade_normalised"] = kuttub_df["Grade"].apply(normalize_grade)

        # ── Step 4: Sampling ────────────────────────────────────
        n_target = args.sample_size
        if args.balance == "book":
            n_per_book = STRATIFIED_SAMPLE_PER_BOOK[n_target]
            print(f"\n--- Step 4: Book-Balanced Sampling ({n_per_book} x 6 books = {n_per_book*6} queries, target {n_target} aligned) ---")
            with timer("Stratified sample"):
                eval_df = stratified_sample(kuttub_df, n_per_book=n_per_book)
        else:
            n_per_grade = BALANCED_SAMPLE_PER_GRADE[n_target]
            max_possible = n_per_grade * 4
            print(f"\n--- Step 4: Grade-Balanced Sampling ({n_per_grade} x 4 grades = {max_possible} queries max, target {n_target} aligned) ---")
            with timer("Balanced sample"):
                eval_df = balanced_sample(kuttub_df, n_per_grade=n_per_grade)

        grade_dist = eval_df["Grade"].apply(normalize_grade).value_counts()
        print(f"\n  Evaluation set grade distribution:")
        for grade, count in grade_dist.items():
            print(f"    {grade:15s}: {count} ({count/len(eval_df)*100:.1f}%)")

        # ── Step 5: Two-tiered alignment ─────────────────────────
        print("\n--- Step 5: Cross-Referencing Datasets (Alignment) ---")
        with timer("Alignment"):
            alignment = align_datasets(
                eval_df,
                sanadset_df,
                query_text_col="Arabic_Text",
                corpus_text_col="Matn",
            )

    # ── Step 6: Export results ───────────────────────────────
    print("\n--- Step 6: Exporting Results ---")
    STAGE1_DIR.mkdir(parents=True, exist_ok=True)

    with timer("Export"):
        # 6a. Save evaluation set (suffixed so both sample sizes can coexist)
        eval_out = STAGE1_DIR / f"kutub_sitta_evaluation_{suffix}.csv"
        save_csv(eval_df, eval_out)

        # 6b. Save alignment results
        align_out = STAGE1_DIR / f"alignment_results_{suffix}.json"
        save_json(alignment.to_dict(orient="records"), align_out)

        # 6c. Save quarantine log
        quarantine_out = STAGE1_DIR / f"quarantine_{suffix}.json"
        export_quarantine(eval_df, alignment, quarantine_out)

        # 6d. Save cleaned Sanadset sample
        if args.sample or args.nrows:
            sanad_out = STAGE1_DIR / "sanadset_cleaned.csv"
            save_csv(sanadset_df, sanad_out)

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STAGE 1 COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Evaluation set:    {len(eval_df)} Hadiths ({args.sample_size}-sample)")
    print(f"  Alignment rate:    {alignment['is_match'].sum()}/{len(alignment)} ({alignment['is_match'].mean()*100:.1f}%)")
    print(f"  Quarantined:       {alignment['quarantine'].sum()}")
    print(f"  Output directory:  {STAGE1_DIR}")


if __name__ == "__main__":
    main()
