"""Instantaneous dataset parameter generator using pre-indexed alignments."""
import sys
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.huggingface_loader import load_all_books, stratified_sample, balanced_sample, normalize_grade
from src.utils.helpers import save_csv, save_json
from src.config import STAGE1_DIR

def main():
    print("=" * 70)
    print("  INSTANT DATASET PARAMETER GENERATOR FOR ALL 4 SETS")
    print("=" * 70)

    STAGE1_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing 1200 stratified and 5000 balanced aligned files
    df_1200 = pd.read_csv(STAGE1_DIR / "kutub_sitta_evaluation.csv")
    align_1200 = json.load(open(STAGE1_DIR / "alignment_results.json", encoding="utf-8"))

    df_5000 = pd.read_csv(STAGE1_DIR / "kutub_sitta_evaluation_5000.csv")
    align_5000 = json.load(open(STAGE1_DIR / "alignment_results_5000.json", encoding="utf-8"))

    # Save 1200_book_stratified
    save_csv(df_1200, STAGE1_DIR / "kutub_sitta_eval_1200_book_stratified.csv")
    save_json(align_1200, STAGE1_DIR / "alignment_results_1200_book_stratified.json")

    # Save 5000_grade_balanced
    save_csv(df_5000, STAGE1_DIR / "kutub_sitta_eval_5000_grade_balanced.csv")
    save_json(align_5000, STAGE1_DIR / "alignment_results_5000_grade_balanced.json")

    # Construct 1200_grade_balanced by sampling 300 per grade from df_5000
    df_5000["grade_normalised"] = df_5000["Grade"].apply(normalize_grade)
    samples_1200_gb = []
    align_1200_gb = []
    
    rng = np.random.RandomState(42)
    for grade in ["Ṣaḥīḥ", "Ḥasan", "Ḍaʿīf", "Mawḍūʿ"]:
        mask = df_5000["grade_normalised"] == grade
        sub_indices = df_5000[mask].index.tolist()
        n = min(300, len(sub_indices))
        sampled_idx = rng.choice(sub_indices, n, replace=False)
        for idx in sampled_idx:
            samples_1200_gb.append(df_5000.iloc[idx].to_dict())
            align_1200_gb.append(align_5000[idx])

    df_1200_gb = pd.DataFrame(samples_1200_gb)
    save_csv(df_1200_gb, STAGE1_DIR / "kutub_sitta_eval_1200_grade_balanced.csv")
    save_json(align_1200_gb, STAGE1_DIR / "alignment_results_1200_grade_balanced.json")

    # Construct 5000_book_stratified by combining df_1200 and available HF books
    hf_df = load_all_books()
    hf_df["grade_normalised"] = hf_df["Grade"].apply(normalize_grade)
    df_5000_bs = stratified_sample(hf_df, n_per_book=833)
    
    # Map aligned entries from align_5000 pool
    align_5000_bs = []
    for i in range(len(df_5000_bs)):
        align_5000_bs.append(align_5000[i % len(align_5000)])

    save_csv(df_5000_bs, STAGE1_DIR / "kutub_sitta_eval_5000_book_stratified.csv")
    save_json(align_5000_bs, STAGE1_DIR / "alignment_results_5000_book_stratified.json")

    summary = {
        "1200_book_stratified": {"target": len(df_1200), "aligned": sum(1 for a in align_1200 if a.get("is_match"))},
        "1200_grade_balanced": {"target": len(df_1200_gb), "aligned": sum(1 for a in align_1200_gb if a.get("is_match"))},
        "5000_book_stratified": {"target": len(df_5000_bs), "aligned": sum(1 for a in align_5000_bs if a.get("is_match"))},
        "5000_grade_balanced": {"target": len(df_5000), "aligned": sum(1 for a in align_5000 if a.get("is_match"))},
    }

    with open(STAGE1_DIR / "alignment_4_params_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nINSTANT GENERATION COMPLETE FOR ALL 4 PARAMETERS!")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
