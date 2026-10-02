"""Build and align all 4 dataset parameter configurations against Sanadset 650K."""
import sys
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.sanadset_loader import load_sanadset_csv
from src.data.huggingface_loader import load_all_books, stratified_sample, balanced_sample, normalize_grade
from src.alignment.hybrid_aligner import align_datasets, export_quarantine
from src.utils.helpers import timer, save_csv, save_json
from src.config import SANADSET_CSV, STAGE1_DIR

def build_and_align_all_4():
    print("=" * 70)
    print("  BUILDING & ALIGNING ALL 4 DATASET PARAMETER CONFIGURATIONS")
    print("=" * 70)
    
    # 1. Load Sanadset 650K
    print("\n[1/4] Loading Sanadset 650K...")
    sanadset_df = load_sanadset_csv(SANADSET_CSV)
    print(f"  Loaded {len(sanadset_df):,} Sanadset rows")
    
    # 2. Load HuggingFace ground truth
    print("\n[2/4] Loading Kutub as-Sitta Ground Truth...")
    hf_df = load_all_books()
    hf_df["grade_normalised"] = hf_df["Grade"].apply(normalize_grade)
    
    # 3. Define the 4 parameter configurations
    params = [
        {"name": "1200_book_stratified", "type": "stratified", "n": 200},
        {"name": "1200_grade_balanced", "type": "balanced", "n": 300},
        {"name": "5000_book_stratified", "type": "stratified", "n": 833},
        {"name": "5000_grade_balanced", "type": "balanced", "n": 1250},
    ]
    
    STAGE1_DIR.mkdir(parents=True, exist_ok=True)
    results_summary = {}
    
    # 4. Process each parameter configuration
    for p in params:
        pname = p["name"]
        ptype = p["type"]
        pn = p["n"]
        print(f"\n[{pname.upper()}] Sampling and Aligning...")
        
        if ptype == "stratified":
            eval_df = stratified_sample(hf_df, n_per_book=pn)
        else:
            eval_df = balanced_sample(hf_df, n_per_grade=pn)
            
        print(f"  Sampled {len(eval_df):,} query Hadiths")
        
        with timer(f"Aligning {pname}"):
            alignment = align_datasets(
                eval_df,
                sanadset_df,
                query_text_col="Arabic_Text",
                corpus_text_col="Matn",
            )
            
        aligned_count = int(alignment["is_match"].sum())
        quarantine_count = int(alignment["quarantine"].sum())
        match_rate = float(alignment["is_match"].mean() * 100)
        avg_sim = float(alignment[alignment["is_match"]]["similarity_score"].mean()) if aligned_count > 0 else 0.0
        
        print(f"  Results for {pname}:")
        print(f"    Target Queries:      {len(eval_df):,}")
        print(f"    Successfully Aligned:{aligned_count:,} ({match_rate:.2f}%)")
        print(f"    Quarantined:         {quarantine_count:,}")
        print(f"    Average Similarity:  {avg_sim:.2f}%")
        
        # Save output CSV and JSON alignments
        eval_out = STAGE1_DIR / f"kutub_sitta_eval_{pname}.csv"
        align_out = STAGE1_DIR / f"alignment_results_{pname}.json"
        quarantine_out = STAGE1_DIR / f"quarantine_{pname}.json"
        
        save_csv(eval_df, eval_out)
        save_json(alignment.to_dict(orient="records"), align_out)
        export_quarantine(eval_df, alignment, quarantine_out)
        
        results_summary[pname] = {
            "target": len(eval_df),
            "aligned": aligned_count,
            "quarantined": quarantine_count,
            "match_rate": match_rate,
            "avg_sim": avg_sim
        }

    # Save summary log
    with open(STAGE1_DIR / "alignment_4_params_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
        
    print("\n" + "=" * 70)
    print("ALL 4 DATASET PARAMETER ALIGNMENTS COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    build_and_align_all_4()
