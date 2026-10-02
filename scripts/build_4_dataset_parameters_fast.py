"""Fast build and alignment for all 4 dataset parameter configurations."""
import sys
import os
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.huggingface_loader import load_all_books, stratified_sample, balanced_sample, normalize_grade
from src.utils.helpers import timer, save_csv, save_json
from src.config import SANADSET_CSV, STAGE1_DIR

def norm_matn(text):
    if not text: return ""
    text = re.sub(r"[ً-ٌٍ-َّْٰٓ]", "", text)
    text = text.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    text = text.replace("ة","ه").replace("ى","ي")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    print("=" * 70)
    print("  FAST ALIGNMENT PIPELINE FOR ALL 4 DATASET PARAMETERS")
    print("=" * 70)

    # 1. Load HuggingFace ground truth
    print("\n[1/3] Loading Kutub as-Sitta Ground Truth...")
    hf_df = load_all_books()
    hf_df["grade_normalised"] = hf_df["Grade"].apply(normalize_grade)

    # 2. Load Sanadset matns and build index
    print("\n[2/3] Loading Sanadset Matns and building fast fingerprint index...")
    sanad_records = []
    fp_index = {}
    
    chunk_idx = 0
    for chunk in pd.read_csv(SANADSET_CSV, usecols=["Matn", "Sanad"], chunksize=100_000, dtype_backend="pyarrow"):
        for row_idx, row in chunk.iterrows():
            m = row.get("Matn", ""); s = row.get("Sanad", "")
            if pd.isna(m) or pd.isna(s): continue
            nm = norm_matn(str(m))
            if len(nm) >= 15:
                global_idx = chunk_idx + row_idx
                sanad_records.append((global_idx, nm, str(s)))
                fp = nm[:60]
                if fp not in fp_index:
                    fp_index[fp] = global_idx
        chunk_idx += len(chunk)

    print(f"  Indexed {len(sanad_records):,} valid Sanadset matns with {len(fp_index):,} unique fingerprints")

    # Build fast map global_idx -> tuple
    idx_to_record = {r[0]: r for r in sanad_records}

    params = [
        {"name": "1200_book_stratified", "type": "stratified", "n": 200},
        {"name": "1200_grade_balanced", "type": "balanced", "n": 300},
        {"name": "5000_book_stratified", "type": "stratified", "n": 833},
        {"name": "5000_grade_balanced", "type": "balanced", "n": 1250},
    ]

    STAGE1_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for p in params:
        pname = p["name"]
        ptype = p["type"]
        pn = p["n"]
        print(f"\n[{pname.upper()}] Processing...")

        if ptype == "stratified":
            eval_df = stratified_sample(hf_df, n_per_book=pn)
        else:
            eval_df = balanced_sample(hf_df, n_per_grade=pn)

        align_records = []
        aligned_count = 0
        quarantine_count = 0

        for q_idx, row in eval_df.iterrows():
            text = str(row.get("Arabic_Text", ""))
            q_norm = norm_matn(text)
            if len(q_norm) < 15:
                quarantine_count += 1
                align_records.append({"query_idx": q_idx, "corpus_idx": -1, "similarity_score": 0.0, "is_match": False, "quarantine": True})
                continue

            fp = q_norm[:60]
            best_score = 0
            best_ci = -1

            if fp in fp_index:
                ci = fp_index[fp]
                rec = idx_to_record[ci]
                score = fuzz.ratio(q_norm[:100], rec[1][:100])
                if score >= 80:
                    best_score = score
                    best_ci = ci

            if best_ci != -1 and best_score >= 80:
                aligned_count += 1
                align_records.append({"query_idx": q_idx, "corpus_idx": float(best_ci), "similarity_score": float(best_score), "is_match": True, "quarantine": False})
            else:
                quarantine_count += 1
                align_records.append({"query_idx": q_idx, "corpus_idx": -1, "similarity_score": float(best_score), "is_match": False, "quarantine": True})

        match_rate = (aligned_count / len(eval_df)) * 100
        avg_sim = np.mean([r["similarity_score"] for r in align_records if r["is_match"]]) if aligned_count > 0 else 0.0

        print(f"  Target Queries:      {len(eval_df):,}")
        print(f"  Successfully Aligned:{aligned_count:,} ({match_rate:.2f}%)")
        print(f"  Quarantined:         {quarantine_count:,}")
        print(f"  Average Similarity:  {avg_sim:.2f}%")

        eval_out = STAGE1_DIR / f"kutub_sitta_eval_{pname}.csv"
        align_out = STAGE1_DIR / f"alignment_results_{pname}.json"

        save_csv(eval_df, eval_out)
        save_json(align_records, align_out)

        summary[pname] = {
            "target": len(eval_df),
            "aligned": aligned_count,
            "quarantined": quarantine_count,
            "match_rate": match_rate,
            "avg_sim": avg_sim
        }

    with open(STAGE1_DIR / "alignment_4_params_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("FAST ALIGNMENT COMPLETE FOR ALL 4 PARAMETERS!")
    print("=" * 70)

if __name__ == "__main__":
    main()
