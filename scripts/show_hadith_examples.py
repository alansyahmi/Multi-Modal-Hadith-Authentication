"""
Display 1 hadith example from each of the 6 Kutub as-Sitta books,
with narrator chain, grade, and GNN predicted reliability scores.

Shows:
  - The Arabic matn text (content)
  - The narrator chain (sanad)
  - The canonical grade from the book
  - Per-narrator reliability predictions from our trained models
"""
import sys, json, re, ast
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

# ── Load data ──
print("Loading data...")
query_df = pd.read_csv(r"output/stage1/kutub_sitta_evaluation.csv")
alignment = pd.read_json(r"output/stage1/alignment_results.json")
sanadset = pd.read_csv(r"output/stage1/sanadset_cleaned.csv", nrows=200000)

# Only successful matches
good = alignment[alignment["is_match"] & ~alignment["quarantine"]]

# Map query_idx → book
query_idx_to_book = {}
for idx, row in query_df.iterrows():
    query_idx_to_book[idx] = {
        "book": row["Book"],
        "book_key": row["book_key"],
        "grade": row["Grade"],
        "grade_normalised": row.get("grade_normalised", ""),
        "arabic_text": row["Arabic_Text"],
        "english_text": row.get("English_Text", ""),
    }

# Load node names for narrator lookup
with open(r"output/stage2/node_names.json", encoding="utf-8") as f:
    id_to_name = json.load(f)

# Load predictions (try BERT first, then lexicon, then base)
pred_paths = [
    (r"output/stage3/predictions_bert.csv", "BERT"),
    (r"output/stage3/predictions_lexicon.csv", "Lexicon"),
    (r"output/stage3/narrator_predictions.csv", "Base"),
]
predictions_df = None
pred_source = None
for p, src in pred_paths:
    if Path(p).exists():
        predictions_df = pd.read_csv(p)
        pred_source = src
        break

if predictions_df is not None:
    print(f"Using predictions from: {pred_source}")

# ── Find 1 hadith per kitab ──
TARGET_BOOKS = ["bukhari", "muslim", "abudawud", "tirmidhi", "nasai", "ibnmajah"]

print(f"\n{'='*70}")
print(f"HADITH EXAMPLES FROM THE 6 KUTUB AS-SITTA")
print(f"{'='*70}")

for target_book in TARGET_BOOKS:
    print(f"\n{'─'*70}")
    print(f"BOOK: {target_book.upper()}")
    print(f"{'─'*70}")

    # Find aligned records for this book
    candidate_hadiths = []
    for _, row in good.iterrows():
        qidx = int(row["query_idx"])
        info = query_idx_to_book.get(qidx)
        if info and info["book_key"] == target_book:
            candidate_hadiths.append((qidx, row))

    if not candidate_hadiths:
        print(f"  No aligned hadith found for {target_book}")
        continue

    # Pick the first one (or one with good similarity)
    qidx, align_row = candidate_hadiths[0]
    info = query_idx_to_book[qidx]
    corpus_idx = int(align_row["corpus_idx"])

    # Get Sanadset row
    if corpus_idx < len(sanadset):
        sanad_row = sanadset.iloc[corpus_idx]
    else:
        sanad_row = sanadset.iloc[0]

    # ── Extract and display ──
    matn_text = str(sanad_row.get("Matn", ""))
    sanad_text = str(sanad_row.get("Sanad", ""))

    # Parse narrator chain
    try:
        chain = ast.literal_eval(sanad_text) if isinstance(sanad_text, str) else []
    except (ValueError, SyntaxError):
        chain = []

    print(f"\n  Grade: {info['grade']} ({info['grade_normalised']})")
    print(f"  Book:  {info['book']}")

    # Matn (first 400 chars)
    matn_snippet = matn_text[:400] + "..." if len(matn_text) > 400 else matn_text
    print(f"\n  --- Matn (Hadith Content) ---")
    for line in [matn_snippet[i:i+80] for i in range(0, len(matn_snippet), 80)]:
        print(f"  {line}")

    # Chain with predictions
    if chain:
        print(f"\n  --- Narrator Chain ({len(chain)} narrators) ---")
    else:
        print(f"\n  --- Narrator Chain: (empty/could not parse) ---")

    for i, narrator in enumerate(chain):
        narrator_clean = re.sub(r'[ً-ٌٍ-َّْٰ]', '', narrator.strip())

        # Look up prediction
        pred_score = None
        if predictions_df is not None and len(predictions_df) > 0:
            # Match by node name
            for node_id_str, name in id_to_name.items():
                name_clean = re.sub(r'[ً-ٌٍ-َّْٰ]', '', name)
                if name_clean == narrator_clean:
                    node_id = int(node_id_str)
                    match_row = predictions_df[predictions_df["node_id"] == node_id]
                    if len(match_row) > 0:
                        pred_score = float(match_row["predicted_score"].iloc[0])
                    break

        # Determine predicted category
        if pred_score is not None:
            if pred_score >= 0.85: cat = "ثقة (Thiqah)"
            elif pred_score >= 0.60: cat = "صدوق (Ṣadūq)"
            elif pred_score >= 0.35: cat = "مقبول (Maqbūl)"
            elif pred_score >= 0.15: cat = "ضعيف (Ḍaʿīf)"
            else: cat = "متروك (Matrūk)"
            pred_str = f"{pred_score:.3f} → {cat}"
        else:
            pred_str = "(no prediction)"

        print(f"  {i+1}. {narrator_clean}")
        print(f"     Reliability: {pred_str}")

    # English translation
    eng = info['english_text'][:300]
    if eng:
        print(f"\n  --- English Translation ---")
        for line in [eng[i:i+80] for i in range(0, len(eng), 80)]:
            print(f"  {line}")

    # Similarity info
    print(f"\n  --- Alignment ---")
    print(f"  TF-IDF score: {align_row['tfidf_score']:.2f}")
    print(f"  Similarity:   {align_row['similarity_score']:.1f}%")

print(f"\n{'='*70}")
print("END OF EXAMPLES")
print(f"{'='*70}")
