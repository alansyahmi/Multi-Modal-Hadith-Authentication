"""
Display controversial hadiths from the 6 Kutub as-Sitta:
  - Fabricated (Mawḍūʿ) and rejected (Munkar) hadiths
  - Weak (Ḍaʿīf) hadiths with diverse narrator chains
  - Hadiths where narrator predictions show wide variance (mixed reliability)
  - Comparison: Sahih vs Daif chains side by side
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

good = alignment[alignment["is_match"] & ~alignment["quarantine"]]

with open(r"output/stage2/node_names.json", encoding="utf-8") as f:
    id_to_name = json.load(f)

# Load predictions
pred_path = r"output/stage3/narrator_predictions.csv"
predictions_df = pd.read_csv(pred_path) if Path(pred_path).exists() else None

# Build quick lookup: clean_name → predicted_score
name_to_pred = {}
if predictions_df is not None:
    for _, row in predictions_df.iterrows():
        nid = str(int(row["node_id"]))
        if nid in id_to_name:
            name_clean = re.sub(r'[ً-ٌٍ-َّْٰ]', '', id_to_name[nid])
            name_to_pred[name_clean] = float(row["predicted_score"])


def get_chain_for_query(qidx):
    """Get narrator chain and predictions for a query index."""
    match = good[good["query_idx"] == qidx]
    if match.empty:
        return None, None, None
    row = match.iloc[0]
    corpus_idx = int(row["corpus_idx"])
    if corpus_idx >= len(sanadset):
        return None, None, None

    sanad_row = sanadset.iloc[corpus_idx]
    matn = str(sanad_row.get("Matn", ""))
    sanad_text = str(sanad_row.get("Sanad", ""))
    try:
        chain = ast.literal_eval(sanad_text) if isinstance(sanad_text, str) else []
    except:
        chain = []

    # Score each narrator
    scored_chain = []
    for name in chain:
        name_clean = re.sub(r'[ً-ٌٍ-َّْٰ]', '', name.strip())
        pred = name_to_pred.get(name_clean, None)
        scored_chain.append((name_clean, pred))

    return matn, scored_chain, row


def print_hadith(info, matn, chain, show_matn=True):
    """Print a formatted hadith entry."""
    print(f"\n  Grade: {info['Grade']}")
    print(f"  Book:  {info['Book']}")
    print(f"  Kitab: {info['book_key']}")

    if show_matn and matn:
        snippet = matn[:350] + "..." if len(matn) > 350 else matn
        print(f"\n  --- Matn ---")
        for line in [snippet[i:i+80] for i in range(0, len(snippet), 80)]:
            print(f"  {line}")

    if chain:
        scores = [s for _, s in chain if s is not None]
        if scores:
            spread = max(scores) - min(scores)
            avg = np.mean(scores)
        else:
            spread = 0
            avg = 0

        print(f"\n  --- Chain ({len(chain)} narrators, mean={avg:.3f}, spread={spread:.3f}) ---")
        for i, (name, pred) in enumerate(chain):
            if pred is not None:
                if pred >= 0.85: cat = "ثقة"
                elif pred >= 0.60: cat = "صدوق"
                elif pred >= 0.35: cat = "مقبول"
                elif pred >= 0.15: cat = "ضعيف"
                else: cat = "متروك"
                p_str = f"{pred:.3f} → {cat}"
            else:
                p_str = "(no prediction)"
            print(f"  {i+1}. {name}")
            print(f"     {p_str}")
    else:
        print(f"\n  --- Chain: (empty) ---")


# ── Find controversial hadiths ──
print(f"\n{'='*70}")
print("CONTROVERSIAL & MIXED-RELIABILITY HADITHS")
print(f"{'='*70}")

# ─── SECTION 1: Fabricated (Mawḍūʿ) ───
print(f"\n{'─'*70}")
print("SECTION 1: FABRICATED (MAWḌŪʿ) & REJECTED (MUNKAR) HADITHS")
print(f"{'─'*70}")

controversial_grades = ["Maudu' (Darussalam)", "Maudu (fabricated) (Darussalam)",
                        "Munkar (Al-Albani)", "Da'if Jiddan (Al-Albani)"]
for target_grade in controversial_grades:
    for qidx, info in query_df.iterrows():
        if info["Grade"] == target_grade:
            matn, chain, align_row = get_chain_for_query(qidx)
            if chain and len(chain) >= 2:
                print_hadith(info, matn, chain)
                break

# ─── SECTION 2: Daif hadiths with the most narrator diversity ───
print(f"\n{'─'*70}")
print("SECTION 2: ḌAʿĪF HADITHS WITH DIVERSE NARRATOR RELIABILITY")
print(f"{'─'*70}")

daif_entries = query_df[query_df["grade_normalised"] == "Ḍaʿīf"]
daif_with_spread = []

for qidx, info in daif_entries.iterrows():
    matn, chain, align_row = get_chain_for_query(qidx)
    if chain and len(chain) >= 3:
        scores = [s for _, s in chain if s is not None]
        if len(scores) >= 3:
            spread = max(scores) - min(scores)
            daif_with_spread.append((spread, qidx, info, matn, chain))

daif_with_spread.sort(key=lambda x: x[0], reverse=True)

for i, (spread, qidx, info, matn, chain) in enumerate(daif_with_spread[:3]):
    print(f"\n  [{i+1}] HIGH-VARIANCE ḌAʿĪF CHAIN (spread={spread:.3f})")
    print_hadith(info, matn, chain)

# ─── SECTION 3: Sahih vs Daif comparison (same narrator appearing in both) ───
print(f"\n{'─'*70}")
print("SECTION 3: NARRATOR APPEARING IN BOTH ṢAḤĪḤ AND ḌAʿĪF CHAINS")
print(f"{'─'*70}")

# Build narrator → (sahih_count, daif_count) mapping
narrator_grade_appearances = defaultdict(lambda: {"sahih": 0, "daif": 0, "hasan": 0})

for qidx, info in query_df.iterrows():
    matn, chain, align_row = get_chain_for_query(qidx)
    if not chain:
        continue
    gn = info.get("grade_normalised", "")
    for name, pred in chain:
        name_clean = re.sub(r'[ً-ٌٍ-َّْٰ]', '', name)
        if gn == "Ṣaḥīḥ":
            narrator_grade_appearances[name_clean]["sahih"] += 1
        elif gn == "Ḍaʿīf":
            narrator_grade_appearances[name_clean]["daif"] += 1
        elif gn == "Ḥasan":
            narrator_grade_appearances[name_clean]["hasan"] += 1

# Find narrators appearing in BOTH sahih and daif chains
cross_narrators = [(name, counts) for name, counts in narrator_grade_appearances.items()
                   if counts["sahih"] > 0 and counts["daif"] > 0]
cross_narrators.sort(key=lambda x: x[1]["sahih"] + x[1]["daif"], reverse=True)

print(f"\n  Narrators appearing in both Ṣaḥīḥ AND Ḍaʿīf hadiths: {len(cross_narrators)}")
for name, counts in cross_narrators[:8]:
    pred = name_to_pred.get(name, None)
    pred_str = f"{pred:.3f}" if pred else "N/A"
    print(f"  - {name[:50]:<50s}   pred={pred_str}   ṣaḥīḥ={counts['sahih']}  ḍaʿīf={counts['daif']}")

# ─── SECTION 4: Best and worst predicted narrators appearing in our sampled hadiths ───
print(f"\n{'─'*70}")
print("SECTION 4: WEAKEST & STRONGEST PREDICTED NARRATORS IN SAMPLED CHAINS")
print(f"{'─'*70}")

# Collect all narrator predictions from aligned chains
all_scored = []
seen = set()
for qidx, info in query_df.iterrows():
    matn, chain, align_row = get_chain_for_query(qidx)
    if not chain:
        continue
    for name, pred in chain:
        if pred is not None and name not in seen:
            all_scored.append((pred, name, info.get("grade_normalised", "")))
            seen.add(name)

all_scored.sort(key=lambda x: x[0])

print(f"\n  --- Weakest predicted (bottom 5) ---")
for pred, name, grade in all_scored[:5]:
    print(f"  {name:<50s} pred={pred:.3f}  (appears in {grade} hadith)")

print(f"\n  --- Strongest predicted (top 5) ---")
for pred, name, grade in all_scored[-5:]:
    print(f"  {name:<50s} pred={pred:.3f}  (appears in {grade} hadith)")

# ─── SECTION 5: Specific hadith with chain containing a weak narrator ───
print(f"\n{'─'*70}")
print("SECTION 5: HADITHS WITH WEAK NARRATOR IN CHAIN (pred < 0.35)")
print(f"{'─'*70}")

found = 0
for qidx, info in query_df.iterrows():
    if found >= 3:
        break
    matn, chain, align_row = get_chain_for_query(qidx)
    if not chain:
        continue
    scores = [s for _, s in chain if s is not None]
    if any(s < 0.35 for s in scores) and len(scores) >= 3:
        # Show this one
        weak_names = [(n, s) for n, s in chain if s is not None and s < 0.35]
        weak_str = ", ".join(f"{n} ({s:.3f})" for n, s in weak_names)
        print(f"\n  [{found+1}] Chain with weak narrator(s): {weak_str}")
        print_hadith(info, matn, chain, show_matn=True)
        found += 1

print(f"\n{'='*70}")
print("END OF CONTROVERSIAL HADITH ANALYSIS")
print(f"{'='*70}")
