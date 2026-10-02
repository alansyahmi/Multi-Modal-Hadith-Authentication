"""
Build a grade-balanced evaluation set of 5,000 hadiths from ALL available data.

Pipeline:
  1. Load ALL 33K+ HuggingFace hadiths with grades
  2. Normalize grades to 4 classes
  3. Match against Sanadset via rapidfuzz (no TF-IDF needed)
  4. Sample balanced by grade → 5,000 total
"""
import sys, re, ast, json, hashlib, time
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
sys.stdout.reconfigure(encoding="utf-8")

def norm_matn(text):
    if not text: return ""
    text = re.sub(r"[ً-ٌٍ-َّْٰٓ]", "", text)
    text = text.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    text = text.replace("ة","ه").replace("ى","ي")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── 1. Load ALL HuggingFace hadiths ──
print("Loading all HuggingFace hadiths...")
all_hf = []
import os
for f in sorted(os.listdir("Dataset/hadith_datasets HuggingFace")):
    if not f.endswith(".csv"): continue
    df = pd.read_csv(f"Dataset/hadith_datasets HuggingFace/{f}")
    for _, row in df.iterrows():
        all_hf.append({
            "book": f.replace(".csv","").replace("%20"," "),
            "text": str(row.get("Arabic_Text","")),
            "grade": str(row.get("Grade","")),
        })

print(f"  Total: {len(all_hf):,}")

# ── 2. Normalize grades ──
print("Normalizing grades...")
grade_map = {
    "sahih": "Sahih", "Sahih": "Sahih",
    "hasan": "Hasan", "Hasan": "Hasan",
    "da'if": "Daif", "Daif": "Daif", "daif": "Daif", "Da`if": "Daif",
    "maudu": "Matruk", "Mawdu": "Matruk", "Maudu": "Matruk",
    "munkar": "Daif", "shadh": "Daif",
}

for h in all_hf:
    raw = h["grade"]
    h["normalized"] = "Unknown"
    for key, val in grade_map.items():
        if key in raw:
            h["normalized"] = val
            break

# Count
dist = Counter(h["normalized"] for h in all_hf)
print(f"  Grade distribution:")
for g, c in dist.most_common():
    print(f"    {g:10s}: {c:,}")

# ── 3. Load Sanadset for matching ──
print("\nLoading Sanadset matns...")
sanad_data = []
for chunk in pd.read_csv(
    "Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv",
    usecols=["Matn", "Sanad"], chunksize=100_000, dtype_backend="pyarrow",
):
    for _, row in chunk.iterrows():
        m = row.get("Matn",""); s = row.get("Sanad","")
        if pd.isna(m) or pd.isna(s): continue
        nm = norm_matn(str(m))
        if len(nm) >= 15:
            sanad_data.append((nm, str(s)))

print(f"  {len(sanad_data):,} matns indexed")

# Build fingerprint index for fast lookup
print("Building fingerprint index...")
fp_index = defaultdict(list)
for idx, (nm, _) in enumerate(sanad_data):
    fp_index[nm[:80]].append(idx)

print(f"  {len(fp_index):,} fingerprints")

# ── 4. Match HF hadiths to Sanadset ──
print("\nMatching HuggingFace hadiths to Sanadset...")
matched_by_grade = defaultdict(list)
total = 0

# Process by grade to ensure balanced sampling later
for h in all_hf:
    if h["normalized"] == "Unknown": continue
    total += 1

# Take graded hadiths only
graded = [h for h in all_hf if h["normalized"] != "Unknown"]
print(f"  Graded hadiths: {len(graded):,}")

t0 = time.time()
matched_count = 0
for h in graded:
    hf_norm = norm_matn(h["text"])
    if len(hf_norm) < 15: continue
    fp = hf_norm[:80]

    # Find candidate rows by fingerprint
    candidates = fp_index.get(fp, [])
    best_score = 0
    best_idx = None

    for idx in candidates[:10]:  # Check top 10 by fingerprint
        sanad_norm = sanad_data[idx][0]
        score = fuzz.partial_ratio(hf_norm[:100], sanad_norm[:100])
        if score > best_score and score >= 80:
            best_score = score
            best_idx = idx

    if best_idx is not None:
        matched_by_grade[h["normalized"]].append({
            "hf_text": h["text"],
            "sanad_matn": sanad_data[best_idx][0],
            "sanad_str": sanad_data[best_idx][1],
            "similarity": best_score,
            "book": h["book"],
        })
        matched_count += 1

    if matched_count % 2000 == 0:
        el = time.time() - t0
        rate = matched_count / max(el, 1)
        print(f"  Matched: {matched_count:,}/{len(graded)} ({rate:.0f}/s)")

ttime = time.time() - t0
print(f"\n  Matched in {ttime:.0f}s")

# ── 5. Grade-balanced sampling ──
print("\nGrade-balanced sampling...")
target_per_grade = 1250  # 4 grades × 1250 = 5000
rng = np.random.RandomState(42)

eval_set = []
for grade in ["Sahih", "Hasan", "Daif", "Matruk"]:
    pool = matched_by_grade.get(grade, [])
    available = len(pool)
    sample_size = min(target_per_grade, available)
    if sample_size < target_per_grade:
        print(f"  ⚠️ {grade}: only {available} available, taking all {available}")
    sampled = rng.choice(pool, sample_size, replace=False).tolist() if isinstance(pool, list) else rng.choice(list(pool), sample_size, replace=False).tolist()
    for s in sampled:
        s["grade"] = grade
    eval_set.extend(sampled)

rng.shuffle(eval_set)
print(f"\n  Final evaluation set: {len(eval_set):,} hadiths")
for g in ["Sahih", "Hasan", "Daif", "Matruk"]:
    cnt = sum(1 for e in eval_set if e["grade"] == g)
    print(f"    {g:10s}: {cnt}")

# ── 6. Save ──
print("\nSaving...")
out = {"metadata": {"total": len(eval_set), "balanced_by": "grade", "target_per_grade": target_per_grade},
       "hadiths": [{
           "grade": e["grade"],
           "book": e["book"],
           "matn_norm": e["sanad_matn"][:100],
           "sanad": e["sanad_str"],
           "similarity": e["similarity"],
       } for e in eval_set]}
with open("output/final_evaluation/balanced_eval_set.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"  Saved: output/final_evaluation/balanced_eval_set.json ({len(eval_set)} hadiths)")
