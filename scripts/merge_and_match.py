"""
Merge Tahdhib al-Kamal + Taqrib al-Tahdhib into unified narrator DB,
then match against 1,233 eval set narrators.
"""
import json, sys, ast
from pathlib import Path
from collections import defaultdict, Counter
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rapidfuzz.fuzz import partial_ratio
from src.preprocessing.arabic_normalizer import normalise_hadith_text
from src.shamela.taqrib_parser import parse_taqrib_volume

# ── Load Tahdhib al-Kamal entries ──
with open(r"output/shamela/narrator_encyclopedia.json", encoding="utf-8") as f:
    tahdhib = json.load(f)
print(f"Tahdhib al-Kamal: {len(tahdhib):,} entries")

# ── Parse Taqrib al-Tahdhib with proper parser ──
taqrib_text = Path(r"output/book_8609/raw_volume_01.txt").read_text(encoding="utf-8")
taqrib_entries = parse_taqrib_volume(taqrib_text)
print(f"Taqrib al-Tahdhib: {len(taqrib_entries):,} entries")

# Build Taqrib lookup by normalized name
taqrib_lookup = {}
for e in taqrib_entries:
    norm = normalise_hadith_text(e.name, for_matching=True)
    if norm and e.grade:
        taqrib_lookup[norm] = e

print(f"Taqrib with grades: {len(taqrib_lookup):,}")

# ── Load eval narrators ──
with open(r"output/stage1/alignment_results.json") as f:
    alignment = json.load(f)
import pandas as pd
sanadset = pd.read_csv(r"output/stage1/sanadset_cleaned.csv")
matched_hadiths = [a for a in alignment if a.get("is_match")]

all_narrators = set()
for a in matched_hadiths:
    cidx = a["corpus_idx"]
    if cidx is not None:
        try:
            row = sanadset.loc[cidx]
            chain = ast.literal_eval(row["Sanad"]) if isinstance(row["Sanad"], str) else []
            for name in chain:
                all_narrators.add(name.strip())
        except:
            pass

print(f"\nEval set narrators: {len(all_narrators)}")

# ── Match ──
results = {}
found_grade = 0
found_death = 0
found_tabaqa = 0
not_found = 0

for name in all_narrators:
    name_norm = normalise_hadith_text(name, for_matching=True)
    if not name_norm or len(name_norm) < 3:
        continue

    best_grade = ""
    best_death = ""
    best_tabaqa = ""
    best_source = ""
    best_name = ""

    # 1. Try exact Taqrib match first (most reliable grades)
    if name_norm in taqrib_lookup:
        te = taqrib_lookup[name_norm]
        best_grade = te.grade
        best_death = te.death_year_text
        best_tabaqa = te.tabaqa
        best_source = "taqrib_exact"
        best_name = te.name

    # 2. Try fuzzy Taqrib match
    if not best_grade:
        best_score = 0
        for tnorm, te in taqrib_lookup.items():
            score = partial_ratio(name_norm, tnorm)
            if score > best_score and score >= 85:
                best_score = score
                best_grade = te.grade
                best_death = te.death_year_text
                best_tabaqa = te.tabaqa
                best_source = f"taqrib_fuzzy_{score:.0f}"
                best_name = te.name

    # 3. Try Tahdhib for death/teacher/student supplement
    best_tahdhib = None
    best_t_score = 0
    for e in tahdhib:
        en = normalise_hadith_text(e.get("name", ""), for_matching=True)
        score = partial_ratio(name_norm, en)
        if score > best_t_score and score >= 85:
            best_t_score = score
            best_tahdhib = e

    if best_tahdhib:
        if not best_death and best_tahdhib.get("death_info"):
            best_death = best_tahdhib["death_info"]
        if best_tahdhib.get("grades") and not best_grade:
            best_grade = best_tahdhib["grades"][0]

    if best_grade or best_death or best_tabaqa:
        results[name] = {
            "status": best_source,
            "matched_name": best_name,
            "grade": best_grade,
            "grade_source": best_source,
            "death_info": best_death,
            "tabaqa": best_tabaqa,
            "tahdhib_teachers": best_tahdhib.get("teachers", [])[:5] if best_tahdhib else [],
            "tahdhib_students": best_tahdhib.get("students", [])[:5] if best_tahdhib else [],
        }
        if best_grade:
            found_grade += 1
        if best_death:
            found_death += 1
        if best_tabaqa:
            found_tabaqa += 1
    else:
        results[name] = {"status": "not_found"}
        not_found += 1

total = found_grade + sum(1 for r in results.values() if r["status"] == "not_found")
print(f"\n=== Final Results ===")
print(f"Total narrators:    {total}")
print(f"With GRADE:         {found_grade} ({found_grade/total*100:.1f}%)")
print(f"With DEATH:         {found_death} ({found_death/total*100:.1f}%)")
print(f"With TABAQA:        {found_tabaqa} ({found_tabaqa/total*100:.1f}%)")
print(f"NOT FOUND:          {not_found} ({not_found/total*100:.1f}%)")

# Grade distribution
grade_dist = Counter(r["grade"] for r in results.values() if r.get("grade"))
print(f"\nGrade distribution ({len(grade_dist)} types):")
for g, c in grade_dist.most_common(25):
    print(f"  {g:20s}: {c:>5}")

# Source distribution
source_dist = Counter(r["grade_source"] for r in results.values() if r.get("grade"))
print(f"\nSource distribution:")
for s, c in source_dist.most_common():
    print(f"  {s}: {c}")

# Save
out_path = Path(r"output/stage2/narrator_grades.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
# Convert defaultdict to plain dict for JSON
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {out_path}")
