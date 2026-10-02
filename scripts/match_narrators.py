"""
Match encyclopedia narrator entries against the 1,233 eval set narrators.
Uses partial_ratio fuzzy matching (same strategy as Stage 1 alignment).
"""
import json, sys, ast
from pathlib import Path
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rapidfuzz.fuzz import partial_ratio
from src.preprocessing.arabic_normalizer import normalise_hadith_text

# Load encyclopedia
with open(r"output/shamela/narrator_encyclopedia.json", encoding="utf-8") as f:
    encyclopedia = json.load(f)

print(f"Encyclopedia: {len(encyclopedia):,} entries")

# Build lookup: normalize encyclopedia names
encyc_names = {}
for e in encyclopedia:
    norm = normalise_hadith_text(e["name"], for_matching=True)
    if norm:
        encyc_names[norm] = e

# Load aligned narrators from Stage 1
with open(r"output/stage1/alignment_results.json") as f:
    alignment = json.load(f)
import pandas as pd
sanadset = pd.read_csv(r"output/stage1/sanadset_cleaned.csv")

# Extract unique narrators from matched hadiths
matched = [a for a in alignment if a.get("is_match")]
print(f"Matched hadiths: {len(matched)}")

all_narrators = set()
for a in matched:
    cidx = a["corpus_idx"]
    if cidx is not None:
        try:
            row = sanadset.loc[cidx]
            chain = ast.literal_eval(row["Sanad"]) if isinstance(row["Sanad"], str) else []
            for name in chain:
                all_narrators.add(name.strip())
        except:
            pass

print(f"Unique narrators in eval set: {len(all_narrators)}")

# Match each narrator against encyclopedia
results = {}
found = 0
not_found = 0

for name in all_narrators:
    name_norm = normalise_hadith_text(name, for_matching=True)
    if not name_norm or len(name_norm) < 3:
        continue

    # Try exact lookup first
    if name_norm in encyc_names:
        results[name] = {"status": "exact", **encyc_names[name_norm]}
        found += 1
        continue

    # Fuzzy match against all encyclopedia names
    best_score = 0
    best_entry = None
    for enc_norm, entry in encyc_names.items():
        score = partial_ratio(name_norm, enc_norm)
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_score >= 80 and best_entry:
        results[name] = {"status": f"fuzzy_{best_score:.0f}", **best_entry}
        found += 1
    else:
        results[name] = {"status": "not_found", "best_score": best_score}
        not_found += 1

print(f"\nMatched: {found}/{found+not_found} ({found/(found+not_found)*100:.1f}%)")
print(f"Not found: {not_found}")

# Stats on matched entries
grade_count = sum(1 for r in results.values() if r.get("grades"))
death_count = sum(1 for r in results.values() if r.get("death_info"))
print(f"With grades: {grade_count}")
print(f"With death:  {death_count}")

# Grade distribution
grade_dist = defaultdict(int)
for r in results.values():
    for g in r.get("grades", []):
        grade_dist[g] += 1
print(f"\nGrade distribution:")
for g, c in sorted(grade_dist.items(), key=lambda x: -x[1]):
    print(f"  {g}: {c}")

# Save
out_path = Path(r"output/shamela/narrator_matches.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nOutput: {out_path}")

# Sample matches
print(f"\n=== Sample matches ===")
for name in list(results.keys())[:8]:
    r = results[name]
    print(f"\n{name}")
    print(f"  Status: {r['status']}")
    print(f"  Name:   {r.get('name','')[:100]}")
    print(f"  Grades: {r.get('grades',[])[:3]}")
    print(f"  Death:  {r.get('death_info','')[:80]}")
