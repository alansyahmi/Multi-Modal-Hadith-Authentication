"""
Extract structured biographies for all matched narrators.

Reads matched_targets.json, finds the page file for each narrator,
and parses their biography entry using the targeted parser.
"""
import json, sys, re
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shamela.targeted_parser import parse_narrator_page, BioEntry

BASE = Path(r"c:\Projects\FYP Project\output\shamela\tahdhib_al_kamal")
HTML_DIR = BASE / "pages"
TARGET_DIR = BASE / "target_pages"
OUTPUT = BASE / "narrator_bios.json"

with open(BASE / "matched_targets.json", encoding="utf-8") as f:
    matched = json.load(f)

def find_page_file(page_num: int) -> Path | None:
    """Find .txt or .html file for a given page number."""
    for ext, d in [(".txt", TARGET_DIR), (".html", HTML_DIR)]:
        p = d / f"page_{page_num:05d}{ext}"
        if p.exists():
            return p
    return None

results = []
found = 0
not_found = 0
no_entry = 0
stats = {"high": 0, "medium": 0, "low": 0}

total = len(matched)
for i, (name, page_num) in enumerate(matched.items()):
    if (i + 1) % 200 == 0:
        print(f"  [{i+1}/{total}] Processed... (found={found}, no_file={not_found}, no_entry={no_entry})")

    page_file = find_page_file(page_num)
    if not page_file:
        not_found += 1
        results.append({
            "narrator_name": name,
            "page_number": page_num,
            "status": "page_not_found",
        })
        continue

    entry = parse_narrator_page(page_file, name, page_num)
    if entry is None:
        no_entry += 1
        results.append({
            "narrator_name": name,
            "page_number": page_num,
            "status": "entry_not_found",
        })
        continue

    found += 1
    stats[entry.confidence] += 1
    results.append({
        "narrator_name": entry.narrator_name,
        "page_number": entry.page_number,
        "canonical_name": entry.canonical_name,
        "status": "parsed",
        "confidence": entry.confidence,
        "teachers": entry.teachers,
        "students": entry.students,
        "grades": entry.grades,
        "death_info": entry.death_info,
        "full_text": entry.full_text[:3000],  # Cap at 3000 chars
    })

# Write output
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n{'='*50}")
print(f"EXTRACTION COMPLETE")
print(f"{'='*50}")
print(f"Total narrators: {total}")
print(f"Parsed:          {found} ({found/total*100:.1f}%)")
print(f"  High:    {stats['high']}")
print(f"  Medium:  {stats['medium']}")
print(f"  Low:     {stats['low']}")
print(f"No page file:    {not_found}")
print(f"No entry found:  {no_entry}")
print(f"Output: {OUTPUT}")

# Show 5 sample high-confidence entries
high = [r for r in results if r.get("confidence") == "high"][:5]
print(f"\n=== Sample High-Confidence Entries ===")
for r in high:
    print(f"\n  {r['narrator_name'][:80]}")
    print(f"  Canonical: {r['canonical_name'][:100]}")
    print(f"  Teachers: {r.get('teachers',[])[:3]}")
    print(f"  Students: {r.get('students',[])[:3]}")
    print(f"  Grades:   {r.get('grades',[])[:3]}")
    print(f"  Death:    {r.get('death_info','')[:100]}")
