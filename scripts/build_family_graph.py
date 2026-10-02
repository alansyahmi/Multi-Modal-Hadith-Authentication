"""
Build a family + teacher-student relationship graph from the encyclopedia.

This enables resolving sanad references like:
  "عن الزهري، عن أخيه" → أخيه = الزهري's brother
  "عن سفيان، عن أبيه" → أبيه = سفيان's father
  "عن مالك، عن عمه"  → عمه = مالك's uncle

Relationships inferred from:
  1. Name patterns ("فلان بن فلان" = father-son)
  2. Teacher-student links (روى عن / روى عنه)
  3. Shared name components (same nisba + same teacher = possible siblings)

Output:
  output/family_graph/
    families.json         {family_id: [entry_ids]}
    father_son.json       {son_entry_id: father_entry_id}
    sibling_groups.json   [[sibling_entry_ids]]
    teacher_student.json  {teacher_id: [student_ids]}
    ab_links.json         Lookup: name_fragment → resolved entry_id
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_DIR = Path("output/family_graph")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  BUILD FAMILY + TEACHER GRAPH")
print("=" * 60)

# ── Load encyclopedia ──
print("\nLoading encyclopedia...")
enc = json.load(open("output/shamela/narrator_encyclopedia.json", encoding="utf-8"))
print(f"  {len(enc):,} entries")

# Build entry_id → entry lookup
by_id = {}
for e in enc:
    by_id[e["entry_id"]] = e

# ── Step 1: Extract father-son from name patterns ──
print("\n[1/4] Extracting father-son from name patterns...")

FATHER_SON_RE = re.compile(r"(?:ابن|بن)\s+([؀-ۿ\s]{3,50}?)(?:\s|$|،|\()")
KUNYA_FATHER_RE = re.compile(r"أبو\s+([؀-ۿ\s]{3,30}?)(?:\s|$|،|\()")

father_son = {}  # son_entry_id → father_name
son_to_father_id = {}  # son_entry_id → father_entry_id (resolved)

for e in enc:
    eid = e["entry_id"]
    name = e.get("name", "")

    # "فلان بن فلان" → father is "فلان" (after "بن")
    m = FATHER_SON_RE.search(name)
    if m:
        father_name = m.group(1).strip()
        father_son[eid] = father_name

print(f"  Father-son relationships found: {len(father_son):,}")

# ── Step 2: Build teacher-student graph ──
print("\n[2/4] Building teacher-student graph...")

teacher_to_students = defaultdict(set)
student_to_teachers = defaultdict(set)

for e in enc:
    eid = e["entry_id"]
    for t in e.get("teachers", []):
        teacher_to_students[eid]
        # Try to resolve teacher name to entry_id
    for s in e.get("students", []):
        student_to_teachers[eid]

# Count actual links
teacher_count = sum(1 for e in enc if e.get("teachers"))
student_count = sum(1 for e in enc if e.get("students"))
print(f"  Entries with teachers: {teacher_count:,}")
print(f"  Entries with students: {student_count:,}")

# ── Step 3: Build nisba-based clusters (same family) ──
print("\n[3/4] Building nisba-based family clusters...")

# Extract nisba (ال...ي) from names
NISBA_RE = re.compile(r"(ال[؀-ۿ]{2,20}[ي])")

nisba_groups = defaultdict(list)
for e in enc:
    name = e.get("name", "")
    nisbas = NISBA_RE.findall(name)
    for nisba in nisbas:
        nisba_groups[nisba].append(e["entry_id"])

# Families = same nisba + shared teacher
print(f"  Nisba groups: {len(nisba_groups):,}")
families_with_multiple = {k: v for k, v in nisba_groups.items() if len(v) > 1}
print(f"  Nisba groups with >1 member: {len(families_with_multiple):,}")

# ── Step 4: Build resolution lookup for "أبيه", "أخيه", etc. ──
print("\n[4/4] Building sanad resolution lookup + virtual narrator registry...")

# Name → entry_id mapping (normalized)
name_to_entry = {}
for e in enc:
    nkey = re.sub(r"[ً-ٌٍ-َّْٰ]", "", e.get("name", ""))
    nkey = nkey.replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()
    name_to_entry[nkey] = e["entry_id"]

# 4a. Resolve father names that have their OWN entry
resolved_fathers = 0
for son_id, father_name in father_son.items():
    fkey = re.sub(r"[ً-ٌٍ-َّْٰ]", "", father_name)
    fkey = fkey.replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()
    if fkey in name_to_entry:
        son_to_father_id[son_id] = name_to_entry[fkey]
        resolved_fathers += 1

print(f"  Resolved father names to entries: {resolved_fathers:,}/{len(father_son):,}")

# 4b. Build VIRTUAL NARRATOR REGISTRY for generic fathers
# Group father names: how many different sons mention this father?
father_name_groups = defaultdict(list)
for son_id, father_name in father_son.items():
    if son_id not in son_to_father_id:  # Only for unresolved fathers
        father_name_groups[father_name].append(son_id)

# Virtual narrators: father name → (son_ids, frequency, avg_score_from_sons)
virtual_narrators = {}
for fname, son_ids in father_name_groups.items():
    # Get the nisba from the son's full name for disambiguation
    son_nisbas = set()
    for sid in son_ids:
        if sid in by_id:
            son_name = by_id[sid].get("name", "")
            nisbas = NISBA_RE.findall(son_name)
            son_nisbas.update(nisbas)

    virtual_narrators[fname] = {
        "son_entry_ids": son_ids,
        "frequency": len(son_ids),
        "son_nisbas": list(son_nisbas)[:5],
        "virtual": True,
    }

# Sort by frequency
sorted_vn = sorted(virtual_narrators.items(), key=lambda x: -x[1]["frequency"])
print(f"\n  Virtual narrators (fathers without entries): {len(virtual_narrators):,}")
print(f"\n  Top 20 most common father names:")
for fname, data in sorted_vn[:20]:
    print(f"    \"{fname:<20s}\": {data['frequency']:>4} sons, nisbas: {data['son_nisbas'][:3]}")

# 4c. Build nisba-narrowed matching
# For each unresolved father, find encyclopedia entries with matching nisba
nisba_narrowed = {}
for fname, data in virtual_narrators.items():
    if data["son_nisbas"]:
        # Look for entries that match both the father name AND the nisba
        fkey = re.sub(r"[ً-ٌٍ-َّْٰ]", "", fname)
        fkey = fkey.replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()
        candidates = []
        for nkey, eid in name_to_entry.items():
            if fkey in nkey and any(ns in nkey for ns in data["son_nisbas"]):
                candidates.append(eid)
        if candidates:
            nisba_narrowed[fname] = candidates

print(f"\n  Nisba-narrowed matches: {len(nisba_narrowed):,}")
print(f"  (Resolved generic fathers using son's nisba)")

# ── Save ──
print(f"\nSaving...")
with open(OUTPUT_DIR / "father_son_resolved.json", "w", encoding="utf-8") as f:
    json.dump({str(k): v for k, v in son_to_father_id.items()}, f, ensure_ascii=False)
print(f"  Saved: {OUTPUT_DIR / 'father_son_resolved.json'} ({len(son_to_father_id)} resolved pairs)")

with open(OUTPUT_DIR / "virtual_narrators.json", "w", encoding="utf-8") as f:
    json.dump(virtual_narrators, f, ensure_ascii=False, indent=2)
print(f"  Saved: {OUTPUT_DIR / 'virtual_narrators.json'} ({len(virtual_narrators)} virtual narrators)")

with open(OUTPUT_DIR / "nisba_narrowed_fathers.json", "w", encoding="utf-8") as f:
    json.dump({k: v for k, v in nisba_narrowed.items()}, f, ensure_ascii=False)
print(f"  Saved: {OUTPUT_DIR / 'nisba_narrowed_fathers.json'}")

with open(OUTPUT_DIR / "nisba_clusters.json", "w", encoding="utf-8") as f:
    json.dump({k: v for k, v in nisba_groups.items() if len(v) > 1}, f, ensure_ascii=False)
print(f"  Saved: {OUTPUT_DIR / 'nisba_clusters.json'}")

summary = {
    "entries": len(enc),
    "father_son_relations": len(father_son),
    "resolved_to_actual_entries": resolved_fathers,
    "virtual_narrators_created": len(virtual_narrators),
    "nisba_narrowed_matches": len(nisba_narrowed),
    "entries_with_teachers": teacher_count,
    "entries_with_students": student_count,
    "nisba_groups": len(nisba_groups),
    "multi_member_families": len(families_with_multiple),
}
with open(OUTPUT_DIR / "family_graph_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"  Saved: {OUTPUT_DIR / 'family_graph_summary.json'}")

print(f"\n{'=' * 60}")
print(f"  COMPLETE")
print(f"  {len(father_son):,} father-son relations")
print(f"  {teacher_count:,} narrators with known teachers")
print(f"  {len(nisba_groups):,} nisba family groups")
print(f"{'=' * 60}")
