"""
Build comprehensive alias index from the narrator encyclopedia.

Extracts all name variants from each entry:
  - Ism (proper name): e.g. "أحمد", "محمد"
  - Kunya (teknonym): e.g. "أبو عبد الله", "أبو الحسن"
  - Nasab (patronym): e.g. "ابن حنبل", "ابن عيينة"
  - Nisba (affiliation): e.g. "البخاري", "المروزي", "الفسوي"
  - Laqab (nickname): e.g. "البخاري", "مسلم"
  - Full name variants with/without diacritics
  - Partial names (ism only, ism + nasab, etc.)

Output:
  output/shamela/alias_index.json    {variant: canonical_entry_id}
  output/shamela/name_variants.json  Detailed breakdown per entry
"""
import sys, json, re
from collections import defaultdict
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_DIR = Path("output/shamela")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def norm(t):
    """Remove diacritics and normalize alifs/ya/ta for matching."""
    if not t: return ""
    t = re.sub(r"[ً-ٌٍ-َّْٰ]", "", t)
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    t = t.replace("ة", "ه").replace("ى", "ي")
    return t.strip()

# Load encyclopedia
print("Loading narrator encyclopedia...")
with open(OUTPUT_DIR / "narrator_encyclopedia.json", encoding="utf-8") as f:
    enc = json.load(f)

print(f"Entries: {len(enc):,}")

# ── Extract name variants ──
# Pattern: kunya = أبو X (or أبو X Y)
KUNYA_RE = re.compile(r"(أبو\s[؀-ۿ]{3,20}(?:\s[؀-ۿ]{3,20})?)")

# Pattern: nisba = الـ...ي/ي at end of name
NISBA_RE = re.compile(r"(ال[؀-ۿ]{2,20}[ي])")

# Pattern: ibn/bin X
NASAB_RE = re.compile(r"(?:ابن|بن)\s([؀-ۿ]{3,30})")

print("\nExtracting aliases...")
alias_to_entry: dict[str, list[int]] = defaultdict(list)  # norm_variant → [entry_ids]
entry_variants: dict[int, dict] = {}  # entry_id → metadata

for entry in enc:
    eid = entry["entry_id"]
    name = entry.get("name", "")
    full_text = entry.get("full_text", "")
    source = entry.get("source", "tahdhib")

    # Skip malformed entries
    if not name or len(name) < 5:
        continue

    variants = set()

    # 1. Full name (normalized)
    n_name = norm(name)
    variants.add(n_name)

    # 2. Full name without parenthetical notes
    clean_name = re.sub(r"\([^)]*\)", "", name).strip()
    variants.add(norm(clean_name))

    # 3. First word (ism) — e.g. "أحمد" from "أحمد بن حنبل"
    parts = name.split()
    if parts:
        variants.add(norm(parts[0]))

    # 4. Kunya (أبو X) — extract from full text too
    for m in KUNYA_RE.finditer(name + " " + full_text[:500]):
        kunya = norm(m.group(1))
        if 5 < len(kunya) < 60:
            variants.add(kunya)

    # 5. Nisba (ال...ي)
    for m in NISBA_RE.finditer(name):
        nisba = norm(m.group(1))
        if 4 < len(nisba) < 40:
            variants.add(nisba)

    # 6. Last part (usually nisba or laqab)
    if parts:
        last = norm(parts[-1])
        if 3 < len(last) < 40:
            variants.add(last)

    # 7. "ابن X" or "بن X" from name
    for m in NASAB_RE.finditer(name):
        ibn = norm(m.group(1))
        if len(ibn) > 3:
            variants.add("ابن " + ibn)
            variants.add(ibn)

    # 8. Title-based: "الحافظ", "الإمام" etc.
    # These are less specific, skip for now

    # Store
    entry_variants[eid] = {
        "name": name,
        "source": source,
        "variants": sorted(variants),
        "canonical": n_name,
    }

    for v in variants:
        if len(v) >= 3:
            alias_to_entry[v].append(eid)

# ── Stats ──
print(f"\nAlias stats:")
total_variants = sum(len(ev["variants"]) for ev in entry_variants.values())
print(f"  Entries with aliases: {len(entry_variants):,}")
print(f"  Total alias forms:    {total_variants:,}")
print(f"  Unique alias strings: {len(alias_to_entry):,}")

# Ambiguous aliases (map to multiple entries)
ambiguous = {k: v for k, v in alias_to_entry.items() if len(v) > 1}
print(f"  Ambiguous aliases:    {len(ambiguous):,}")
if ambiguous:
    print(f"  Sample ambiguous:")
    for k in sorted(ambiguous, key=lambda x: len(alias_to_entry[x]), reverse=True)[:5]:
        ids = alias_to_entry[k]
        names = [entry_variants[i]["name"][:40] for i in ids]
        print(f"    \"{k[:40]}\" → {ids}: {names}")

# ── Save ──
# Convert alias_to_entry sets to lists for JSON
alias_serializable = {k: v for k, v in alias_to_entry.items()}
with open(OUTPUT_DIR / "alias_index.json", "w", encoding="utf-8") as f:
    json.dump(alias_serializable, f, ensure_ascii=False)
print(f"\nSaved: {OUTPUT_DIR / 'alias_index.json'}")

with open(OUTPUT_DIR / "name_variants.json", "w", encoding="utf-8") as f:
    json.dump(entry_variants, f, ensure_ascii=False)
print(f"Saved: {OUTPUT_DIR / 'name_variants.json'}")

print(f"\nDone — {len(alias_to_entry):,} unique alias forms for {len(entry_variants):,} entries")
