"""Debug why grade extraction fails."""
import sys, re
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shamela.narrator_parser import (
    segment_volume, GRADE_RE, STANDALONE_GRADE_RE, BARE_GRADE_RE,
    GRADE_TERMS, DEATH_RE
)

txt = Path(r"output/tahdhib_al_kamal/raw_volume_02.txt").read_text(encoding="utf-8")
entries, _ = segment_volume(txt, volume=2)

# Check first entry
e = entries[0]
print(f"Entry: {e.name}")
print(f"Text length: {len(e.full_text)}")
print(f"\n=== FULL TEXT ===")
print(e.full_text[:2000])

# Check which regex matches
print(f"\n=== GRADE_RE matches ===")
for m in GRADE_RE.finditer(e.full_text):
    print(f"  MATCH: {m.group(0)[:100]} → grade: {m.group(1)}")

print(f"\n=== STANDALONE_GRADE_RE matches ===")
for m in STANDALONE_GRADE_RE.finditer(e.full_text):
    print(f"  MATCH: {m.group(0)[:100]} → grade: {m.group(1)}")

print(f"\n=== BARE_GRADE_RE matches ===")
for m in BARE_GRADE_RE.finditer(e.full_text):
    print(f"  MATCH: {m.group(0)[:100]} → grade: {m.group(1)}")

print(f"\n=== DEATH_RE matches ===")
for m in DEATH_RE.finditer(e.full_text):
    print(f"  MATCH: {m.group(0)[:120]}")

# Check: does any grade term appear at all?
for term in ["ثقة", "متروك", "ليس به بأس", "مجهول"]:
    idx = e.full_text.find(term)
    if idx >= 0:
        ctx = e.full_text[max(0,idx-50):idx+len(term)+50]
        print(f"\n'{term}' at pos {idx}: ...{ctx}...")
    else:
        print(f"\n'{term}': NOT FOUND in entry text")
