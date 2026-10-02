"""Test Taqrib-specific parser."""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shamela.taqrib_parser import parse_taqrib_volume

txt = Path(r"output/book_8609/raw_volume_01.txt").read_text(encoding="utf-8")
entries = parse_taqrib_volume(txt)
print(f"Entries: {len(entries)}")
with_grade = sum(1 for e in entries if e.grade)
with_death = sum(1 for e in entries if e.death_year_text)
with_tabaqa = sum(1 for e in entries if e.tabaqa)
print(f"With grade: {with_grade} ({with_grade/len(entries)*100:.1f}%)")
print(f"With death: {with_death} ({with_death/len(entries)*100:.1f}%)")
print(f"With tabaqa: {with_tabaqa}")

# Grade distribution
from collections import Counter
gc = Counter(e.grade for e in entries if e.grade)
print(f"\nGrade distribution ({len(gc)} unique):")
for g, c in gc.most_common(20):
    print(f"  {g}: {c}")

# Sample
print(f"\n=== Sample entries ===")
for e in entries[0:8]:
    print(f"\n[{e.entry_id}] {e.confidence}")
    print(f"  Name:       {e.name[:120]}")
    print(f"  Grade:      {e.grade}")
    print(f"  Tabaqa:     {e.tabaqa}")
    print(f"  Death:      {e.death_year_text}")
    print(f"  Book codes: {e.book_codes}")
