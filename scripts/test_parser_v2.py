"""Test fixed narrator parser on a real volume."""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shamela.narrator_parser import segment_volume

txt = Path(r"output/tahdhib_al_kamal/raw_volume_02.txt").read_text(encoding="utf-8")
print(f"Volume 2: {len(txt):,} chars")

entries, _ = segment_volume(txt, volume=2)
print(f"Entries found: {len(entries)}")
h = sum(1 for e in entries if e.confidence == "high")
m = sum(1 for e in entries if e.confidence == "medium")
l = sum(1 for e in entries if e.confidence == "low")
print(f"Confidence: {h} high, {m} medium, {l} low")
has_t = sum(1 for e in entries if e.teachers)
has_s = sum(1 for e in entries if e.students)
has_g = sum(1 for e in entries if e.grades)
has_d = sum(1 for e in entries if e.death_info)
print(f"Has teachers: {has_t}, students: {has_s}, grades: {has_g}, death: {has_d}")

for e in entries[:5]:
    print(f"\n[{e.entry_id}] {e.confidence}")
    print(f"  Name:     {e.name[:120]}")
    print(f"  Teachers: {list(e.teachers)[:3]}")
    print(f"  Students: {list(e.students)[:3]}")
    print(f"  Grades:   {list(e.grades)[:3]}")
    print(f"  Death:    {e.death_info[:100]}")
