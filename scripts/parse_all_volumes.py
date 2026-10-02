"""
Parse ALL raw volumes from both Tahdhib al-Kamal and Taqrib al-Tahdhib.
Produces unified narrator database.
"""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shamela.narrator_parser import segment_volume
from src.shamela.taqrib_parser import parse_taqrib_volume, TaqribEntry

# ── Tahdhib al-Kamal (35 volumes) ──
TAHDHIB_DIR = Path(r"output/tahdhib_al_kamal")
all_entries = []

vol_files = sorted(TAHDHIB_DIR.glob("raw_volume_*.txt"))
print(f"Tahdhib al-Kamal: {len(vol_files)} volumes found")

next_id = 1
for vf in vol_files:
    vol_num = int(vf.stem.replace("raw_volume_", ""))
    print(f"  Volume {vol_num:02d}: {vf.stat().st_size:>10,} bytes ... ", end="")
    txt = vf.read_text(encoding="utf-8")
    entries, next_id = segment_volume(txt, vol_num, start_entry_id=next_id)
    all_entries.extend(entries)
    h = sum(1 for e in entries if e.confidence == "high")
    m = sum(1 for e in entries if e.confidence == "medium")
    print(f"{len(entries):>5} entries ({h}h, {m}m)")

print(f"\nTahdhib al-Kamal total: {len(all_entries):,} entries")

# ── Taqrib al-Tahdhib (book_8609, 1 volume) ──
# Uses its own parser — Taqrib has a compressed single-line format
TAQRIB_FILE = Path(r"output/book_8609/raw_volume_01.txt")
if TAQRIB_FILE.exists():
    print(f"\nTaqrib al-Tahdhib: {TAQRIB_FILE.stat().st_size:,} bytes")
    txt = TAQRIB_FILE.read_text(encoding="utf-8")
    taqrib_entries = parse_taqrib_volume(txt)
    print(f"  {len(taqrib_entries)} parsed entries")
    # Convert TaqribEntry to NarratorEntry-like dicts
    for te in taqrib_entries:
        all_entries.append({
            "entry_id": next_id,
            "volume": 0,  # separate source
            "name": te.name,
            "full_text": te.full_text,
            "teachers": [],
            "students": [],
            "grades": [te.grade] if te.grade else [],
            "death_info": te.death_year_text,
            "confidence": te.confidence,
            "source": "taqrib",
            "tabaqa": te.tabaqa,
            "book_codes": te.book_codes,
        })
        next_id += 1
    print(f"  Added {len(taqrib_entries):,} entries")

# ── Export as unified narrator encyclopedia ──
# Tahdhib entries use NarratorEntry dataclass with ._asdict-like access
export_data = []
for e in all_entries:
    if isinstance(e, dict):
        export_data.append(e)
    else:
        export_data.append({
            "entry_id": e.entry_id,
            "volume": e.volume,
            "name": e.name,
            "full_text": e.full_text[:3000] if e.full_text else "",
            "teachers": list(e.teachers) if hasattr(e, "teachers") else [],
            "students": list(e.students) if hasattr(e, "students") else [],
            "grades": list(e.grades) if hasattr(e, "grades") else [],
            "death_info": e.death_info if hasattr(e, "death_info") else "",
            "confidence": e.confidence,
            "source": "tahdhib",
        })

out_dir = Path(r"output/shamela")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "narrator_encyclopedia.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(export_data, f, ensure_ascii=False, indent=2)

print(f"\n{'='*50}")
print(f"Total entries: {len(all_entries):,}")
tahdhib_count = sum(1 for e in all_entries if not isinstance(e, dict) or e.get("source") == "tahdhib")
taqrib_count = sum(1 for e in all_entries if isinstance(e, dict) and e.get("source") == "taqrib")
print(f"  Tahdhib al-Kamal: {tahdhib_count:,}")
print(f"  Taqrib al-Tahdhib: {taqrib_count:,}")
has_grades = sum(1 for e in export_data if e.get("grades"))
print(f"  With grades: {has_grades:,}")
print(f"Output: {out_path}")

