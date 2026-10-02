"""Debug why common narrators aren't being matched."""
import json, sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shamela.targeted_parser import (
    load_page_text, find_narrator_entry, ENTRY_START_RE, _normalize_arabic,
)
from rapidfuzz.fuzz import partial_ratio

BASE = Path(r"c:\Projects\FYP Project\output\shamela\tahdhib_al_kamal")
TARGET_DIR = BASE / "target_pages"

with open(BASE / "matched_targets.json", encoding="utf-8") as f:
    matched = json.load(f)

# Test specific common narrators
test_names = [
    "أَبِي هُرَيْرَةَ",
    "الزُّهْرِيِّ",
    "ابْنِ عَبَّاسٍ",
    "الْأَعْمَشِ",
    "سُفْيَانُ",
    "وَكِيعٌ",
    "سُفْيَانُ بْنُ عُيَيْنَةَ",
    "مَعْمَرٌ",
]

for name in test_names:
    page_num = matched.get(name)
    if not page_num:
        print(f"\n{name}: NOT IN TOC")
        continue

    # Find page file
    page_file = None
    for ext in [".txt", ".html"]:
        p = TARGET_DIR / f"page_{page_num:05d}{ext}"
        if p.exists():
            page_file = p
            break

    if not page_file:
        print(f"\n{name}: p{page_num} — FILE NOT FOUND")
        continue

    txt = load_page_text(page_file)
    name_norm = _normalize_arabic(name)

    # Check: is name anywhere in page?
    page_norm = _normalize_arabic(txt)
    in_page = name_norm in page_norm

    # Check entry headers on page
    entries = list(ENTRY_START_RE.finditer(txt))
    entry_names = [m.group(1).strip() for m in entries]

    print(f"\n{name}: p{page_num} ({len(txt)} chars) {'FOUND_IN_PAGE' if in_page else 'NOT_IN_PAGE'}")

    if entries:
        # Show best matching entry
        best_score = 0
        best_name = ""
        for ename in entry_names:
            score = partial_ratio(name_norm, _normalize_arabic(ename))
            if score > best_score:
                best_score = score
                best_name = ename
        print(f"  Entries on page: {len(entries)}")
        print(f"  Best match: {best_score:.0f}% -> {best_name[:100]}")
        print(f"  All names: {[e[:60] for e in entry_names[:5]]}")
    else:
        print(f"  NO ENTRIES FOUND ON PAGE")
        print(f"  Page text: {txt[:300]}")
