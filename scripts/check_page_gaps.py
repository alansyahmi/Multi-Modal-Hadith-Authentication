"""Check which matched pages we have vs need."""
import json, re, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

with open(r"c:\Projects\FYP Project\output\shamela\tahdhib_al_kamal\matched_targets.json", encoding="utf-8") as f:
    matched = json.load(f)

# Pages we need
needed_pages = set(matched.values())
print(f"Total needed pages: {len(needed_pages)} (range {min(needed_pages)}-{max(needed_pages)})")

# Pages we have (HTML)
html_dir = Path(r"c:\Projects\FYP Project\output\shamela\tahdhib_al_kamal\pages")
html_pages = set()
for f in html_dir.glob("*.html"):
    m = re.search(r"page_(\d+)", f.name)
    if m:
        html_pages.add(int(m.group(1)))

# Pages we have (target text)
target_dir = Path(r"c:\Projects\FYP Project\output\shamela\tahdhib_al_kamal\target_pages")
target_pages = set()
for f in target_dir.glob("*.txt"):
    m = re.search(r"page_(\d+)", f.name)
    if m:
        target_pages.add(int(m.group(1)))

print(f"HTML pages: {len(html_pages)} (range {min(html_pages)}-{max(html_pages)})")
print(f"Target pages: {len(target_pages)} (range {min(target_pages)}-{max(target_pages)})")

# Overlap analysis
html_have = needed_pages & html_pages
target_have = needed_pages & target_pages
either_have = needed_pages & (html_pages | target_pages)
missing = needed_pages - (html_pages | target_pages)

print(f"\n=== Coverage ===")
print(f"Need:       {len(needed_pages)} pages")
print(f"HTML cover: {len(html_have)} ({len(html_have)/len(needed_pages)*100:.1f}%)")
print(f"Target cov: {len(target_have)} ({len(target_have)/len(needed_pages)*100:.1f}%)")
print(f"Combined:   {len(either_have)} ({len(either_have)/len(needed_pages)*100:.1f}%)")
print(f"MISSING:    {len(missing)} ({len(missing)/len(needed_pages)*100:.1f}%)")

# Show missing page range
if missing:
    ms = sorted(missing)
    print(f"\nMissing pages: {len(ms)} (range {min(ms)}-{max(ms)})")
    print(f"First 10: {ms[:10]}")
    print(f"Last 10:  {ms[-10:]}")

# Narrators covered vs not
narrators_have = 0
narrators_missing = 0
for name, page in matched.items():
    if page in (html_pages | target_pages):
        narrators_have += 1
    else:
        narrators_missing += 1

print(f"\n=== Narrator Coverage ===")
print(f"Pages available: {narrators_have} narrators")
print(f"Pages missing:   {narrators_missing} narrators")
