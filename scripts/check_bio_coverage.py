"""Check how many narrators from eval set are covered in biographies JSON."""
import json, ast, sys, re
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
from collections import Counter

# Load all narrator names from the 1,200 eval hadiths
eval_df = pd.read_csv(
    r"c:\Projects\FYP Project\output\stage1\kutub_sitta_evaluation.csv"
)
sanadset = pd.read_csv(
    r"c:\Projects\FYP Project\output\stage1\sanadset_cleaned.csv"
)
with open(r"c:\Projects\FYP Project\output\stage1\alignment_results.json") as f:
    alignment = json.load(f)

# Get unique narrators from both eval and matched Sanadset records
all_narrators = set()
eval_narrators = set()  # From HF side
sanad_narrators = set()  # From Sanadset side

# Parse Sanadset chains for the matched hadiths
matched = [a for a in alignment if a.get("is_match")]
print(f"Matched hadiths: {len(matched)}")

for a in matched:
    cidx = a["corpus_idx"]
    if cidx is not None:
        try:
            row = sanadset.loc[cidx]
            chain = ast.literal_eval(row["Sanad"]) if isinstance(row["Sanad"], str) else []
            for name in chain:
                all_narrators.add(name.strip())
                sanad_narrators.add(name.strip())
        except:
            pass

# Get HF chain (from Kutub as-Sitta — the sanads in HF don't have extracted chains,
# but we don't need them since sanadset has them for matched hadiths)
print(f"\nUnique narrators in matched chains: {len(all_narrators)}")
print(f"  From Sanadset: {len(sanad_narrators)}")

# Load biography JSON
with open(r"c:\Projects\FYP Project\output\structured_biographies.json", encoding="utf-8") as f:
    bios = json.load(f)

# Build searchable text corpus from all biography entries
bio_full_text = " ".join(d.get("biography_text_preview", "") for d in bios)
print(f"\nBiography corpus: {len(bio_full_text):,} chars, {len(bios)} pages")

# Check coverage: how many unique narrators can we find?
found = set()
not_found = set()
for name in all_narrators:
    # Try exact match first
    if name in bio_full_text:
        found.add(name)
    else:
        # Try shortest meaningful part (last 2 tokens of name)
        parts = name.split()
        if len(parts) >= 2 and " ".join(parts[-2:]) in bio_full_text:
            found.add(name)
        else:
            not_found.add(name)

coverage = len(found) / len(all_narrators) * 100 if all_narrators else 0
print(f"\n=== Coverage ===")
print(f"Found:     {len(found)} / {len(all_narrators)} ({coverage:.1f}%)")
print(f"Not found: {len(not_found)}")

# Show sample of found and not found
print(f"\nSample FOUND (10):")
for n in list(found)[:10]:
    print(f"  {n}")
    # Show context
    idx = bio_full_text.find(n)
    if idx >= 0:
        ctx = bio_full_text[max(0,idx-30):idx+len(n)+30]
        print(f"    ...{ctx}...")

print(f"\nSample NOT FOUND (10):")
for n in list(not_found)[:10]:
    print(f"  {n}")

# Frequency distribution — most common narrators
name_freq = Counter()
for a in matched:
    cidx = a["corpus_idx"]
    if cidx is not None:
        try:
            row = sanadset.loc[cidx]
            chain = ast.literal_eval(row["Sanad"]) if isinstance(row["Sanad"], str) else []
            for name in chain:
                name_freq[name.strip()] += 1
        except:
            pass

print(f"\n=== Top 20 Most Frequent Narrators ===")
for name, freq in name_freq.most_common(20):
    in_bio = "FOUND" if name in found else "MISSING"
    print(f"  {freq:4d}x  [{in_bio:7s}]  {name[:80]}")
