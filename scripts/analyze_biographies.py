"""Analyze structured_biographies.json quality."""
import json, sys
sys.stdout.reconfigure(encoding="utf-8")

with open(r"c:\Projects\FYP Project\output\structured_biographies.json", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total entries: {len(data)}")

empty_teachers = sum(1 for d in data if not d["teachers"])
empty_students = sum(1 for d in data if not d["students"])
empty_grades = sum(1 for d in data if not d["grades"])
empty_death = sum(1 for d in data if not d["death_info"])
page_unknown = sum(1 for d in data if d["target_names"] and "Page" in str(d["target_names"][0]))
non_empty_names = sum(1 for d in data if any(n for n in d["target_names"] if "Page" not in n and "Unknown" not in n))
text_has_grade = sum(1 for d in data if any(w in d.get("biography_text_preview","") for w in ["ثقة", "صدوق", "ضعيف", "متروك", "كذاب", "مقبول"]))
text_has_rah = sum(1 for d in data if "رَوَى عَن" in d.get("biography_text_preview","") or "رَوَى عَنه" in d.get("biography_text_preview",""))
text_has_mawt = sum(1 for d in data if any(w in d.get("biography_text_preview","") for w in ["مات", "توفي", "سنة"]))

print(f"Empty teachers:     {empty_teachers}/{len(data)}")
print(f"Empty students:     {empty_students}/{len(data)}")
print(f"Empty grades:       {empty_grades}/{len(data)}")
print(f"Empty death_info:   {empty_death}/{len(data)}")
print(f"Page/Unknown names: {page_unknown}/{len(data)}")
print(f"Has real names:     {non_empty_names}/{len(data)}")
print(f"Text has grade kw:  {text_has_grade}/{len(data)}")
print(f"Text has chains:    {text_has_rah}/{len(data)}")
print(f"Text has death:     {text_has_mawt}/{len(data)}")

# Top 10 pages represented
from collections import Counter
pages = Counter(d["page_number"] for d in data)
print(f"\nBook pages covered: {len(pages)} unique pages (range {min(pages)}-{max(pages)})")

# Sample with grades
print("\n=== 3 samples with grade keywords ===")
count = 0
for d in data:
    txt = d.get("biography_text_preview", "")
    has_grade = any(w in txt for w in ["ثقة", "صدوق", "ضعيف", "متروك"])
    if has_grade and count < 3:
        print(f"\n[p{d['page_number']}] {d['target_names']}")
        # Find grade context in text
        for kw in ["ثقة", "صدوق", "ضعيف"]:
            idx = txt.find(kw)
            if idx >= 0:
                ctx = txt[max(0,idx-40):idx+len(kw)+40]
                print(f"  ...{ctx}...")
        count += 1
