"""
Fix alias index ordering: when a short name maps to multiple entries,
sort by reliability score so the most trustworthy narrator comes FIRST.

The bug: alias["هشام"][0] pointed to a ضعيف narrator (entry 794)
instead of the ثقة هشام بن عروة (entry 6592).
"""
import json, sys, re
sys.stdout.reconfigure(encoding="utf-8")

alias = json.load(open("output/shamela/alias_index.json", encoding="utf-8"))
enc = json.load(open("output/shamela/narrator_encyclopedia.json", encoding="utf-8"))

GRADE_MAP = {
    "صحابي": 1.0, "ثقة ثبت": 0.95, "ثقة حافظ": 0.95, "ثقة متقن": 0.95,
    "إمام": 0.95, "حافظ": 0.90, "ثبت": 0.90, "متقن": 0.90,
    "ثقة": 0.85, "ثقات": 0.85, "مأمون": 0.85,
    "صدوق": 0.70, "لا بأس به": 0.65, "ليس به بأس": 0.65, "محله الصدق": 0.65,
    "صدوق يهم": 0.55, "صدوق سيء الحفظ": 0.50, "سيء الحفظ": 0.40,
    "مقبول": 0.40, "مستور": 0.25, "مجهول الحال": 0.20,
    "ضعيف": 0.15, "ضعيف الحديث": 0.15, "لين": 0.15, "لين الحديث": 0.15,
    "ليس بالقوي": 0.15, "فيه ضعف": 0.15, "منكر الحديث": 0.10, "مجهول": 0.10,
    "مجهول العين": 0.05, "متروك": 0.05, "متروك الحديث": 0.05, "ساقط": 0.05,
    "مختلط": 0.05, "متهم": 0.0, "كذاب": 0.0, "وضاع": 0.0, "دجال": 0.0,
}
DEFAULT = 0.30

def best_score(entry):
    grades = entry.get("grades", [])
    if not grades: return DEFAULT
    scores = []
    for g in grades:
        for term in sorted(GRADE_MAP, key=len, reverse=True):
            if term in g:
                scores.append(GRADE_MAP[term])
                break
    return max(scores) if scores else DEFAULT

entry_score = {e["entry_id"]: best_score(e) for e in enc}
print(f"Entries scored: {len(entry_score):,}")

fixed = 0
for ak, eids in alias.items():
    if len(eids) > 1:
        sorted_eids = sorted(eids, key=lambda eid: entry_score.get(eid, DEFAULT), reverse=True)
        if sorted_eids != eids:
            alias[ak] = sorted_eids
            fixed += 1

print(f"Alias entries reordered: {fixed:,}")

# Verify
checks = {"هشام": "هشام بن عروة"}
for short, expected in checks.items():
    if short in alias:
        first = alias[short][0]
        for e in enc:
            if e["entry_id"] == first:
                print(f"  {short} → entry {first}: {e['name'][:50]}")
                print(f"    Grades: {e.get('grades',[])}")
                break

with open("output/shamela/alias_index.json", "w", encoding="utf-8") as f:
    json.dump(alias, f, ensure_ascii=False)
print(f"Saved ({len(alias):,} entries)")
