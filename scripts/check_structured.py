"""Check structured.json quality from Tahdhib al-Kamal parse."""
import json, sys
sys.stdout.reconfigure(encoding="utf-8")

with open(r"c:\Projects\FYP Project\output\tahdhib_al_kamal\structured.json", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total entries: {len(data)}")
h = sum(1 for e in data if e.get("confidence") == "high")
m = sum(1 for e in data if e.get("confidence") == "medium")
l = sum(1 for e in data if e.get("confidence") == "low")
print(f"Confidence: {h} high, {m} medium, {l} low")
has_teachers = sum(1 for e in data if e.get("teachers"))
has_students = sum(1 for e in data if e.get("students"))
has_grades = sum(1 for e in data if e.get("grades"))
has_death = sum(1 for e in data if e.get("death_info"))
print(f"Has teachers: {has_teachers}")
print(f"Has students: {has_students}")
print(f"Has grades:   {has_grades}")
print(f"Has death:    {has_death}")

# Sample entries
for e in data[:5]:
    print(f"\n--- Entry {e['entry_id']} [conf={e['confidence']}] vol={e['volume']} ---")
    print(f"  Name:     {e['name'][:120]}")
    print(f"  Teachers: {e.get('teachers',[])[:3]}")
    print(f"  Students: {e.get('students',[])[:3]}")
    print(f"  Grades:   {e.get('grades',[])[:3]}")
    print(f"  Death:    {e.get('death_info','')[:100]}")
    print(f"  Text:     {e.get('full_text','')[:200]}")

# Check if volume 1 entries are introduction or actual bios
print(f"\n=== Volume distribution ===")
from collections import Counter
vc = Counter(e["volume"] for e in data)
for v, count in sorted(vc.items()):
    print(f"  Volume {v}: {count} entries")
