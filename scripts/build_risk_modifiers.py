"""
Extract edge risk modifiers from biographical text for §3.4.2.2.

Modifiers (from thesis Table 4):
  - tadlīs (concealing): ×0.80
  - ikhtilāṭ (mental deterioration): ×0.50
  - irsāl (omission): ×0.85
  - ghalat (frequent errors): ×0.75
  - kadhib (accused of lying): ×0.00 (chain-breaking)

Output: risk_modifiers.json → {entry_id: {"type": "tadlis", "modifier": 0.80}}
"""
import json, re, sys, hashlib
from collections import defaultdict
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

ENC_PATH = "output/shamela/narrator_encyclopedia.json"
OUT_DIR = Path("output/risk_modifiers")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Risk patterns → modifier value
RISK_PATTERNS = {
    "tadlis": {
        "patterns": ["تدليس", "مدلس", "يدلس"],
        "modifier": 0.80,
        "desc": "Concealed weakness (tadlīs)",
    },
    "ikhtilat": {
        "patterns": ["مختلط", "اختلط بآخره", "تغير بآخره", "اختلط قبل", "تغير قبل", "الاختلاط"],
        "modifier": 0.50,
        "desc": "Mental deterioration (ikhtilāṭ)",
    },
    "kadhib": {
        "patterns": ["كذاب", "يضع الحديث", "وضاع", "متهم بالكذب",
                    "يضع الأحاديث", "واضع الحديث"],
        "modifier": 0.00,
        "desc": "Accused of lying / fabricator",
    },
    "jahala": {
        "patterns": ["مجهول العين", "مجهول الحال"],
        "modifier": 0.30,
        "desc": "Unknown identity (jahālah)",
    },
}

def norm(t):
    if not t: return ""
    t = re.sub(r"[ً-ٌٍ-َّْٰ]", "", t)
    t = t.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    t = t.replace("ة","ه").replace("ى","ي")
    return t.strip()

print("=" * 60)
print("  EXTRACTING EDGE RISK MODIFIERS")
print("=" * 60)

enc = json.load(open(ENC_PATH, encoding="utf-8"))
print(f"Encyclopedia: {len(enc):,} entries")

# Extract modifiers per entry
entry_modifiers = {}  # entry_id → [(type, modifier)]
stats = defaultdict(int)

for e in enc:
    eid = e["entry_id"]
    text = norm(e.get("full_text", "") + " " + e.get("name", ""))
    text = text.replace("لت", "").replace("لى", "").replace("لا ", "")  # Avoid false matches
    if not text:
        continue

    eid_mods = []
    for risk_type, info in RISK_PATTERNS.items():
        for pat in info["patterns"]:
            if pat in text:
                eid_mods.append({"type": risk_type, "modifier": info["modifier"], "desc": info["desc"]})
                stats[risk_type] += 1
                break  # Only count once per type

    if eid_mods:
        entry_modifiers[str(eid)] = eid_mods

print(f"\nEntries with risk modifiers: {len(entry_modifiers):,}")
print(f"\nBy type:")
for risk_type, count in sorted(stats.items(), key=lambda x: -x[1]):
    print(f"  {risk_type:12s}: {count:>4} ({RISK_PATTERNS[risk_type]['modifier']:.2f})")

# Multi-modifier: worst one wins (most severe applies)
multi = sum(1 for mods in entry_modifiers.values() if len(mods) > 1)
print(f"\nEntries with >1 risk factor: {multi}")
if multi > 0:
    print("  Example:")
    for eid, mods in entry_modifiers.items():
        if len(mods) > 1:
            types = [m["type"] for m in mods]
            worst = min(mods, key=lambda m: m["modifier"])
            print(f"    Entry {eid}: {types} → worst: {worst['type']} ({worst['modifier']})")
            break

# ── Save ──
with open(OUT_DIR / "risk_modifiers.json", "w", encoding="utf-8") as f:
    json.dump(entry_modifiers, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {OUT_DIR / 'risk_modifiers.json'} ({len(entry_modifiers):,} entries)")

summary = {
    "total_entries": len(enc),
    "entries_with_modifiers": len(entry_modifiers),
    "by_type": {k: v for k, v in stats.items()},
    "modifier_values": {k: v["modifier"] for k, v in RISK_PATTERNS.items()},
}
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {OUT_DIR / 'summary.json'}")
