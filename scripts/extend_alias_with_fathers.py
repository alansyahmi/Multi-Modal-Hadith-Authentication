"""
Fast alias extension: build word→entry index, then match multi-word fragments.
"""
import json, re, sys
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")

print("=" * 60)
print("  EXTEND ALIAS (FAST)")
print("=" * 60)

alias = json.load(open("output/shamela/alias_index.json", encoding="utf-8"))
enc = json.load(open("output/shamela/narrator_encyclopedia.json", encoding="utf-8"))
by_id = {e["entry_id"]: e for e in enc}

def norm(t):
    if not t: return ""
    t = re.sub(r"[ً-ٌٍ-َّْٰ]", "", t)
    t = t.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    t = t.replace("ة","ه").replace("ى","ي")
    return t.strip()

print(f"Alias: {len(alias):,} | Encyclopedia: {len(enc):,}")

# Step 1: Build word→entry index
print("\n[1] Building word→entry index...")
word_index = defaultdict(set)
entry_words = {}  # eid → set of words
for e in enc:
    eid = e["entry_id"]
    words = set(norm(e.get("name", "")).split())
    entry_words[eid] = words
    for w in words:
        if len(w) >= 3:
            word_index[w].add(eid)
print(f"  Indexed {len(word_index):,} words from {len(enc):,} entries")

# Step 2: Extract multi-word father fragments
print("\n[2] Extracting multi-word father fragments...")
fragments = set()
FATHER_RE = re.compile(r"(?:بن|ابن)\s+([؀-ۿ\s]{3,60}?)(?:\s|$|،|\(|\.)")

for e in enc:
    name = e.get("name", "")
    for m in FATHER_RE.finditer(name):
        frag = norm(m.group(1).strip())
        if 5 <= len(frag) <= 100:
            fragments.add(frag)
    parts = re.split(r"\s+(?:بن|ابن)\s+", name)
    if len(parts) >= 3:
        for i in range(1, len(parts)-1):
            for j in range(i+1, min(i+3, len(parts))):
                chain = " ".join(parts[i:j+1])
                nchain = norm(chain)
                if 6 <= len(nchain) <= 120:
                    fragments.add(nchain)

print(f"  Fragments: {len(fragments):,}")

# Step 3: Fast matching using word index
print("\n[3] Fast matching fragments → entries...")
new_aliases = defaultdict(list)

for frag in sorted(fragments, key=len, reverse=True):
    fwords = [w for w in frag.split() if len(w) >= 3]
    if len(fwords) < 2:
        continue

    # Find entries containing ALL fragment words
    candidates = word_index.get(fwords[0], set())
    for w in fwords[1:]:
        candidates &= word_index.get(w, set())

    # Verify word-set subset
    for eid in sorted(candidates)[:50]:  # Check top 50 matches
        if eid in entry_words and set(fwords).issubset(entry_words[eid]):
            new_aliases[frag].append(eid)
            if len(new_aliases[frag]) >= 2:
                break

print(f"  Matched: {len(new_aliases):,}")

# Show key results
print("\nSample matches:")
for frag in sorted(new_aliases, key=len, reverse=True)[:30]:
    eids = new_aliases[frag][:2]
    names = [by_id[eid]["name"][:50] for eid in eids if eid in by_id]
    print(f"  \"{frag:<35s}\" → {names}")

# Add to alias
print("\n[4] Adding to alias index...")
added = 0
for frag, eids in new_aliases.items():
    if frag not in alias and len(frag) >= 8:
        alias[frag] = eids[:2]
        added += 1

print(f"  Added: {added:,}")
print(f"  Total: {len(alias):,}")

# Check specific
for check in ["يزيد بن إبراهيم", "يزيد بن إبراهيم بن الورتنيس"]:
    cnorm = norm(check)
    found = [(k, v) for k, v in alias.items() if cnorm in norm(k)]
    print(f"\n  \"{check}\": {len(found)} matches in alias")
    for k, v in found[:3]:
        names = [by_id[eid]["name"][:50] for eid in v if eid in by_id]
        print(f"    \"{k}\" → {names}")

# Save
with open("output/shamela/alias_index.json", "w", encoding="utf-8") as f:
    json.dump(alias, f, ensure_ascii=False)
print(f"\nSaved: output/shamela/alias_index.json")
