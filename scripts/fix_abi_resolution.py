"""
Resolve "أبي" (أبيه = his father) in chains using the family graph.
26,071 chains have "أبي" as the weakest link — this fixes them.
"""
import json, re, ast, sys, time
from collections import defaultdict
import pandas as pd
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

GRADE_MAP = {"صحابي":1.0,"ثقة":0.85,"صدوق":0.70,"مقبول":0.40,"ضعيف":0.15,"متروك":0.05,"كذاب":0.0}
def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

# ── Build scores ──
print("Building narrator scores...")
enc = json.load(open("output/shamela/narrator_encyclopedia.json", encoding="utf-8"))
alias = json.load(open("output/shamela/alias_index.json", encoding="utf-8"))
with open("output/stage2/node_names.json", encoding="utf-8") as f: id_to_name = json.load(f)
classifier = json.load(open("output/track_a_classifier/narrator_scores_classifier.json", encoding="utf-8"))

entry_best = {}
for e in enc:
    for g in e.get("grades", []):
        for t in sorted(GRADE_MAP, key=len, reverse=True):
            if t in g: entry_best[e["entry_id"]] = max(entry_best.get(e["entry_id"],0), GRADE_MAP[t]); break

name_score = {}
for nid_str, name in id_to_name.items():
    nid, nk = int(nid_str), norm(name)
    score = 0.632
    if nk in alias:
        eid = alias[nk][0]
        if eid in entry_best: score = entry_best[eid]
    if score == 0.632:
        cv = classifier.get(nid_str, classifier.get(str(nid)))
        if cv and float(cv) != 0.632: score = float(cv)
    name_score[nk] = score

# ── Load family graph ──
print("Loading family graph...")
father_son = json.load(open("output/family_graph/father_son_resolved.json", encoding="utf-8"))
son_to_father_id = {int(k): v for k, v in father_son.items()}

virtual = json.load(open("output/family_graph/virtual_narrators.json", encoding="utf-8"))

# Build entry_id → name lookup
entry_name = {e["entry_id"]: e.get("name", "") for e in enc}

# Build entry_id → score lookup
entry_score = {}
for e in enc:
    eid = e["entry_id"]
    grade_scores = []
    for g in e.get("grades", []):
        for t in sorted(GRADE_MAP, key=len, reverse=True):
            if t in g: grade_scores.append(GRADE_MAP[t]); break
    entry_score[eid] = max(grade_scores) if grade_scores else 0.30

# Build name → entry_id lookup (normalized)
name_to_eid = {}
for e in enc:
    nk = norm(e.get("name", ""))
    if nk: name_to_eid[nk] = e["entry_id"]

# Virtual father scores: average of known sons' scores
virtual_scores = {}
for fname, data in virtual.items():
    son_scores = [entry_score.get(sid, 0.30) for sid in data["son_entry_ids"] if sid in entry_score]
    if son_scores:
        virtual_scores[norm(fname)] = float(np.mean(son_scores))
    else:
        virtual_scores[norm(fname)] = 0.40

print(f"  Father→son pairs: {len(son_to_father_id):,}")
print(f"  Virtual fathers with scores: {len(virtual_scores):,}")

# ── Process chains ──
print("\nProcessing 650K chains (resolving 'أبي' references)...")

t0 = time.time()
ABI_RE = re.compile(r"أبي|أبيه|أبوه|والده")

total = 0
abi_count = 0
resolved = 0
unresolved = 0
abi_improved = 0

for chunk_idx, chunk in enumerate(pd.read_csv(
    "Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv",
    usecols=["Sanad"], chunksize=100_000, dtype_backend="pyarrow",
)):
    for _, row in chunk.iterrows():
        s = row["Sanad"]
        if pd.isna(s): continue
        try: ch = json.loads(s.replace("'", '"'))
        except:
            try: ch = ast.literal_eval(s)
            except: continue
        names = [n.strip() for n in ch if len(n.strip()) > 3]
        if len(names) < 2: continue

        # Score with resolution
        prev_narrator = ""
        chain_scores = []
        had_abi = False

        for name in names:
            nk = norm(name)
            score = name_score.get(nk, 0.632)

            # Check if this is "أبي" / "أبيه"
            if ABI_RE.search(name):
                had_abi = True
                abi_count += 1
                # Try to resolve via previous narrator
                if prev_narrator:
                    pnk = norm(prev_narrator)
                    resolved_score = None

                    # Method 1: Look up prev narrator's entry → their father
                    if pnk in name_to_eid:
                        prev_eid = name_to_eid[pnk]
                        if prev_eid in son_to_father_id:
                            father_eid = son_to_father_id[prev_eid]
                            if father_eid in entry_score:
                                resolved_score = entry_score[father_eid]

                    # Method 2: Virtual father score (from sons' average)
                    if resolved_score is None:
                        for vname, vscore in virtual_scores.items():
                            if len(vname) >= 4 and (vname in nk or nk in vname):
                                resolved_score = vscore
                                break

                    if resolved_score is not None:
                        score = resolved_score
                        resolved += 1
                    else:
                        unresolved += 1
                else:
                    unresolved += 1

            chain_scores.append(score)
            prev_narrator = name

        sc = sorted(chain_scores)
        cs = (sc[0] + sc[1]) / 2 if len(sc) >= 2 else sc[0]
        total += 1

    print(f"  Chunk {chunk_idx+1}: {total:,} chains, {abi_count:,} 'أبي' refs, {resolved:,} resolved")

ttime = time.time() - t0
print(f"\nResults:")
print(f"  Total chains: {total:,}")
print(f"  'أبي' references: {abi_count:,}")
print(f"  Resolved: {resolved:,} ({resolved/max(abi_count,1)*100:.1f}%)")
print(f"  Unresolved: {unresolved:,}")
