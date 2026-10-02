"""
Wire geospatial verification into chain scoring to complete PO2.
Uses nisba-based locations + haversine distance to flag implausible meetings.
"""
import json, re, ast, sys, math
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
sys.stdout.reconfigure(encoding="utf-8")

GRADE_MAP = {"صحابي":1.0,"ثقة":0.85,"صدوق":0.70,"مقبول":0.40,"ضعيف":0.15,"متروك":0.05,"كذاب":0.0}
def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371; dlat = math.radians(lat2-lat1); dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R*2*math.atan2(math.sqrt(a), math.sqrt(1-a))

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

# ── Build geo lookup ──
print("Building geospatial lookup...")
entry_locs = json.load(open("output/geo/entry_locations.json", encoding="utf-8"))
nisba_city = json.load(open("output/geo/nisba_to_city.json", encoding="utf-8"))
city_coords = json.load(open("output/geo/city_coords.json", encoding="utf-8"))
NISBA_RE = re.compile(r"(ال[؀-ۿ]{2,20}[ي])")

# Map names to locations through alias + direct nisba extraction
narrator_locs = {}
for ak, eids in alias.items():
    eid_str = str(eids[0])
    if eid_str in entry_locs:
        narrator_locs[norm(ak)] = entry_locs[eid_str]

# Direct nisba extraction for chain narrators
for nk in list(name_score.keys()):
    if nk not in narrator_locs:
        for m in NISBA_RE.finditer(nk):
            nisba = m.group(1)
            if nisba in nisba_city:
                city = nisba_city[nisba]
                narrator_locs[nk] = [(city, city_coords[city])]
                break

print(f"  Narrators with locations: {len(narrator_locs):,}")

GEO_THRESHOLD = 3000  # km
GEO_PENALTY = 0.85

def geo_penalty(names):
    """Check geographic feasibility of consecutive narrator meetings."""
    penalty = 1.0
    geo_pairs_checked = 0
    geo_violations = 0
    prev_loc = None
    for name in names:
        nk = norm(name)
        if nk in narrator_locs:
            this_loc = narrator_locs[nk][0]
            if prev_loc:
                geo_pairs_checked += 1
                dist = haversine(prev_loc[1][0], prev_loc[1][1], this_loc[1][0], this_loc[1][1])
                if dist > GEO_THRESHOLD:
                    penalty *= GEO_PENALTY
                    geo_violations += 1
            prev_loc = this_loc
        else:
            prev_loc = None
    return penalty, geo_violations

# ── Evaluate on Kutub as-Sitta ──
print("\nEvaluating with geospatial verification...")
sanad_col = []
for chunk in pd.read_csv("Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv",
                         usecols=["Sanad"], chunksize=100_000, dtype_backend="pyarrow"):
    sanad_col.extend(chunk["Sanad"].tolist())
align = json.load(open("output/stage1/alignment_results.json", encoding="utf-8"))
eval_df = pd.read_csv("output/stage1/kutub_sitta_evaluation.csv")
gl = {"Ṣaḥīḥ":"Sahih","Ḥasan":"Hasan","Ḍaʿīf":"Daif","Mawḍūʿ":"Matruk"}
eval_df["label"] = eval_df["grade_normalised"].map(gl).fillna("Daif")
T = (0.45, 0.425, 0.15)

def g(s):
    return "Sahih" if s>=T[0] else "Hasan" if s>=T[1] else "Daif" if s>=T[2] else "Matruk"

preds_no_geo, preds_with_geo, actuals = [], [], []
geo_hits = 0
pair_checks = 0

for aidx, entry in enumerate(align):
    if not entry.get("is_match"): continue
    ci = int(entry["corpus_idx"])
    if ci >= len(sanad_col): continue
    s = sanad_col[ci]
    if pd.isna(s): continue
    try: ch = json.loads(s.replace("'", '"'))
    except:
        try: ch = ast.literal_eval(s)
        except: continue
    names = [n.strip() for n in ch if len(n.strip()) > 3]
    if len(names) < 2: continue

    sc = sorted([name_score.get(norm(n), 0.632) for n in names])
    score_base = (sc[0] + sc[1]) / 2 if len(sc) >= 2 else sc[0]
    gp, gv = geo_penalty(names)
    score_geo = score_base * gp
    if gv > 0:
        geo_hits += 1
        pair_checks += gv

    preds_no_geo.append(g(score_base))
    preds_with_geo.append(g(score_geo))
    actuals.append(eval_df.iloc[aidx]["label"])

print(f"Geo violations found: {geo_hits} chains, {pair_checks} pairs")
acc_no = accuracy_score(actuals, preds_no_geo)
acc_geo = accuracy_score(actuals, preds_with_geo)
print(f"\nAccuracy without geo: {acc_no:.4f}")
print(f"Accuracy with geo:    {acc_geo:.4f}")
print(f"Delta:                {acc_geo-acc_no:+.4f}")

# ── Also test on full 650K ──
print(f"\nTesting geo on full 650K...")
total = 0; violations = 0
for chunk in pd.read_csv("Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv",
                         usecols=["Sanad"], chunksize=100_000, dtype_backend="pyarrow"):
    for _, row in chunk.iterrows():
        s = row["Sanad"]
        if pd.isna(s): continue
        try: ch = json.loads(s.replace("'", '"'))
        except:
            try: ch = ast.literal_eval(s)
            except: continue
        names = [n.strip() for n in ch if len(n.strip()) > 3]
        if len(names) < 2: continue
        _, gv = geo_penalty(names)
        if gv > 0: violations += 1
        total += 1
        if total % 100000 == 0: print(f"  {total:,} chains, {violations:,} violations")
print(f"Total: {total:,} chains, {violations:,} geographic violations ({violations/total*100:.1f}%)")

print(f"\nPO2 COMPLETE: Geospatial verification integrated.")
print(f"Geographic data: {len(narrator_locs):,} narrators mapped across {len(city_coords)} cities.")
