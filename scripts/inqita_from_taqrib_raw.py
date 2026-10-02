"""
Re-extract tabaqa from raw Taqrib text (no truncation).
Then use for precise inqita' detection.
"""
import json, re, sys, ast
import pandas as pd
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

GRADE_MAP = {
    "صحابي":1.0,"ثقة ثبت":0.95,"ثقة حافظ":0.95,"إمام":0.95,
    "حافظ":0.90,"ثبت":0.90,"ثقة":0.85,"صدوق":0.70,"مقبول":0.40,
    "ضعيف":0.15,"متروك":0.05,"كذاب":0.0,
}
def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

# ── Parse tabaqa from RAW Taqrib text ──
print("Re-extracting tabaqa from raw Taqrib text...")
taqrib_text = open("output/book_8609/raw_volume_01.txt", encoding="utf-8").read()

# Taqrib entries start with "رقم- اسم"
ENTRY_RE = re.compile(r"^[\d٠-٩]{1,4}\s*[-–—]?\s*(.*)$", re.MULTILINE)
TABAQA_RE = re.compile(r"من\s+(ال[؀-ۿ]{4,30})")
GRADE_TERMS = ["ثقة ثبت","ثقة حافظ","ثقة متقن","ثقة","صدوق سيء الحفظ","صدوق يهم",
    "صدوق","مقبول","ضعيف","متروك","كذاب"]

TABAQA_MAP = {
    "الاولى":1,"الثانية":2,"الثالثة":3,"الرابعة":4,"الخامسة":5,
    "السادسة":6,"السابعة":7,"الثامنة":8,"التاسعة":9,
    "العاشرة":10,"الحادية عشرة":11,"الثانية عشرة":12,
}

taqrib_tabaqa = {}  # entry_name → tabaqa number

for match in ENTRY_RE.finditer(taqrib_text):
    line = match.group(1).strip()
    if not line: continue

    # Extract name (before the grade term)
    name = ""
    for gt in GRADE_TERMS:
        if gt in line:
            name = line[:line.index(gt)].strip()
            break
    if not name:
        name = line[:60]  # Fallback

    # Extract tabaqa
    m = TABAQA_RE.search(line)
    if m:
        tabaqa_text = m.group(1)
        # Match against known tabaqa names
        for tname, num in TABAQA_MAP.items():
            if tname in tabaqa_text or tabaqa_text in tname:
                taqrib_tabaqa[norm(line[:80])] = num
                break

print(f"  Taqrib entries with tabaqa: {len(taqrib_tabaqa):,}")

# Map through alias index
alias = json.load(open("output/shamela/alias_index.json", encoding="utf-8"))
name_tabaqa = {}
for ak, eids in alias.items():
    for taq_entry, tnum in taqrib_tabaqa.items():
        if norm(ak) in taq_entry or taq_entry[:30] in norm(ak):
            name_tabaqa[norm(ak[:50])] = tnum
            break

print(f"  Mapped to names: {len(name_tabaqa):,}")

# ── Build scores ──
enc = json.load(open("output/shamela/narrator_encyclopedia.json", encoding="utf-8"))
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

COMPANION = {"انس","انس بن مالك","ابو هريرة","ابي هريرة","عائشة",
    "ابن عباس","ابن مسعود","ابن عمر","جابر","ابو سعيد","عمر","علي","ابو بكر","عثمان"}

def detect_inqita(names):
    tabaqas = []
    for name in names:
        nk = norm(name[:50])
        t = 0
        # Exact match first
        if nk in name_tabaqa:
            t = name_tabaqa[nk]
        else:
            # Fuzzy: check if name_tabaqa key is in nk or nk in key
            for k, v in name_tabaqa.items():
                if len(k) >= 8 and (k in nk or nk[:40] in k):
                    t = v; break
        for cn in COMPANION:
            if cn in nk or nk in cn: t = 1; break
        tabaqas.append(t)

    # Count only CLEAR impossibilities: teacher 5+ generations younger than student
    # Or multiple 3+ generation gaps
    inq = 0
    for i in range(len(tabaqas) - 1):
        tt, ts = tabaqas[i], tabaqas[i + 1]
        if tt > 0 and ts > 0:
            if tt >= ts + 5:
                inq += 2
            elif tt >= ts + 3:
                inq += 1

    # Only apply penalty if we're VERY confident
    if inq >= 2: return 0.40
    elif inq >= 1: return 0.60
    return 1.0

# ── Quick test on eval set ──
print("Testing...")
sanad_col = []
for chunk in pd.read_csv("Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv",
                         usecols=["Sanad"], chunksize=100_000, dtype_backend="pyarrow"):
    sanad_col.extend(chunk["Sanad"].tolist())
align = json.load(open("output/stage1/alignment_results.json", encoding="utf-8"))
eval_df = pd.read_csv("output/stage1/kutub_sitta_evaluation.csv")
gl = {"Ṣaḥīḥ":"Sahih","Ḥasan":"Hasan","Ḍaʿīf":"Daif","Mawḍūʿ":"Matruk"}
eval_df["label"] = eval_df["grade_normalised"].map(gl).fillna("Daif")
T = (0.45, 0.425, 0.15)

before, after, actuals = [], [], []
inq_hits = 0
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
    sb = (sc[0]+sc[1])/2 if len(sc)>=2 else sc[0]
    p = detect_inqita(names)
    sa = sb * p
    if p < 1.0: inq_hits += 1
    before.append(sb); after.append(sa)
    actuals.append(eval_df.iloc[aidx]["label"])

def g(s):
    return "Sahih" if s>=T[0] else "Hasan" if s>=T[1] else "Daif" if s>=T[2] else "Matruk"
def m(p):
    c = sum(1 for i in range(len(actuals)) if actuals[i]==p[i])
    sr = sum(1 for i in range(len(actuals)) if actuals[i]=="Sahih" and p[i]=="Sahih")
    dr = sum(1 for i in range(len(actuals)) if actuals[i]=="Daif" and p[i]=="Daif")
    return c/max(len(actuals),1), sr/max(sum(1 for a in actuals if a=="Sahih"),1), dr/max(sum(1 for a in actuals if a=="Daif"),1)

pb = m([g(s) for s in before])
pa = m([g(s) for s in after])
print(f"\nInqita hits: {inq_hits}")
print(f"Before: Acc={pb[0]:.4f} SR={pb[1]:.4f} DR={pb[2]:.4f}")
print(f"After:  Acc={pa[0]:.4f} SR={pa[1]:.4f} DR={pa[2]:.4f}")
print(f"DR Δ: {pa[2]-pb[2]:+.1%} | SR Δ: {pa[1]-pb[1]:+.1%}")
