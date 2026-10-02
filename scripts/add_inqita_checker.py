"""
Inqita' detection using tabaqa data — fixed for truncated fields.
"""
import json, re, sys, ast
from collections import defaultdict
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

# ── Tabaqa extraction (handles truncated text) ──
TABAQA_PREFIXES = {
    "الاول": 1, "الاولى": 1,
    "الثان": 2, "الثاني": 2, "الثانية": 2,
    "الثال": 3, "الثالث": 3, "الثالثة": 3,
    "الرابع": 4, "الرابعة": 4,
    "الخامس": 5, "الخامسة": 5,
    "السادس": 6, "السادسة": 6,
    "السابع": 7, "السابعة": 7,
    "الثامن": 8, "الثامنة": 8,
    "التاسع": 9, "التاسعة": 9,
    "العاش": 10, "العاشر": 10, "العاشرة": 10,
    "الحاد": 11, "الحادي": 11, "الحادية": 11,
    "الثان": 12, # second for الثانية عشرة
}

def parse_tabaqa(text):
    if not text: return 0
    # Extract "من XXX" portion
    m = re.search(r"من\s+(ال[؀-ۿ]{3,20})", text)
    if not m: return 0
    tname = m.group(1)
    # Check for "عشرة" suffix (e.g., "الحادية عشرة" → 11)
    if "عشرة" in tname or "عشر" in tname:
        if "الحاد" in tname or "الحادي" in tname or "الحادية" in tname:
            return 11
        if "الثان" in tname:
            return 12
    # Simple prefix matching
    for prefix, num in sorted(TABAQA_PREFIXES.items(), key=lambda x: -len(x[0])):
        if prefix in tname:
            return num
    return 0

# Build tabaqa lookup
entry_tabaqa = {}
for e in enc:
    eid = e["entry_id"]
    t = parse_tabaqa(e.get("tabaqa", ""))
    if t > 0:
        entry_tabaqa[eid] = t

name_tabaqa = {}
for ak, eids in alias.items():
    eid = eids[0]
    if eid in entry_tabaqa:
        name_tabaqa[norm(ak)] = entry_tabaqa[eid]

COMPANION = {"انس","انس بن مالك","ابو هريرة","ابي هريرة","عائشة",
    "ابن عباس","ابن مسعود","ابن عمر","جابر","ابو سعيد","عمر","علي","ابو بكر","عثمان"}

print(f"Tabaqa entries: {len(entry_tabaqa):,} | Mapped to names: {len(name_tabaqa):,}")

def detect_inqita(names):
    tabaqas = []
    for name in names:
        nk = norm(name)
        t = name_tabaqa.get(nk, 0)
        for cn in COMPANION:
            if cn in nk or nk in cn: t = 1; break
        tabaqas.append(t)

    inq_count = 0
    for i in range(len(tabaqas) - 1):
        tt, ts = tabaqas[i], tabaqas[i + 1]
        if tt > 0 and ts > 0:
            # Teacher must be from EARLIER generation (lower number) than student
            if tt >= ts + 4:
                # Teacher is 4+ generations younger than student! Impossible.
                inq_count += 2
            elif tt >= ts + 2:
                # Teacher is 2-3 generations younger — likely impossible
                inq_count += 1

    if inq_count >= 2: return 0.40
    elif inq_count >= 1: return 0.70
    return 1.0

# ── Test ──
print("Loading Sanadset...")
sanad_col = []
for chunk in pd.read_csv(
    "Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv",
    usecols=["Sanad"], chunksize=100_000, dtype_backend="pyarrow",
):
    sanad_col.extend(chunk["Sanad"].tolist())
print(f"  {len(sanad_col):,} rows")

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
    sb = (sc[0] + sc[1]) / 2 if len(sc) >= 2 else sc[0]
    penalty = detect_inqita(names)
    sa = sb * penalty
    if penalty < 1.0: inq_hits += 1

    before.append(sb); after.append(sa)
    actuals.append(eval_df.iloc[aidx]["label"])

def g(s):
    return "Sahih" if s>=T[0] else "Hasan" if s>=T[1] else "Daif" if s>=T[2] else "Matruk"

def metrics(preds):
    c = sum(1 for i in range(len(actuals)) if actuals[i]==preds[i])
    sr = sum(1 for i in range(len(actuals)) if actuals[i]=="Sahih" and preds[i]=="Sahih")
    dr = sum(1 for i in range(len(actuals)) if actuals[i]=="Daif" and preds[i]=="Daif")
    return c/max(len(actuals),1), sr/max(sum(1 for a in actuals if a=="Sahih"),1), dr/max(sum(1 for a in actuals if a=="Daif"),1)

pb, pb_sr, pb_dr = metrics([g(s) for s in before])
pa, pa_sr, pa_dr = metrics([g(s) for s in after])

print(f"\nInqita' hits: {inq_hits}")
print(f"\nBefore: Acc={pb:.4f} SR={pb_sr:.4f} DR={pb_dr:.4f}")
print(f"After:  Acc={pa:.4f} SR={pa_sr:.4f} DR={pa_dr:.4f}")
print(f"DR change: {pb_dr:.4f} → {pa_dr:.4f} ({pa_dr-pb_dr:+.1%})")

# Show changes
changes = [(i, before[i], after[i]) for i in range(len(before)) if before[i]!=after[i]]
if changes:
    print(f"\nSample changes ({len(changes)} total):")
    for i, sb, sa in sorted(changes, key=lambda x: -abs(x[2]-x[1]))[:10]:
        print(f"  {actuals[i]:8s} {sb:.3f} → {sa:.3f} ({g(sa):>6s})")
