"""
FINAL FYP1 pipeline: all fixes combined.
  - Hybrid GRADE_SCORE_MAP + classifier scores
  - Companion rule
  - Avg-bottom-2 chain scoring (relaxed weakest-link)
  - Calibrated thresholds
  - Risk modifiers
  - 650K chain evaluation
"""
import json, re, ast, sys, time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
sys.stdout.reconfigure(encoding="utf-8")

GRADE_MAP = {
    "صحابي":1.0,"ثقة ثبت":0.95,"ثقة حافظ":0.95,"ثقة متقن":0.95,"إمام":0.95,
    "حافظ":0.90,"ثبت":0.90,"متقن":0.90,"ثقة":0.85,"ثقات":0.85,"مأمون":0.85,
    "صدوق":0.70,"لا بأس به":0.65,"ليس به بأس":0.65,"محله الصدق":0.65,
    "صدوق يهم":0.55,"صدوق سيء الحفظ":0.50,"سيء الحفظ":0.40,"مقبول":0.40,
    "مستور":0.25,"مجهول الحال":0.20,"ضعيف":0.15,"ضعيف الحديث":0.15,
    "لين":0.15,"لين الحديث":0.15,"ليس بالقوي":0.15,"فيه ضعف":0.15,
    "منكر الحديث":0.10,"مجهول":0.10,"مجهول العين":0.05,
    "متروك":0.05,"متروك الحديث":0.05,"ساقط":0.05,"مختلط":0.05,
    "متهم":0.0,"كذاب":0.0,"وضاع":0.0,"دجال":0.0,
}
COMPANIONS = {"انس","انس بن مالك","ابو هريرة","ابي هريرة","عائشة","ابن عباس",
    "ابن مسعود","ابن عمر","جابر","ابو سعيد","عمر","عمر بن الخطاب","علي","ابو بكر","عثمان"}

def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

print("=" * 65)
print("  FINAL FYP1 PIPELINE")
print("  Hybrid + avg-bottom-2 + calibrated thresholds + 650K evaluation")
print("=" * 65)

# ── Build hybrid scores ──
print("\nBuilding hybrid narrator scores...")
enc = json.load(open("output/shamela/narrator_encyclopedia.json", encoding="utf-8"))
alias = json.load(open("output/shamela/alias_index.json", encoding="utf-8"))
with open("output/stage2/node_names.json", encoding="utf-8") as f: id_to_name = json.load(f)
classifier = json.load(open("output/track_a_classifier/narrator_scores_classifier.json", encoding="utf-8"))

entry_best = {}
for e in enc:
    for g in e.get("grades", []):
        for t in sorted(GRADE_MAP, key=len, reverse=True):
            if t in g: entry_best[e["entry_id"]] = max(entry_best.get(e["entry_id"],0), GRADE_MAP[t]); break

companion_eid = set(e["entry_id"] for e in enc if "صحاب" in norm(e.get("full_text","")+e.get("name","")))

name_score = {}
for nid_str, name in id_to_name.items():
    nid, nk = int(nid_str), norm(name)
    score = 0.632
    if nk in COMPANIONS: score = 1.0
    elif nk in alias:
        eid = alias[nk][0]
        if eid in companion_eid: score = 1.0
        elif eid in entry_best: score = entry_best[eid]
    if score == 0.632:
        cv = classifier.get(nid_str, classifier.get(str(nid)))
        if cv and float(cv) != 0.632: score = float(cv)
    name_score[nk] = score

rm = json.load(open("output/risk_modifiers/risk_modifiers.json", encoding="utf-8"))
er = {int(k): min(m["modifier"] for m in v) for k, v in rm.items()}
nr = {}
for ak, eids in alias.items():
    if eids[0] in er: nr[norm(ak)] = er[eids[0]]

print(f"  Scores built: {len(name_score):,}")

# ── Score 650K chains ──
print(f"\nScoring 650K chains (avg-bottom-2)...")
t0 = time.time()
total = 0
score_buckets = {g:0 for g in ["Sahih","Hasan","Daif","Matruk"]}

for chunk in pd.read_csv("Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv",
                         usecols=["Sanad"], chunksize=100_000, dtype_backend="pyarrow"):
    ct0 = time.time()
    for _, row in chunk.iterrows():
        s = row["Sanad"]
        if pd.isna(s): continue
        try: ch = json.loads(s.replace("'", '"'))
        except:
            try: ch = ast.literal_eval(s)
            except: continue
        names = [n.strip() for n in ch if len(n.strip()) > 3]
        if len(names) < 2: continue

        scores = sorted([name_score.get(norm(n), 0.632) * nr.get(norm(n), 1.0) for n in names])
        avg2 = (scores[0] + scores[1]) / 2 if len(scores) >= 2 else scores[0]
        g = "Sahih" if avg2 >= 0.45 else "Hasan" if avg2 >= 0.425 else "Daif" if avg2 >= 0.15 else "Matruk"
        score_buckets[g] += 1
        total += 1

    print(f"  Chunk: {total:,} chains ({time.time()-ct0:.0f}s)")

ttime = time.time() - t0
print(f"\n{'=' * 65}")
print(f"  650K CHAIN EVALUATION COMPLETE")
print(f"  {total:,} chains in {ttime:.0f}s")
print(f"{'=' * 65}")
print(f"\nGrade distribution (avg-bottom-2):")
for g in ["Sahih","Hasan","Daif","Matruk"]:
    pct = score_buckets[g] / total * 100
    print(f"  {g:10s}: {score_buckets[g]:>8,} ({pct:.1f}%)")

# ── Evaluate against Kutub as-Sitta ──
print(f"\nKutub as-Sitta validation:")
eval_df = pd.read_csv("output/stage1/kutub_sitta_evaluation.csv")
gl = {"Ṣaḥīḥ":"Sahih","Ḥasan":"Hasan","Ḍaʿīf":"Daif","Mawḍūʿ":"Matruk"}
eval_df["label"] = eval_df["grade_normalised"].map(gl).fillna("Daif")

sanad_col = []
for chunk in pd.read_csv("Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv",
                         usecols=["Sanad"], chunksize=100_000, dtype_backend="pyarrow"):
    sanad_col.extend(chunk["Sanad"].tolist())

align = json.load(open("output/stage1/alignment_results.json", encoding="utf-8"))
preds, actuals = [], []
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
    scores = sorted([name_score.get(norm(n), 0.632) * nr.get(norm(n), 1.0) for n in names])
    avg2 = (scores[0] + scores[1]) / 2 if len(scores) >= 2 else scores[0]
    g = "Sahih" if avg2 >= 0.45 else "Hasan" if avg2 >= 0.425 else "Daif" if avg2 >= 0.15 else "Matruk"
    preds.append(g)
    actuals.append(eval_df.iloc[aidx]["label"])

print(f"\n  Confusion matrix:")
grades = ["Sahih","Hasan","Daif","Matruk"]
cm = {(a,p):0 for a in grades for p in grades}
for a,p in zip(actuals, preds): cm[(a,p)] += 1
print(f"{'':>10s} " + " ".join(f"{g:>8s}" for g in grades))
for a in grades:
    print(f"{a:>8s}  " + " ".join(f"{cm[(a,p)]:>8d}" for p in grades))

acc = sum(1 for i in range(len(actuals)) if actuals[i]==preds[i])/len(actuals)
print(f"\n  Accuracy: {acc:.4f}")
print(f"  Sahih recall: {cm[('Sahih','Sahih')]/sum(cm[('Sahih',p)] for p in grades):.4f}")
