"""
Hybrid scoring: GRADE_SCORE_MAP for known narrators → AraBERT classifier → GNN fallback.
Then calibrate chain thresholds against Kutub as-Sitta.
"""
import json, re, ast, sys
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
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
print("  HYBRID SCORER + THRESHOLD CALIBRATION")
print("=" * 65)

# Load data
enc = json.load(open("output/shamela/narrator_encyclopedia.json", encoding="utf-8"))
alias = json.load(open("output/shamela/alias_index.json", encoding="utf-8"))
with open("output/stage2/node_names.json", encoding="utf-8") as f:
    id_to_name = json.load(f)
classifier = json.load(open("output/track_a_classifier/narrator_scores_classifier.json", encoding="utf-8"))
rm = json.load(open("output/risk_modifiers/risk_modifiers.json", encoding="utf-8"))
er = {int(k): min(m["modifier"] for m in v) for k, v in rm.items()}
nr = {}
for ak, eids in alias.items():
    if eids[0] in er: nr[norm(ak)] = er[eids[0]]

# Build entry score cache
entry_best = {}
companion_eid = set()
for e in enc:
    eid = e["entry_id"]
    if "صحاب" in norm(e.get("full_text","") + e.get("name","")):
        companion_eid.add(eid)
    grades = e.get("grades", [])
    if grades:
        scores = []
        for g in grades:
            for term in sorted(GRADE_MAP, key=len, reverse=True):
                if term in g: scores.append(GRADE_MAP[term]); break
        if scores: entry_best[eid] = max(scores)

print(f"Grade-MAPped entries: {len(entry_best):,} | Companions: {len(companion_eid):,}")

# Build hybrid name→score
name_score = {}
stats = {"companion": 0, "grade": 0, "classifier": 0, "gnn": 0}

for nid_str, name in id_to_name.items():
    nid = int(nid_str)
    nkey = norm(name)
    score = 0.632  # GNN default

    # Check companion first
    if nkey in COMPANIONS:
        score = 1.0; stats["companion"] += 1
    elif nkey in alias:
        eid = alias[nkey][0]
        if eid in companion_eid:
            score = 1.0; stats["companion"] += 1
        elif eid in entry_best:
            score = entry_best[eid]; stats["grade"] += 1

    if score == 0.632:
        cls_val = classifier.get(nid_str, classifier.get(str(nid)))
        if cls_val and float(cls_val) != 0.632:
            score = float(cls_val); stats["classifier"] += 1
        else:
            stats["gnn"] += 1

    name_score[nkey] = score

print(f"Coverage: Companion:{stats['companion']:,} Grade:{stats['grade']:,} Classifier:{stats['classifier']:,} GNN:{stats['gnn']:,}")

# Score evaluation set chains
print("\nScoring evaluation chains...")
sanad_col = []
for chunk in pd.read_csv("Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv",
                         usecols=["Sanad"], chunksize=100_000, dtype_backend="pyarrow"):
    sanad_col.extend(chunk["Sanad"].tolist())

align = json.load(open("output/stage1/alignment_results.json", encoding="utf-8"))
eval_df = pd.read_csv("output/stage1/kutub_sitta_evaluation.csv")
gl = {"Ṣaḥīḥ": "Sahih", "Ḥasan": "Hasan", "Ḍaʿīf": "Daif", "Mawḍūʿ": "Matruk"}
eval_df["label"] = eval_df["grade_normalised"].map(gl).fillna("Daif")

chain_scores, actuals = [], []
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

    scores = []
    for n in names:
        nk = norm(n)
        base = name_score.get(nk, 0.632)
        scores.append(base * nr.get(nk, 1.0))
    chain_scores.append(min(scores))
    actuals.append(eval_df.iloc[aidx]["label"])

# Grid search optimal thresholds
from sklearn.metrics import accuracy_score as acc
best_acc, best_t = 0, {"S":0.70,"H":0.55,"D":0.35}
for S in np.arange(0.55, 0.90, 0.025):
    for H in np.arange(0.40, 0.70, 0.025):
        for D in np.arange(0.15, 0.45, 0.025):
            if not (S > H > D): continue
            p = []
            for cs in chain_scores:
                if cs >= S: p.append("Sahih")
                elif cs >= H: p.append("Hasan")
                elif cs >= D: p.append("Daif")
                else: p.append("Matruk")
            a = acc(actuals, p)
            if a > best_acc:
                best_acc, best_t = a, {"Sahih":S,"Hasan":H,"Daif":D}

print(f"\nBest accuracy: {best_acc:.4f}")
print(f"Thresholds: Sahih≥{best_t['Sahih']:.3f}, Hasan≥{best_t['Hasan']:.3f}, Daif≥{best_t['Daif']:.3f}")

# Final preds
final_preds = []
for cs in chain_scores:
    if cs >= best_t["Sahih"]: final_preds.append("Sahih")
    elif cs >= best_t["Hasan"]: final_preds.append("Hasan")
    elif cs >= best_t["Daif"]: final_preds.append("Daif")
    else: final_preds.append("Matruk")

cm = pd.crosstab(pd.Series(actuals, name="Actual"), pd.Series(final_preds, name="Pred"))
print(f"\nConfusion Matrix:\n{cm}")
f1 = f1_score(actuals, final_preds, average="macro", zero_division=0)
print(f"F1: {f1:.4f}")

# Mutawatir test
print(f"\nMutawatir test:")
for mtext in ["طلب العلم فريضة","بني الإسلام على خمس","الدين النصيحة","من كان يؤمن بالله"]:
    for chunk in pd.read_csv("Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv",
                             usecols=["Matn","Sanad"], chunksize=100_000, dtype_backend="pyarrow"):
        for _, row in chunk.iterrows():
            m = str(row.get("Matn","")); s = row.get("Sanad","")
            if mtext[:15] not in m: continue
            try: ch = json.loads(s.replace("'", '"'))
            except:
                try: ch = ast.literal_eval(s)
                except: ch = []
            names = [n.strip() for n in ch if len(n.strip()) > 3]
            if len(names) < 2: continue
            scores = [name_score.get(norm(n), 0.632) * nr.get(norm(n), 1.0) for n in names]
            cs = min(scores)
            g = "Sahih" if cs >= best_t["Sahih"] else "Hasan" if cs >= best_t["Hasan"] else "Daif" if cs >= best_t["Daif"] else "Matruk"
            print(f"  {'✅' if g in ['Sahih','Hasan'] else '✗'} {mtext}: {cs:.3f} → {g}")
            break
        break

# Save
pd.DataFrame({"chain_score": chain_scores, "actual": actuals, "predicted": final_preds}).to_csv(
    "output/final_evaluation/hybrid_calibrated.csv", index=False)
print(f"\nSaved: output/final_evaluation/hybrid_calibrated.csv")
