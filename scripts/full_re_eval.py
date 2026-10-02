"""
Full re-evaluation with fixed alias + ṭuruq + companion rule + mutawatir test.
"""
import json, ast, sys, re, hashlib, time
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
sys.stdout.reconfigure(encoding="utf-8")

SANADSET = "Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv"
NODE_NAMES = "output/stage2/node_names.json"
CLASSIFIER = "output/track_a_classifier/narrator_scores_classifier.json"
RISK = "output/risk_modifiers/risk_modifiers.json"
ALIAS = "output/shamela/alias_index.json"

def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

# ── Load scores with fixes ──
with open(NODE_NAMES, encoding="utf-8") as f: id_to_name = json.load(f)
cls = json.load(open(CLASSIFIER, encoding="utf-8"))
name_score = {}
for nid_str, name in id_to_name.items():
    name_score[norm(name)] = float(cls.get(nid_str, cls.get(str(int(nid_str)), 0.632)))

# Risk
rm = json.load(open(RISK, encoding="utf-8"))
er = {int(k): min(m["modifier"] for m in v) for k, v in rm.items()}
al = json.load(open(ALIAS, encoding="utf-8"))
nr = {}
for ak, eids in al.items():
    if eids[0] in er: nr[norm(ak)] = er[eids[0]]

# Companion rule
enc = json.load(open("output/shamela/narrator_encyclopedia.json", encoding="utf-8"))
companion_entries = set()
for e in enc:
    text = norm(e.get("full_text","") + e.get("name",""))
    if "صحاب" in text: companion_entries.add(e["entry_id"])
companion_names = set()
for ak, eids in al.items():
    if eids[0] in companion_entries: companion_names.add(norm(ak))

KEY_COMPANIONS = ["انس", "انس بن مالك", "ابو هريرة", "ابي هريرة", "عائشة",
                  "ابن عباس", "ابن مسعود", "ابن عمر", "جابر", "ابو سعيد", "عمر"]
def fix_score(nkey):
    s = name_score.get(nkey, 0.632)
    if nkey in companion_names: return 1.0
    for cn in KEY_COMPANIONS:
        if cn in nkey or nkey in cn: return 1.0
    return s

def chain_min_score(names):
    sc = [fix_score(norm(n)) * nr.get(norm(n), 1.0) for n in names]
    return min(sc)

def grade(s):
    if s >= 0.70: return "Sahih"
    elif s >= 0.55: return "Hasan"
    elif s >= 0.35: return "Daif"
    else: return "Matruk"

# ── Build matn index for ṭuruq ──
print("Building matn index...")
matn_index = defaultdict(list)
for chunk in pd.read_csv(SANADSET, usecols=["Matn","Sanad"], chunksize=100_000, dtype_backend="pyarrow"):
    for _, row in chunk.iterrows():
        m = row.get("Matn",""); s = row.get("Sanad","")
        if pd.isna(m) or pd.isna(s): continue
        mk = re.sub(r"[ً-ٌٍ-َّْٰٓ]", "", m).replace("أ","ا").replace("إ","ا").replace("ة","ه")[:100]
        if len(mk) < 15: continue
        try: ch = json.loads(s.replace("'", '"'))
        except:
            try: ch = ast.literal_eval(s)
            except: continue
        ns = tuple(n.strip() for n in ch if len(n.strip()) > 3)
        if len(ns) >= 2:
            h = hashlib.md5(mk.encode()).hexdigest()
            matn_index[h].append(ns)
print(f"  {len(matn_index):,} unique matns | {sum(1 for v in matn_index.values() if len(v)>=2):,} with ≥2 chains")

# ── Mutawatir test ──
MUTAWATIR = [
    "طلب العلم فريضة على كل مسلم", "بني الإسلام على خمس",
    "الدين النصيحة", "من كان يؤمن بالله واليوم الآخر فليقل خيرا أو ليصمت",
    "لا يؤمن أحدكم حتى يحب لأخيه ما يحب لنفسه",
]
print(f"\n=== MUTAWATIR TEST ===")
remaining = list(MUTAWATIR)
for chunk in pd.read_csv(SANADSET, usecols=["Matn","Sanad"], chunksize=50_000, dtype_backend="pyarrow"):
    for _, row in chunk.iterrows():
        m = str(row.get("Matn","")); s = row.get("Sanad","")
        if pd.isna(m) or pd.isna(s): continue
        for mt in list(remaining):
            if mt[:20] not in m: continue
            try: ch = json.loads(s.replace("'", '"'))
            except:
                try: ch = ast.literal_eval(s)
                except: ch = []
            ns = [n.strip() for n in ch if len(n.strip()) > 3]
            if len(ns) < 2: continue
            mk = re.sub(r"[ً-ٌٍ-َّْٰٓ]", "", m).replace("أ","ا").replace("إ","ا").replace("ة","ه")[:100]
            h = hashlib.md5(mk.encode()).hexdigest()
            chains = matn_index.get(h, [ns])

            # Single chain
            s_single = chain_min_score(ns)

            # Ṭuruq
            all_s = [chain_min_score(c) for c in chains]
            s_best = max(all_s)

            arrow = f"↑ to {grade(s_best)}" if grade(s_best) != grade(s_single) else ""
            flag = "✅" if grade(s_best) in ["Sahih","Hasan"] else "✗"
            print(f"  {flag} {mt[:40]:40s} single={s_single:.3f} {grade(s_single):>8s} → best={s_best:.3f} {grade(s_best):>8s} {arrow} ({len(chains)} chains)")
            remaining.remove(mt)
            break
    if not remaining: break

# ── Kutub as-Sitta evaluation ──
print(f"\n=== KUTUB AS-SITTA EVALUATION ===")
eval_df = pd.read_csv("output/stage1/kutub_sitta_evaluation.csv")
gl = {"Ṣaḥīḥ": "Sahih", "Ḥasan": "Hasan", "Ḍaʿīf": "Daif", "Mawḍūʿ": "Matruk"}
eval_df["label"] = eval_df["grade_normalised"].map(gl).fillna("Daif")

sanad_col = []
for chunk in pd.read_csv(SANADSET, usecols=["Sanad"], chunksize=100_000, dtype_backend="pyarrow"):
    sanad_col.extend(chunk["Sanad"].tolist())

align = json.load(open("output/stage1/alignment_results.json", encoding="utf-8"))
single_preds, turuq_preds, actuals = [], [], []
t0 = time.time()

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
    ns = [n.strip() for n in ch if len(n.strip()) > 3]
    if len(ns) < 2: continue

    row = eval_df.iloc[aidx]
    actual = row["label"]

    # Single chain
    s1 = chain_min_score(ns)
    single_preds.append(grade(s1))

    # Ṭuruq via matn matching
    mtext = str(row.get("Arabic_Text",""))
    mk = re.sub(r"[ً-ٌٍ-َّْٰٓ]", "", mtext).replace("أ","ا").replace("إ","ا").replace("ة","ه")[:100]
    h = hashlib.md5(mk.encode()).hexdigest()
    if h in matn_index and len(matn_index[h]) >= 2:
        all_s = [chain_min_score(c) for c in matn_index[h]]
        turuq_preds.append(grade(max(all_s)))
    else:
        turuq_preds.append(grade(s1))
    actuals.append(actual)

t1 = time.time()
print(f"Scored {len(actuals):,} hadiths in {t1-t0:.0f}s")

grades = ["Sahih","Hasan","Daif","Matruk"]
for label, preds in [("Single-chain", single_preds), ("Ṭuruq", turuq_preds)]:
    cm = confusion_matrix(actuals, preds, labels=grades)
    a = accuracy_score(actuals, preds)
    f = f1_score(actuals, preds, average="macro", zero_division=0)
    k = cohen_kappa_score(actuals, preds, weights="quadratic")
    print(f"\n  {label}: Acc={a:.4f} F1={f:.4f} κ={k:.4f}")
    print("          " + " ".join(f"{g:>8s}" for g in grades))
    for i, g in enumerate(grades):
        print(f"  {g:8s} " + " ".join(f"{cm[i][j]:>8d}" for j in range(4)))

# Save
pd.DataFrame({"actual": actuals, "single": single_preds, "turuq": turuq_preds}).to_csv(
    "output/final_evaluation/full_re_eval.csv", index=False)
print(f"\nSaved: output/final_evaluation/full_re_eval.csv")
