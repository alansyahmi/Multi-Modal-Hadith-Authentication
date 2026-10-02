"""
Ṭuruq corroboration — properly fixed.

The key insight: read Matn from the SANADSET (not evaluation set) via corpus_idx,
so the matn hash matches what was indexed.
"""
import json, re, ast, sys, hashlib, time
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
sys.stdout.reconfigure(encoding="utf-8")

GRADE_MAP = {
    "صحابي":1.0,"ثقة ثبت":0.95,"ثقة حافظ":0.95,"إمام":0.95,
    "حافظ":0.90,"ثبت":0.90,"ثقة":0.85,"صدوق":0.70,"مقبول":0.40,
    "ضعيف":0.15,"متروك":0.05,"كذاب":0.0,
}
def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

# ── Build narrator scores ──
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

def chain_avg2(names):
    sc = sorted([name_score.get(norm(n), 0.632) for n in names])
    return (sc[0] + sc[1]) / 2 if len(sc) >= 2 else sc[0]

print("=" * 65)
print("  ṬURŪQ CORROBORATION — FIXED MATN MATCHING")
print("=" * 65)

# ── Load Sanadset with BOTH Matn and Sanad ──
print("\nLoading Sanadset (Matn + Sanad)...")
sanad_rows = []  # (matn_norm, sanad_str)
for chunk in pd.read_csv(
    "Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv",
    usecols=["Matn", "Sanad"], chunksize=100_000, dtype_backend="pyarrow",
):
    for _, row in chunk.iterrows():
        m = row.get("Matn", ""); s = row.get("Sanad", "")
        if pd.isna(m) or pd.isna(s): continue
        sanad_rows.append((str(m), str(s)))

print(f"  {len(sanad_rows):,} rows loaded")

# ── Build matn hash → [row_indices] ──
print("Building matn hash index...")
matn_index = defaultdict(list)
for idx, (matn_text, _) in enumerate(sanad_rows):
    mk = re.sub(r"[ً-ٌٍ-َّْٰٓ]", "", matn_text).replace("أ","ا").replace("إ","ا").replace("ة","ه")[:100]
    h = hashlib.md5(mk.encode()).hexdigest()
    matn_index[h].append(idx)

multi = sum(1 for v in matn_index.values() if len(v) >= 2)
print(f"  {len(matn_index):,} unique matns | {multi:,} with ≥2 chains")

# ── Evaluate on Kutub as-Sitta ──
print("\nEvaluating with ṭuruq...")
eval_df = pd.read_csv("output/stage1/kutub_sitta_evaluation.csv")
gl = {"Ṣaḥīḥ":"Sahih","Ḥasan":"Hasan","Ḍaʿīf":"Daif","Mawḍūʿ":"Matruk"}
eval_df["label"] = eval_df["grade_normalised"].map(gl).fillna("Daif")
align = json.load(open("output/stage1/alignment_results.json", encoding="utf-8"))

T = (0.45, 0.425, 0.15)  # Optimized thresholds
turuq_chains_used = 0
single_preds, turuq_preds, actuals = [], [], []

for aidx, entry in enumerate(align):
    if not entry.get("is_match"): continue
    ci = int(entry["corpus_idx"])
    if ci >= len(sanad_rows): continue
    matn_text, sanad_str = sanad_rows[ci]
    if pd.isna(sanad_str): continue
    try: ch = json.loads(sanad_str.replace("'", '"'))
    except:
        try: ch = ast.literal_eval(sanad_str)
        except: continue
    names_single = [n.strip() for n in ch if len(n.strip()) > 3]
    if len(names_single) < 2: continue

    # Single-chain
    s_single = chain_avg2(names_single)
    def g(s):
        if s >= T[0]: return "Sahih"
        elif s >= T[1]: return "Hasan"
        elif s >= T[2]: return "Daif"
        else: return "Matruk"
    single_preds.append(g(s_single))

    # Ṭuruq: find ALL chains with same matn
    mk = re.sub(r"[ً-ٌٍ-َّْٰٓ]", "", matn_text).replace("أ","ا").replace("إ","ا").replace("ة","ه")[:100]
    h = hashlib.md5(mk.encode()).hexdigest()
    all_indices = matn_index.get(h, [ci])

    all_scores = []
    for idx in all_indices[:10]:  # Max 10 chains per matn
        _, ss = sanad_rows[idx]
        if pd.isna(ss): continue
        try: ch2 = json.loads(ss.replace("'", '"'))
        except:
            try: ch2 = ast.literal_eval(ss)
            except: continue
        ns2 = [n2.strip() for n2 in ch2 if len(n2.strip()) > 3]
        if len(ns2) >= 2:
            all_scores.append(chain_avg2(ns2))

    if len(all_scores) >= 2:
        turuq_preds.append(g(max(all_scores)))
        if max(all_scores) > s_single:
            turuq_chains_used += 1
    else:
        turuq_preds.append(g(s_single))

    actuals.append(eval_df.iloc[aidx]["label"])

# ── RESULTS ──
print(f"\n{'=' * 65}")
print(f"  ṬURŪQ RESULTS")
print(f"  Chains corroborated: {turuq_chains_used}/{len(actuals)}")
print(f"{'=' * 65}")

for label, preds in [("Single-chain (avg-bottom-2)", single_preds),
                      ("Ṭuruq (multi-chain)", turuq_preds)]:
    acc = accuracy_score(actuals, preds)
    sr = sum(1 for i in range(len(actuals)) if actuals[i]=="Sahih" and preds[i]=="Sahih")
    tr_sahih = sum(1 for a in actuals if a=="Sahih")
    dr = sum(1 for i in range(len(actuals)) if actuals[i]=="Daif" and preds[i]=="Daif")
    tr_daif = sum(1 for a in actuals if a=="Daif")

    print(f"\n  {label}:")
    grades = ["Sahih","Hasan","Daif","Matruk"]
    cm = {(a,p):0 for a in grades for p in grades}
    for a,p in zip(actuals, preds): cm[(a,p)] += 1
    print(f"  {'':>10s} " + " ".join(f"{g:>8s}" for g in grades))
    for a in grades:
        print(f"  {a:>8s}  " + " ".join(f"{cm[(a,p)]:>8d}" for p in grades))
    print(f"  Accuracy: {acc:.4f} | Sahih recall: {sr}/{tr_sahih} ({sr/tr_sahih*100:.1f}%) | Daif recall: {dr}/{tr_daif} ({dr/tr_daif*100:.1f}%)")

# Improvement from ṭuruq
tur_improved = sum(1 for i in range(len(actuals)) if single_preds[i] != turuq_preds[i])
print(f"\n  Hadiths improved by ṭuruq: {tur_improved}")
if tur_improved > 0:
    print(f"  Showing up to 10:")
    count = 0
    for i in range(len(actuals)):
        if single_preds[i] != turuq_preds[i] and count < 10:
            ci = int(align[i]["corpus_idx"])
            mk = re.sub(r"[ً-ٌٍ-َّْٰٓ]", "", sanad_rows[ci][0]).replace("أ","ا").replace("إ","ا").replace("ة","ه")[:100]
            h = hashlib.md5(mk.encode()).hexdigest()
            n_chain = len(matn_index.get(h, []))
            print(f"    {eval_df.iloc[i]['book_name'][:25]:25s} {actuals[i]:>8s} → {single_preds[i]:>8s} → {turuq_preds[i]:>8s} ({n_chain} chains)")
            count += 1

# ── Save ──
pd.DataFrame({"actual": actuals, "single": single_preds, "turuq": turuq_preds}).to_csv(
    "output/final_evaluation/turuq_breakthrough.csv", index=False)
print(f"\nSaved: output/final_evaluation/turuq_breakthrough.csv")
