"""Evaluate 1,200 from Sanadset ONLY. No HF, no alignment."""
import sys, re, ast, json
import pandas as pd
import numpy as np
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")

def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

# ── Build narrator scores ──
GRADE_MAP = {"صحابي":1.0,"ثقة":0.85,"صدوق":0.70,"مقبول":0.40,"ضعيف":0.15,"متروك":0.05,"كذاب":0.0}
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

# ── Sample 200 per book from Sanadset ──
print("Loading Sanadset...")
s = pd.read_csv(
    "Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv",
    usecols=["Book", "Sanad"], dtype_backend="pyarrow",
)

patterns = {
    "Bukhari": "صحيح البخاري", "Muslim": "صحيح مسلم",
    "Abu Dawud": "سنن أبي داود", "Tirmidhi": "جامع الترمذي",
    "Nasai": "السنن الكبرى للنسائي", "Ibn Majah": "سنن ابن ماجه",
}

rng = np.random.RandomState(42)
T = (0.45, 0.425, 0.15)
results = []

for name, pat in patterns.items():
    pool = s[s["Book"].str.contains(pat[:6], na=False) & s["Sanad"].notna()]
    pool = pool.sample(200, random_state=rng)
    correct = 0
    for _, row in pool.iterrows():
        sanad = row["Sanad"]
        try: ch = json.loads(sanad.replace("'", '"'))
        except:
            try: ch = ast.literal_eval(sanad)
            except: continue
        names = [n.strip() for n in ch if len(n.strip()) > 3]
        if len(names) < 2: continue
        sc = sorted([name_score.get(norm(n), 0.632) for n in names])
        cs = (sc[0]+sc[1])/2 if len(sc)>=2 else sc[0]
        grade = "Sahih" if cs>=T[0] else "Hasan" if cs>=T[1] else "Daif" if cs>=T[2] else "Matruk"
        is_sahih_pred = (grade == "Sahih")
        results.append({"book": name, "predicted": grade, "score": cs})

df = pd.DataFrame(results)
print(f"\nTotal: {len(df)} hadiths (200 per book × 6)")
print()

# Per-book summary
for name in patterns:
    sub = df[df["book"] == name]
    dist = sub["predicted"].value_counts()
    sahih_pct = dist.get("Sahih", 0) / 200 * 100
    daif_pct = (dist.get("Daif", 0) + dist.get("Matruk", 0)) / 200 * 100
    mean_s = sub["score"].mean()
    print(f"  {name:15s}: Sahih={sahih_pct:5.1f}%  Daif/M={daif_pct:5.1f}%  mean={mean_s:.3f}  dist={dict(dist)}")

# Bukhari + Muslim = known Sahih
for bm in ["Bukhari", "Muslim"]:
    sub = df[df["book"] == bm]
    sahih = (sub["predicted"] == "Sahih").sum()
    print(f"\n  {bm} (known Sahih): {sahih}/200 correctly Sahih ({sahih/200*100:.1f}%)")

print(f"\n  Total correctly Sahih (Bukhari+Muslim): {(df[df['book'].isin(['Bukhari','Muslim'])]['predicted']=='Sahih').sum()}/400 ({(df[df['book'].isin(['Bukhari','Muslim'])]['predicted']=='Sahih').sum()/400*100:.1f}%)")
