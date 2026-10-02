"""
Fast balanced evaluation set builder.
Uses rapidfuzz + fingerprint matching (no TF-IDF memory issues).
"""
import sys, re, json, ast, time, os
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
sys.stdout.reconfigure(encoding="utf-8")

def norm_matn(text):
    if not text: return ""
    text = re.sub(r"[ً-ٌٍ-َّْٰٓ]", "", text)
    text = text.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    text = text.replace("ة","ه").replace("ى","ي")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:200]

# ── 1. Load HF and normalize grades ──
print("Loading HuggingFace hadiths...")
all_hf = []
for f in sorted(os.listdir("Dataset/hadith_datasets HuggingFace")):
    if not f.endswith(".csv"): continue
    df = pd.read_csv(f"Dataset/hadith_datasets HuggingFace/{f}")
    for _, row in df.iterrows():
        all_hf.append({
            "book": f.replace(".csv","").replace("%20"," "),
            "text": str(row.get("Arabic_Text","")),
            "grade_raw": str(row.get("Grade","")),
        })

grade_map = {"sahih":"Sahih","hasan":"Hasan","daif":"Daif","da'if":"Daif",
             "da`if":"Daif","maudu":"Matruk","mawdu":"Matruk","munkar":"Daif"}
for h in all_hf:
    h["grade"] = "Unknown"
    for k, v in grade_map.items():
        if k in h["grade_raw"].lower(): h["grade"] = v; break

graded = [h for h in all_hf if h["grade"] != "Unknown"]
print(f"  {len(graded):,} graded ({len(all_hf):,} total)")
for g, c in Counter(h["grade"] for h in graded).most_common():
    print(f"    {g:10s}: {c:,}")

# ── 2. Load Sanadset and build fingerprint index ──
print("\nBuilding Sanadset fingerprint index...")
# Only load rows that have the SAME matn start as our queries
# (Filter by first word of matn to reduce candidates)
query_first_words = {}
for h in graded:
    nm = norm_matn(h["text"])
    if nm:
        fw = nm.split()[0][:10] if nm.split() else ""
        if fw:
            query_first_words.setdefault(fw, []).append(h)

# Load Sanadset and index by first word
fp_index = defaultdict(list)  # first_word → [(norm_matn, sanad_str)]
for chunk in pd.read_csv(
    "Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv",
    usecols=["Matn", "Sanad"], chunksize=100_000, dtype_backend="pyarrow",
):
    for _, row in chunk.iterrows():
        m = row.get("Matn",""); s = row.get("Sanad","")
        if pd.isna(m) or pd.isna(s): continue
        nm = norm_matn(str(m))
        if len(nm) >= 15:
            fw = nm.split()[0][:10] if nm.split() else ""
            if fw:
                fp_index[fw].append((nm, str(s)))

print(f"  {sum(len(v) for v in fp_index.values()):,} indexed under {len(fp_index):,} first words")

# ── 3. Match queries to Sanadset ──
print(f"\nMatching {len(graded):,} queries...")
matched = {"Sahih": [], "Hasan": [], "Daif": [], "Matruk": []}
t0 = time.time()

for h in graded:
    nm = norm_matn(h["text"])
    if not nm: continue
    fw = nm.split()[0][:10] if nm.split() else ""
    candidates = fp_index.get(fw, [])
    if not candidates: continue

    best_score, best_sanad = 0, ""
    for sanad_nm, sanad_raw in candidates[:50]:  # Check up to 50 per first word
        score = fuzz.partial_ratio(nm, sanad_nm)
        if score > best_score and score >= 80:
            best_score, best_sanad = score, sanad_raw

    if best_sanad:
        matched[h["grade"]].append({
            "grade": h["grade"],
            "book": h["book"],
            "sanad": best_sanad,
            "similarity": best_score,
        })

ttime = time.time() - t0
print(f"  Done in {ttime:.0f}s")
for g in ["Sahih","Hasan","Daif","Matruk"]:
    print(f"    {g:10s}: {len(matched[g]):,}")

# ── 4. Grade-balanced sampling (5,000 total) ──
print(f"\nSampling 5,000 grade-balanced...")
T = 1250
rng = np.random.RandomState(42)
eval_set = []
for g in ["Sahih","Hasan","Daif","Matruk"]:
    pool = matched[g]
    take = min(T, len(pool))
    chosen = rng.choice(pool, take, replace=False).tolist() if isinstance(pool, list) else list(pool)[:take]
    eval_set.extend(chosen)
    print(f"  {g:10s}: taking {take}/{len(pool)}")

rng.shuffle(eval_set)
print(f"\n  Total: {len(eval_set):,}")

# ── 5. Score and evaluate ──
print("\nScoring chains...")
def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

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

from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
actual, preds = [], []
T = (0.45, 0.425, 0.15)

for h in eval_set:
    s = h["sanad"]
    try: ch = json.loads(s.replace("'", '"'))
    except:
        try: ch = ast.literal_eval(s)
        except: continue
    names = [n.strip() for n in ch if len(n.strip()) > 3]
    if len(names) < 2: continue
    sc = sorted([name_score.get(norm(n), 0.632) for n in names])
    cs = (sc[0]+sc[1])/2 if len(sc)>=2 else sc[0]
    g = "Sahih" if cs>=T[0] else "Hasan" if cs>=T[1] else "Daif" if cs>=T[2] else "Matruk"
    actual.append(h["grade"])
    preds.append(g)

if actual:
    baseline = max(Counter(actual).values())/len(actual)
    acc = accuracy_score(actual, preds)
    f1 = f1_score(actual, preds, average="macro", zero_division=0)
    k = cohen_kappa_score(actual, preds, weights="quadratic")
    print(f"\n  Baseline: {baseline:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Macro: {f1:.4f}")
    print(f"  Cohen's κ: {k:.4f}")

    grades = ["Sahih","Hasan","Daif","Matruk"]
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>10s} " + " ".join(f"{g:>8s}" for g in grades))
    for a in grades:
        row = [sum(1 for i in range(len(actual)) if actual[i]==a and preds[i]==p) for p in grades]
        print(f"  {a:>8s} " + " ".join(f"{c:>8d}" for c in row))

# ── Save ──
with open("output/final_evaluation/balanced_5k_results.json", "w", encoding="utf-8") as f:
    json.dump({"n": len(eval_set), "accuracy": acc, "f1_macro": f1,
               "cohen_kappa": k, "predictions": list(zip(actual, preds))}, f, indent=2)
print(f"\nSaved: output/final_evaluation/balanced_5k_results.json")
