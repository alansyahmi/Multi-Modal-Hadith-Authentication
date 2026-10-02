"""
Align ALL graded HuggingFace hadiths against Sanadset using TF-IDF + fuzzy matching.
Then build grade-balanced 5K evaluation set.
"""
import sys, re, time, os
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz.fuzz import partial_ratio
sys.stdout.reconfigure(encoding="utf-8")

def norm_matn(text):
    if not text: return ""
    text = re.sub(r"[ً-ٌٍ-َّْٰٓ]", "", text)
    text = text.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    text = text.replace("ة","ه").replace("ى","ي")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── 1. Load ALL HF hadiths and normalize grades ──
print("Loading all HF hadiths...")
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

grade_map = {"sahih": "Sahih", "hasan": "Hasan", "da'if": "Daif", "da`if": "Daif",
             "daif": "Daif", "maudu": "Matruk", "mawdu": "Matruk", "munkar": "Daif", "shadh": "Daif"}
for h in all_hf:
    h["grade"] = "Unknown"
    for key, val in grade_map.items():
        if key in h["grade_raw"].lower():
            h["grade"] = val; break

graded = [h for h in all_hf if h["grade"] != "Unknown"]
print(f"  Total HF: {len(all_hf):,} | Graded: {len(graded):,}")
dist = Counter(h["grade"] for h in graded)
for g, c in dist.most_common(): print(f"    {g:10s}: {c:,}")

# ── 2. Build Sanadset index ──
print("\nBuilding Sanadset corpus...")
corpus_texts = []
corpus_sanads = []
for chunk in pd.read_csv(
    "Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv",
    usecols=["Matn", "Sanad"], chunksize=100_000, dtype_backend="pyarrow",
):
    for _, row in chunk.iterrows():
        m = row.get("Matn",""); s = row.get("Sanad","")
        if pd.isna(m) or pd.isna(s): continue
        nm = norm_matn(str(m))
        if len(nm) >= 20:
            corpus_texts.append(nm)
            corpus_sanads.append(str(s))
print(f"  Corpus: {len(corpus_texts):,} matns")

# ── 3. TF-IDF fit_transform on corpus ──
print("\nFitting TF-IDF on corpus...")
t0 = time.time()
vectorizer = TfidfVectorizer(
    analyzer="word", ngram_range=(1, 2), max_features=50000,
    sublinear_tf=True, dtype=np.float32
)
corpus_tfidf = vectorizer.fit_transform(corpus_texts)
print(f"  TF-IDF matrix: {corpus_tfidf.shape} in {time.time()-t0:.0f}s")

# ── 4. Query each graded hadith (batched for memory) ──
print(f"\nQuerying {len(graded):,} graded hadiths...")
from sklearn.metrics.pairwise import cosine_similarity

BATCH = 100
matched = {"Sahih": [], "Hasan": [], "Daif": [], "Matruk": []}
t0 = time.time()

for batch_start in range(0, len(graded), BATCH):
    batch = graded[batch_start:batch_start + BATCH]
    batch_norms = [norm_matn(h["text"]) for h in batch]
    query_vec = vectorizer.transform(batch_norms)
    sims = cosine_similarity(query_vec, corpus_tfidf)

    for i, h in enumerate(batch):
        top = np.argsort(sims[i])[-5:][::-1]
        best = None
        for idx in top:
            if sims[i][idx] < 0.5: continue
            score = partial_ratio(batch_norms[i][:200], corpus_texts[idx][:200])
            if score >= 80:
                best = (idx, score)
                break
        if best:
            idx, score = best
            matched[h["grade"]].append({
                "hf_text": h["text"][:200],
                "sanad_norm": corpus_texts[idx],
                "sanad_raw": corpus_sanads[idx],
                "similarity": score,
                "book": h["book"],
            })

    if (batch_start // BATCH) % 10 == 0:
        el = time.time() - t0
        print(f"  {batch_start + BATCH}/{len(graded)} ({batch_start/max(el,1):.0f}/s, {sum(len(v) for v in matched.values()):,} matched)")

ttime = time.time() - t0
print(f"\n  Done in {ttime:.0f}s")

# ── 5. Grade-balanced sampling ──
print("\nGrade-balanced sampling (target: 5,000)...")
TARGET = 1250  # 4 × 1250 = 5000
rng = np.random.RandomState(42)

from rapidfuzz import fuzz as rfuzz
eval_hadiths = []

for grade in ["Sahih", "Hasan", "Daif", "Matruk"]:
    pool = matched[grade]
    print(f"  {grade:10s}: {len(pool):,} matched", end="")
    sample = min(TARGET, len(pool))
    if sample < TARGET:
        print(f" ⚠️ only {sample} available")
    else:
        print()
    chosen = rng.choice(pool, sample, replace=False).tolist() if isinstance(pool, list) else list(pool)
    for c in chosen:
        c["grade"] = grade
    eval_hadiths.extend(chosen)

rng.shuffle(eval_hadiths)
print(f"\n  Total balanced set: {len(eval_hadiths):,}")
for g in ["Sahih", "Hasan", "Daif", "Matruk"]:
    print(f"    {g:10s}: {sum(1 for e in eval_hadiths if e['grade']==g)}")

# ── 6. Evaluate with avg-bottom-2 + hybrid scores ──
print("\nEvaluating chain scores...")
GRADE_MAP = {"صحابي":1.0,"ثقة":0.85,"صدوق":0.70,"مقبول":0.40,"ضعيف":0.15,"متروك":0.05,"كذاب":0.0}
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

from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
actual, preds = [], []
T = (0.45, 0.425, 0.15)

for h in eval_hadiths:
    s = h["sanad_raw"]
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
    baseline = max(Counter(actual).values()) / len(actual)
    acc = accuracy_score(actual, preds)
    print(f"\n  Baseline (always majority class): {baseline:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Macro: {f1_score(actual, preds, average='macro', zero_division=0):.4f}")
    print(f"  Cohen's κ: {cohen_kappa_score(actual, preds, weights='quadratic'):.4f}")

    # Confusion matrix
    grades = ["Sahih","Hasan","Daif","Matruk"]
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>10s} " + " ".join(f"{g:>8s}" for g in grades))
    for a in grades:
        row = [sum(1 for i in range(len(actual)) if actual[i]==a and preds[i]==p) for p in grades]
        print(f"  {a:>8s} " + " ".join(f"{c:>8d}" for c in row))

# ── Save ──
import json
out_data = []
for h in eval_hadiths:
    out_data.append({"grade": h["grade"], "book": h["book"], "sanad": h["sanad_raw"]})
with open("output/final_evaluation/balanced_eval_5k.json", "w", encoding="utf-8") as f:
    json.dump(out_data, f, ensure_ascii=False)
print(f"\nSaved: output/final_evaluation/balanced_eval_5k.json ({len(out_data)} hadiths)")
