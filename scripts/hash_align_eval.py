"""
Grade-balanced 5K evaluation set via HashingVectorizer (memory efficient).
"""
import sys, re, json, ast, time, os
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from rapidfuzz import fuzz
sys.stdout.reconfigure(encoding="utf-8")

def norm(text, maxlen=200):
    if not text: return ""
    text = re.sub(r"[ً-ٌٍ-َّْٰٓ]", "", text)
    text = text.replace("أ","ا").replace("إ","ا").replace("آ","a").replace("ة","ه").replace("ى","ي")
    return re.sub(r"\s+", " ", text).strip()[:maxlen]

# ── Load HF hadiths ──
print("Loading HF hadiths...")
all_hf = []
for f in sorted(os.listdir("Dataset/hadith_datasets HuggingFace")):
    if not f.endswith(".csv"): continue
    df = pd.read_csv(f"Dataset/hadith_datasets HuggingFace/{f}")
    for _, row in df.iterrows():
        all_hf.append({"book": f.replace(".csv","").replace("%20"," "),
            "text": str(row.get("Arabic_Text","")), "grade_raw": str(row.get("Grade",""))})
gm = {"sahih":"Sahih","hasan":"Hasan","daif":"Daif","da'if":"Daif","maudu":"Matruk","mawdu":"Matruk"}
for h in all_hf:
    h["grade"] = "Unknown"
    for k, v in gm.items():
        if k in h["grade_raw"].lower(): h["grade"] = v; break
graded = [h for h in all_hf if h["grade"] != "Unknown"]
print(f"  {len(graded):,} graded")
for g, c in Counter(h["grade"] for h in graded).most_common():
    print(f"    {g}: {c:,}")

# ── Hashing vectorizer (no fitting needed) ──
vec = HashingVectorizer(n_features=50000, norm="l2", alternate_sign=False)

# ── Pre-hash queries ──
print(f"\nHashing {len(graded):,} queries...")
query_texts = [norm(h["text"]) for h in graded]
query_vec = vec.transform(query_texts)
print(f"  Query matrix: {query_vec.shape}")

# ── Match chunk by chunk ──
print(f"\nMatching against Sanadset chunks...")
matched = {"Sahih":[], "Hasan":[], "Daif":[], "Matruk":[]}
matched_q = set()
t0 = time.time()

for ci, chunk in enumerate(pd.read_csv(
    "Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv",
    usecols=["Matn", "Sanad"], chunksize=50000, dtype_backend="pyarrow",
)):
    texts, sancs = [], []
    for _, row in chunk.iterrows():
        m, s = row.get("Matn",""), row.get("Sanad","")
        if pd.isna(m) or pd.isna(s): continue
        nm = norm(str(m))
        if len(nm) >= 15: texts.append(nm); sancs.append(str(s))
    if not texts: continue

    sims = query_vec * vec.transform(texts).T  # Sparse dot product

    for qi in range(len(graded)):
        if qi in matched_q: continue
        row = sims[qi].toarray().flatten() if hasattr(sims[qi], "toarray") else sims[qi]
        top = np.argsort(row)[-3:][::-1]
        for idx in top:
            if row[idx] < 0.5: continue
            sc = fuzz.partial_ratio(query_texts[qi][:100], texts[idx][:100])
            if sc >= 80:
                matched[graded[qi]["grade"]].append({"sanad": sancs[idx],
                    "grade": graded[qi]["grade"], "book": graded[qi]["book"]})
                matched_q.add(qi)
                break

    if (ci + 1) % 2 == 0:
        el = time.time() - t0
        print(f"  Chunk {ci+1}: {len(matched_q):,}/{len(graded):,} matched ({el:.0f}s)")
    if len(matched_q) >= len(graded):
        print("  All matched!")
        break

t = time.time() - t0
print(f"\nDone in {t:.0f}s")
for g in ["Sahih","Hasan","Daif","Matruk"]:
    print(f"  {g}: {len(matched[g]):,}")

# ── Grade-balanced sample ──
rng = np.random.RandomState(42)
eval_set = []
for g in ["Sahih","Hasan","Daif","Matruk"]:
    take = min(1250, len(matched[g]))
    chosen = rng.choice(matched[g], take, replace=False).tolist() if isinstance(matched[g], list) and len(matched[g]) > take else matched[g]
    if not isinstance(chosen, list): chosen = chosen.tolist()
    eval_set.extend(chosen)
    print(f"  Taking {take}/{len(matched[g])} from {g}")
rng.shuffle(eval_set)
print(f"\nTotal eval set: {len(eval_set):,}")

# ── Evaluate ──
print("\nScoring chains...")
def norm_t(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","a").replace("إ","a").replace("آ","a").replace("ة","ه").replace("ى","ي").strip()
G = {"صحابي":1.0,"ثقة":0.85,"صدوق":0.70,"مقبول":0.40,"ضعيف":0.15,"متروك":0.05,"كذاب":0.0}
enc = json.load(open("output/shamela/narrator_encyclopedia.json", encoding="utf-8"))
alias = json.load(open("output/shamela/alias_index.json", encoding="utf-8"))
with open("output/stage2/node_names.json", encoding="utf-8") as f: id_to_name = json.load(f)
classifier = json.load(open("output/track_a_classifier/narrator_scores_classifier.json", encoding="utf-8"))
entry_best = {}
for e in enc:
    for g in e.get("grades", []):
        for t in sorted(G, key=len, reverse=True):
            if t in g: entry_best[e["entry_id"]] = max(entry_best.get(e["entry_id"],0), G[t]); break
name_score = {}
for nid_str, name in id_to_name.items():
    nid, nk = int(nid_str), norm_t(name)
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
    sc = sorted([name_score.get(norm_t(n), 0.632) for n in names])
    cs = (sc[0]+sc[1])/2 if len(sc)>=2 else sc[0]
    g = "Sahih" if cs>=T[0] else "Hasan" if cs>=T[1] else "Daif" if cs>=T[2] else "Matruk"
    actual.append(h["grade"]); preds.append(g)

bl = max(Counter(actual).values())/max(len(actual),1)
acc = accuracy_score(actual, preds)
f1 = f1_score(actual, preds, average="macro", zero_division=0)
k = cohen_kappa_score(actual, preds, weights="quadratic")
print(f"\nBaseline: {bl:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Macro: {f1:.4f}")
print(f"Cohen's κ: {k:.4f}")
grades = ["Sahih","Hasan","Daif","Matruk"]
cm = {(a,p):0 for a in grades for p in grades}
for a,p in zip(actual, preds): cm[(a,p)] += 1
print(f"\n{'':>10s} " + " ".join(f"{g:>8s}" for g in grades))
for a in grades:
    print(f"  {a:>8s} " + " ".join(f"{cm[(a,p)]:>8d}" for p in grades))

with open("output/final_evaluation/balanced_5k_final.json","w") as f:
    json.dump({"n":len(eval_set),"accuracy":acc,"f1":f1,"kappa":k,"baseline":bl}, f, indent=2)
print(f"\nSaved: output/final_evaluation/balanced_5k_final.json")
