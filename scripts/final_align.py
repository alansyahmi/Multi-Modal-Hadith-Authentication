"""
Stable alignment: batch=5, gc every 20 batches, save progress every 1000 queries.
~60 min, won't OOM.
"""
import sys, os, pickle, time, re, gc, json
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz
sys.stdout.reconfigure(encoding="utf-8")

def norm(t, maxlen=200):
    if not t: return ""
    t = re.sub(r"[ً-ٌٍ-َّْٰٓ]", "", t)
    t = t.replace("أ","ا").replace("إ","ا").replace("آ","a").replace("ة","ه").replace("ى","ي")
    return re.sub(r"\s+", " ", t).strip()[:maxlen]

PROGRESS_FILE = "output/final_evaluation/align_progress.json"

# ── Load TF-IDF ──
print("Loading TF-IDF...")
with open("output/final_evaluation/tfidf_corpus.pkl", "rb") as f: vec = pickle.load(f)
tfidf = load_npz("output/final_evaluation/tfidf_matrix.npz")
print(f"  Corpus: {tfidf.shape}")

# ── Load queries ──
print("Loading queries...")
all_hf = []
for f in sorted(os.listdir("Dataset/hadith_datasets HuggingFace")):
    if not f.endswith(".csv"): continue
    df = pd.read_csv(f"Dataset/hadith_datasets HuggingFace/{f}")
    for _, row in df.iterrows():
        all_hf.append({"book_file": f, "text": str(row.get("Arabic_Text","")),
                       "grade_raw": str(row.get("Grade",""))})

gm = {"sahih":"Sahih","hasan":"Hasan","daif":"Daif","da'if":"Daif","maudu":"Matruk","mawdu":"Matruk"}
queries, q_grades, q_books = [], [], []
for h in all_hf:
    grade = "Unknown"
    for k, v in gm.items():
        if k in h["grade_raw"].lower(): grade = v; break
    if grade != "Unknown":
        queries.append(norm(h["text"]))
        q_grades.append(grade)
        q_books.append(h["book_file"])
print(f"  {len(queries):,} queries")

# ── Resume check ──
results = []
start_from = 0
if os.path.exists(PROGRESS_FILE):
    saved = json.load(open(PROGRESS_FILE))
    results = saved["results"]
    start_from = saved["next_qi"]
    print(f"  Resuming from query {start_from} ({len(results):,} already matched)")

# ── Batch loop ──
print(f"\nProcessing queries {start_from} to {len(queries)} (batch=5)...")
BATCH = 5
GC_INTERVAL = 20
SAVE_INTERVAL = 1000
t0 = time.time()

for start in range(start_from, len(queries), BATCH):
    end = min(start + BATCH, len(queries))
    batch_texts = queries[start:end]
    batch_vec = vec.transform(batch_texts)
    sims = cosine_similarity(batch_vec, tfidf)  # 5 x 647K = 13MB

    for i in range(end - start):
        row = sims[i]
        best_idx = int(np.argmax(row))
        if row[best_idx] >= 0.5:
            results.append({
                "qi": start + i, "ci": best_idx,
                "grade": q_grades[start + i],
                "book": q_books[start + i],
                "sim": float(row[best_idx]),
            })
    del sims, batch_vec, batch_texts

    # Periodic cleanup
    batch_num = start // BATCH
    if batch_num % GC_INTERVAL == 0:
        gc.collect()

    # Progress + save every 1000 queries
    if (start + BATCH) % SAVE_INTERVAL == 0:
        elapsed = time.time() - t0
        rate = (start + BATCH - start_from) / max(elapsed, 1)
        eta = (len(queries) - start - BATCH) / max(rate, 1)
        print(f"  {start+BATCH}/{len(queries)} ({rate:.0f}/s, {len(results):,} matched, ETA {eta:.0f}s)")
        with open(PROGRESS_FILE, "w") as f:
            json.dump({"next_qi": end, "results": results}, f)

# ── Done ──
tt = time.time() - t0
print(f"\nDone: {len(results):,} matched in {tt:.0f}s")
by_grade = defaultdict(list)
for r in results: by_grade[r["grade"]].append(r)
for g in ["Sahih","Hasan","Daif","Matruk"]:
    print(f"  {g}: {len(by_grade[g]):,}")

# ── Sample 5K grade-balanced ──
TARGET = 1250
rng = np.random.RandomState(42)
final = []
for grade in ["Sahih", "Hasan", "Daif", "Matruk"]:
    pool = by_grade.get(grade, [])
    n = min(TARGET, len(pool))
    chosen = rng.choice(pool, n, replace=False).tolist() if n < len(pool) else pool[:n]
    final.extend(chosen)
    print(f"  {grade:10s}: taking {n}/{len(pool)}")

with open("output/final_evaluation/5k_stratified.json", "w", encoding="utf-8") as f:
    json.dump({"total": len(final), "results": final, "time_seconds": tt}, f, indent=2)
print(f"\nSaved: {len(final)} hadiths")

# Clean up progress file
if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)
