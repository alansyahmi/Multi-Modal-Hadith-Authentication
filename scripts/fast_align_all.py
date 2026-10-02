"""
Fast alignment: batch queries against cached TF-IDF matrix.
Small batch size (25) + explicit GC to avoid OOM.
"""
import sys, os, pickle, time, re, gc, json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"c:\Projects\FYP Project")

def norm(t):
    if not t: return ""
    t = re.sub(r"[ً-ٌٍ-َّْٰٓ]", "", t)
    t = t.replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي")
    return re.sub(r"\s+", " ", t).strip()[:200]

TFIDF_PATH = "output/final_evaluation/tfidf_corpus.pkl"
MATRIX_PATH = "output/final_evaluation/tfidf_matrix.npz"
OUTPUT_PATH = "output/final_evaluation/fast_aligned_results.json"

# ── Load cached TF-IDF (already built from earlier run) ──
print("Loading cached TF-IDF...")
t_start = time.time()
with open(TFIDF_PATH, "rb") as f: vec = pickle.load(f)
tfidf = load_npz(MATRIX_PATH)
print(f"Loaded: {tfidf.shape} in {time.time()-t_start:.0f}s")

# ── Load queries ──
print("\nLoading HF queries...")
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

print(f"Queries: {len(queries):,}")

# ── Batch query (small batches, explicit cleanup) ──
print(f"\nBatched querying (25/batch, resuming from 13500)...")
BATCH = 25
matched = {"Sahih":[], "Hasan":[], "Daif":[], "Matruk":[]}
t0 = time.time()

for start in range(13500, len(queries), BATCH):
    end = min(start + BATCH, len(queries))
    texts = queries[start:end]
    batch_vec = vec.transform(texts)
    sims = cosine_similarity(batch_vec, tfidf)

    for i in range(end - start):
        row = sims[i]
        best_idx = np.argmax(row)
        if row[best_idx] >= 0.5:
            qi = start + i
            matched[q_grades[qi]].append({
                "corpus_idx": int(best_idx),
                "grade": q_grades[qi],
                "book": q_books[qi],
                "similarity": float(row[best_idx]),
            })

    del sims, batch_vec, texts
    if start % 500 == 0 and start > 0:
        gc.collect()
        elapsed = time.time() - t0
        rate = start / max(elapsed, 1)
        total_matched = sum(len(v) for v in matched.values())
        eta = (len(queries) - start) / max(rate, 1)
        print(f"  {start}/{len(queries)} ({rate:.0f}/s, {total_matched:,} matched, ETA {eta:.0f}s)")

tt = time.time() - t0
print(f"\nDone: {sum(len(v) for v in matched.values()):,} matched in {tt:.0f}s")
for g in ["Sahih","Hasan","Daif","Matruk"]:
    print(f"  {g}: {len(matched[g]):,}")

# ── Sample 200 per book ──
print("\nSampling 200 per book...")
from collections import defaultdict
by_book = defaultdict(list)
for g in matched:
    for m in matched[g]:
        by_book[m["book"].lower()].append(m)

canon = {"bukhari":"Bukhari","muslim":"Muslim","dawud":"Abu Dawud",
         "tirmidhi":"Tirmidhi","nasai":"Nasai","majah":"Ibn Majah"}

rng = np.random.RandomState(42)
final = []
book_counts = {}
for fname, short in canon.items():
    pool = by_book.get(fname, [])
    if not pool:
        pool = [m for k, v in by_book.items() if fname[:4] in k for m in v]
    n = min(200, len(pool))
    chosen = rng.choice(pool, n, replace=False).tolist() if n < len(pool) else pool
    for m in chosen:
        m["canonical_book"] = short
    final.extend(chosen)
    book_counts[short] = n
    print(f"  {short:15s}: {n}/{len(pool)}")

# Save
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump({"metadata": {"total": len(final), "time_seconds": tt},
               "results": final}, f, indent=2)
print(f"\nSaved: {OUTPUT_PATH} ({len(final)} hadiths)")
