"""
650K Experiments: Book rankings, weakest narrators, chain length analysis.
Runs all three in one pass.
"""
import json, re, ast, sys, time
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

GRADE_MAP = {"صحابي":1.0,"ثقة":0.85,"صدوق":0.70,"مقبول":0.40,"ضعيف":0.15,"متروك":0.05,"كذاب":0.0}
def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

# ── Build scores ──
print("Building narrator scores...")
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
print(f"  {len(name_score):,} narrators scored")

# ── Process 650K chains ──
print("\nProcessing 650K chains...")
t0 = time.time()

BOOKS = {}      # book → {n, scores, grades}
WEAKEST = Counter()  # narrator_name → times as weakest link
CHAIN_LENGTHS = []   # (length, score)

total = 0
for chunk in pd.read_csv(
    "Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv",
    usecols=["Sanad", "Book", "Matn"], chunksize=100_000, dtype_backend="pyarrow",
):
    for _, row in chunk.iterrows():
        s = row["Sanad"]
        if pd.isna(s): continue
        try: ch = json.loads(s.replace("'", '"'))
        except:
            try: ch = ast.literal_eval(s)
            except: continue
        names = [n.strip() for n in ch if len(n.strip()) > 3]
        if len(names) < 2: continue

        # Score chain
        sc = sorted([name_score.get(norm(n), 0.632) for n in names])
        cs = (sc[0] + sc[1]) / 2 if len(sc) >= 2 else sc[0]

        # A: Per-book
        book = str(row.get("Book", "Unknown"))
        if book not in BOOKS:
            BOOKS[book] = {"n":0, "scores":[], "grades":defaultdict(int)}
        BOOKS[book]["n"] += 1
        BOOKS[book]["scores"].append(cs)
        g = "Sahih" if cs >= 0.45 else "Hasan" if cs >= 0.425 else "Daif" if cs >= 0.15 else "Matruk"
        BOOKS[book]["grades"][g] += 1

        # B: Weakest narrator
        min_score_names = [name for name in names if name_score.get(norm(name), 0.632) == sc[0]]
        for wn in min_score_names:
            WEAKEST[wn] += 1

        # C: Chain length
        CHAIN_LENGTHS.append((len(names), cs))

        total += 1

ttime = time.time() - t0
print(f"  Processed {total:,} chains in {ttime:.0f}s")

# ── A: Per-Book Reliability Ranking ──
print(f"\n{'=' * 65}")
print(f"  A: PER-BOOK RELIABILITY RANKING")
print(f"  (Books with ≥100 hadiths)")
print(f"{'=' * 65}")
print(f"{'Book':55s} {'n':>6s} {'Mean':>6s} {'Std':>5s} {'%Sahih':>7s} {'%Daif':>6s}")
print(f"{'─' * 90}")

sorted_books = sorted([(b, d) for b, d in BOOKS.items() if d["n"] >= 100],
                      key=lambda x: np.mean(x[1]["scores"]), reverse=True)

for book, data in sorted_books[:20]:
    scores = data["scores"]
    mean_s = np.mean(scores)
    std_s = np.std(scores)
    pct_sahih = data["grades"]["Sahih"] / data["n"] * 100
    pct_daif = (data["grades"]["Daif"] + data["grades"]["Matruk"]) / data["n"] * 100
    book_short = book[:53] if isinstance(book, str) else str(book)[:53]
    print(f"  {book_short:53s} {data['n']:>6d} {mean_s:>6.3f} {std_s:>5.3f} {pct_sahih:>6.1f}% {pct_daif:>5.1f}%")

# Bottom 5
print("\n  Bottom 5 books:")
for book, data in sorted_books[-5:]:
    mean_s = np.mean(data["scores"])
    pct_daif = (data["grades"]["Daif"] + data["grades"]["Matruk"]) / data["n"] * 100
    book_short = book[:53] if isinstance(book, str) else str(book)[:53]
    print(f"  {book_short:53s} {data['n']:>6d} mean={mean_s:.3f} Daif={pct_daif:.1f}%")

# ── B: Weakest Narrator Frequency ──
print(f"\n{'=' * 65}")
print(f"  B: WEAKEST NARRATOR FREQUENCY")
print(f"  (Narrators most often the weakest link)")
print(f"{'=' * 65}")
print(f"{'Narrator':55s} {'Times Weakest':>14s}")
print(f"{'─' * 70}")

for name, count in WEAKEST.most_common(25):
    nk = norm(name)
    score = name_score.get(nk, 0.632)
    name_short = name[:53] if isinstance(name, str) else str(name)[:53]
    print(f"  {name_short:53s} {count:>8d} (score={score:.3f})")

# ── C: Chain Length vs Reliability ──
print(f"\n{'=' * 65}")
print(f"  C: CHAIN LENGTH vs RELIABILITY")
print(f"{'=' * 65}")

length_buckets = defaultdict(list)
for length, score in CHAIN_LENGTHS:
    if 2 <= length <= 20:
        length_buckets[length].append(score)

print(f"{'Length':>8s} {'n':>8s} {'Mean':>6s} {'Std':>5s} {'%≥0.45':>8s}")
print(f"{'─' * 40}")
for length in sorted(length_buckets.keys()):
    scores = length_buckets[length]
    mean_s = np.mean(scores)
    std_s = np.std(scores)
    pct_strong = sum(1 for s in scores if s >= 0.45) / len(scores) * 100
    print(f"  {length:>4d} chains {len(scores):>8,d} {mean_s:>6.3f} {std_s:>5.3f} {pct_strong:>7.1f}%")

# ── Save results ──
print(f"\nSaving results...")
out = {"book_ranking": {b: {"n":d["n"], "mean_score":float(np.mean(d["scores"])),
                            "std_score":float(np.std(d["scores"])),
                            "pct_sahih":d["grades"]["Sahih"]/d["n"]*100,
                            "pct_daif":(d["grades"]["Daif"]+d["grades"]["Matruk"])/d["n"]*100}
                       for b, d in sorted_books},
       "weakest_narrators": {name: count for name, count in WEAKEST.most_common(100)},
       "chain_length": {l: {"n":len(scores), "mean":float(np.mean(scores)),
                            "std":float(np.std(scores)),
                            "pct_strong":sum(1 for s in scores if s>=0.45)/len(scores)*100}
                       for l, scores in length_buckets.items()}}
with open("output/final_evaluation/experiments_650k.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved: output/final_evaluation/experiments_650k.json")
print(f"\nDone — {total:,} chains analyzed in {ttime:.0f}s")
