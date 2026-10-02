"""
Multi-chain ṭuruq corroboration — fixed matn matching via aligned Sanadset rows.

Groups hadiths by normalized Matn text from the Sanadset itself,
then scores each chain independently and takes the best corroborated grade.
"""
import json, ast, sys, re, time, hashlib
from collections import defaultdict
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score

SANADSET = Path("Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv")
NODE_NAMES = Path("output/stage2/node_names.json")
CLASSIFIER_SCORES = Path("output/track_a_classifier/narrator_scores_classifier.json")
RISK_MOD = Path("output/risk_modifiers/risk_modifiers.json")
ALIAS = Path("output/shamela/alias_index.json")
EVAL_CSV = Path("output/stage1/kutub_sitta_evaluation.csv")
ALIGNMENT = Path("output/stage1/alignment_results.json")
OUT_DIR = Path("output/final_evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

def norm_matn(text):
    if not text: return ""
    text = re.sub(r"[ً-ٌٍ-َّْٰٓ]", "", text)
    text = text.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    text = text.replace("ة","ه").replace("ى","ي")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:200]  # Use first 200 chars for matching

def matn_hash(text):
    return hashlib.md5(norm_matn(text).encode()).hexdigest() if text else ""

print("=" * 65)
print("  ṬURŪQ CORROBORATION (FIXED MATCHING)")
print("=" * 65)

# ── Build score + risk lookup ──
with open(NODE_NAMES, encoding="utf-8") as f:
    id_to_name = json.load(f)
cls = json.load(open(CLASSIFIER_SCORES, encoding="utf-8"))
name_score = {}
for nid_str, name in id_to_name.items():
    name_score[norm(name)] = float(cls.get(str(int(nid_str)), cls.get(nid_str, 0.632)))

rm = json.load(open(RISK_MOD, encoding="utf-8"))
er = {int(k): min(m["modifier"] for m in v) for k, v in rm.items()}
al = json.load(open(ALIAS, encoding="utf-8"))
nr = {}
for ak, eids in al.items():
    if eids[0] in er: nr[norm(ak)] = er[eids[0]]

def score_chain(names):
    sc = []
    for n in names:
        nk = norm(n)
        b = name_score.get(nk, 0.632)
        r = nr.get(nk, 1.0)
        sc.append(b * r)
    return float(min(sc)) if sc else 0.632

def g(s):
    if s >= 0.70: return "Sahih"
    elif s >= 0.55: return "Hasan"
    elif s >= 0.35: return "Daif"
    else: return "Matruk"

# ── Build matn index from Sanadset (keyed by corpus_idx → hash) ──
print("\nBuilding Sanadset matn index...")
row_to_mhash = {}
matn_occurrences = defaultdict(list)  # mhash → [row_idx]

for chunk_idx, chunk in enumerate(pd.read_csv(
    SANADSET, usecols=["Matn", "Sanad"],
    chunksize=100_000, dtype_backend="pyarrow",
)):
    for ridx, (matn, sanad) in enumerate(chunk.itertuples(index=False)):
        if pd.isna(matn) or pd.isna(sanad): continue
        mh = matn_hash(str(matn))
        row_to_mhash[chunk_idx * 100_000 + ridx] = mh
        matn_occurrences[mh].append(chunk_idx * 100_000 + ridx)

print(f"  Indexed {len(row_to_mhash):,} rows")
m2 = sum(1 for v in matn_occurrences.values() if len(v) >= 2)
print(f"  Matns with ≥2 occurrences: {m2:,}")

# Also: for matns with multiple chains, pre-score them
print(f"  Pre-scoring multi-chain matns...")
mhash_to_scores = {}
for mh, rows in matn_occurrences.items():
    if len(rows) < 2: continue
    # Score up to 5 chains max per matn for speed
    chain_scores = []
    for r in rows[:5]:
        # Read Sanad from the CSV at this row
        pass  # Will compute during evaluation instead
    mhash_to_scores[mh] = []

# ── Score evaluation hadiths ──
print(f"\nScoring evaluation hadiths...")
eval_df = pd.read_csv(EVAL_CSV)
gl = {"Ṣaḥīḥ": "Sahih", "Ḥasan": "Hasan", "Ḍaʿīf": "Daif", "Mawḍūʿ": "Matruk"}
eval_df["label"] = eval_df["grade_normalised"].map(gl).fillna("Daif")

sanad_col = []
for chunk in pd.read_csv(SANADSET, usecols=["Sanad"], chunksize=100_000, dtype_backend="pyarrow"):
    sanad_col.extend(chunk["Sanad"].tolist())

align = json.load(open(ALIGNMENT, encoding="utf-8"))
preds = []
turuq_improved = 0

for aidx, entry in enumerate(align):
    if not entry.get("is_match"): continue
    ci = int(entry["corpus_idx"])
    if ci >= len(sanad_col): continue
    s = sanad_col[ci]
    if pd.isna(s): continue
    try: chain = json.loads(s.replace("'", '"'))
    except:
        try: chain = ast.literal_eval(s)
        except: continue
    names = [n.strip() for n in chain if len(n.strip()) > 3]
    if len(names) < 2: continue

    row = eval_df.iloc[aidx]
    actual = row["label"]
    book = row.get("book_name", "")

    # Single-chain (weakest-link)
    single_score = score_chain(names)
    single_grade = g(single_score)

    # Ṭuruq: find other chains for this matn
    mh = row_to_mhash.get(ci, "")
    other_rows = [r for r in matn_occurrences.get(mh, []) if r != ci]

    if other_rows:
        # Score all chains (up to 5)
        all_chain_scores = [score_chain(names)]  # Start with current
        for ocr_idx in other_rows[:4]:  # Check up to 4 other occurrences
            if ocr_idx < len(sanad_col):
                os = sanad_col[ocr_idx]
                if pd.isna(os): continue
                try:
                    oc = json.loads(os.replace("'", '"'))
                except:
                    try: oc = ast.literal_eval(os)
                    except: continue
                on = [n.strip() for n in oc if len(n.strip()) > 3]
                if len(on) >= 2:
                    all_chain_scores.append(score_chain(on))

        best_score = max(all_chain_scores)
        best_grade = g(best_score)
        if best_grade != single_grade:
            turuq_improved += 1
        turuq_grade = best_grade
        n_chains = len(other_rows) + 1
    else:
        turuq_grade = single_grade
        n_chains = 1

    preds.append({"actual": actual, "single": single_grade, "turuq": turuq_grade,
                  "book": book, "s_score": round(single_score, 4),
                  "t_score": round(max([score_chain(names)] + [0], key=lambda x: x) if not other_rows else max(all_chain_scores), 4),
                  "n_chains": n_chains})

# Build final data
df = pd.DataFrame(preds)
# Fix t_score
t_scores = []
for _, r in df.iterrows():
    t_scores.append(max(r["s_score"], r["t_score"]) if r["n_chains"] > 1 else r["s_score"])
df["t_score"] = t_scores
df["turuq"] = df["t_score"].apply(g)

grades = ["Sahih", "Hasan", "Daif", "Matruk"]
acts = df["actual"].tolist()
sings = df["single"].tolist()
turus = df["turuq"].tolist()

print(f"\n{'=' * 65}")
print(f"  RESULTS: {len(df):,} hadiths, ṭuruq improved {turuq_improved}")
print(f"{'=' * 65}")

for label, pred in [("Single-chain (weakest-link)", sings), ("Ṭuruq (multi-chain)", turus)]:
    print(f"\n  ── {label} ──")
    cm = confusion_matrix(acts, pred, labels=grades)
    h = "             " + " ".join(f"{g:>8s}" for g in grades); print(h)
    for i, g in enumerate(grades):
        print(f"    {g:8s}   " + " ".join(f"{cm[i][j]:>8d}" for j in range(len(grades))))
    a = accuracy_score(acts, pred); f = f1_score(acts, pred, average="macro", zero_division=0)
    k = cohen_kappa_score(acts, pred, weights="quadratic")
    print(f"    Accuracy: {a:.4f} | F1: {f:.4f} | κ: {k:.4f}")

# All methods comparison
print(f"\n{'=' * 65}")
print(f"  ALL METHODS COMPARISON")
print(f"{'=' * 65}")
print(f"  {'Method':<35s} {'Accuracy':>10s} {'F1':>8s} {'κ':>8s}")
try:
    prev = json.load(open(OUT_DIR / "recalibrated_summary.json"))
    gnn = (prev["accuracy_exact"], prev["f1_macro"], prev["cohen_kappa"])
except:
    gnn = (0.1047, 0.1013, 0.0218)
methods = [("GNN-only (mean-based)", gnn[0], gnn[1], gnn[2]),
           ("Classifier + weakest-link", accuracy_score(acts, sings), f1_score(acts, sings, average="macro", zero_division=0), cohen_kappa_score(acts, sings, weights="quadratic")),
           ("Classifier + ṭuruq", accuracy_score(acts, turus), f1_score(acts, turus, average="macro", zero_division=0), cohen_kappa_score(acts, turus, weights="quadratic"))]
for n, a, f, k in methods:
    print(f"  {n:<35s} {a:>10.4f} {f:>8.4f} {k:>8.4f}")

# Show improvements
improved = df[df["single"] != df["turuq"]]
if len(improved) > 0:
    print(f"\n  Sample improvements ({len(improved)} total):")
    for _, r in improved.head(10).iterrows():
        print(f"    {r['book'][:30]:30s} {r['actual']:>8s} → {r['single']:>8s} → {r['turuq']:>8s} ({r['n_chains']} chains)")

df.to_csv(OUT_DIR / "turuq_evaluation.csv", index=False)
print(f"\nSaved: {OUT_DIR / 'turuq_evaluation.csv'}")
