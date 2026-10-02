"""
Track B at scale: use GRADE_SCORE_MAP via alias index for all 262K KG narrators.
Then: weakest-link + Companion rule + turuq → evaluate vs Kutub as-Sitta.

This bypasses the classifier entirely for chain evaluation and uses direct
grade terms from the encyclopedia (which are more reliable for well-known narrators).
"""
import json, ast, re, hashlib, sys
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score

# ── Config ──
SANADSET = "Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv"
NODE_NAMES = "output/stage2/node_names.json"
GNN_PRED = "output/stage3/narrator_predictions.csv"
ENC = "output/shamela/narrator_encyclopedia.json"
ALIAS = "output/shamela/alias_index.json"
EVAL_CSV = "output/stage1/kutub_sitta_evaluation.csv"
ALIGNMENT = "output/stage1/alignment_results.json"
OUT_DIR = "output/final_evaluation"

GRADE_SCORE_MAP = {
    "صحابي": 1.0, "ثقة ثبت": 0.95, "ثقة حافظ": 0.95, "ثقة متقن": 0.95, "إمام": 0.95,
    "حافظ": 0.90, "ثبت": 0.90, "متقن": 0.90, "ثقة": 0.85, "ثقات": 0.85, "مأمون": 0.85,
    "صدوق": 0.70, "لا بأس به": 0.65, "ليس به بأس": 0.65, "محله الصدق": 0.65,
    "صدوق يهم": 0.55, "صدوق سيء الحفظ": 0.50, "سيء الحفظ": 0.40, "مقبول": 0.40,
    "مستور": 0.25, "مجهول الحال": 0.20, "ضعيف": 0.15, "ضعيف الحديث": 0.15,
    "لين": 0.15, "لين الحديث": 0.15, "ليس بالقوي": 0.15, "فيه ضعف": 0.15,
    "منكر الحديث": 0.10, "مجهول": 0.10, "مجهول العين": 0.05,
    "متروك": 0.05, "متروك الحديث": 0.05, "ساقط": 0.05, "مختلط": 0.05,
    "متهم": 0.0, "كذاب": 0.0, "وضاع": 0.0, "دجال": 0.0,
}

def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

def map_grade(grade_text):
    if not grade_text: return None
    for term in sorted(GRADE_SCORE_MAP, key=len, reverse=True):
        if term in grade_text: return GRADE_SCORE_MAP[term]
    return None

def grade_from_score(s):
    if s >= 0.70: return "Sahih"
    elif s >= 0.55: return "Hasan"
    elif s >= 0.35: return "Daif"
    else: return "Matruk"

print("=" * 65)
print("  TRACK B AT SCALE: GRADE_SCORE_MAP via Alias Index")
print("=" * 65)

# ── Step 1: Build entry_id → GRADE_SCORE_MAP score ──
print("\n[1] Building entry score lookup from encyclopedia grade terms...")
enc = json.load(open(ENC, encoding="utf-8"))
entry_score = {}
entry_grade_terms = {}
for e in enc:
    eid = e["entry_id"]
    grades = e.get("grades", [])
    scores = [map_grade(g) for g in grades]
    scores = [s for s in scores if s is not None]
    if scores:
        entry_score[eid] = float(np.mean(scores))
        entry_grade_terms[eid] = grades

# Companion rule: ṣaḥābah → 1.0
for e in enc:
    text = e.get("full_text", "") + e.get("name", "")
    tnorm = norm(text)
    if "صحاب" in tnorm:
        entry_score[e["entry_id"]] = 1.0

print(f"  Encyclopedia entries: {len(enc):,}")
print(f"  With GRADE_SCORE_MAP scores: {len(entry_score):,}")

# ── Step 2: Extend to all KG narrators via alias index ──
print("\n[2] Mapping KG narrators via alias index...")
alias = json.load(open(ALIAS, encoding="utf-8"))
with open(NODE_NAMES, encoding="utf-8") as f:
    id_to_name = json.load(f)

# Build: alias_key → average score from all matched entries
# For multi-word aliases, take the highest-score entry (best match)
alias_to_score = {}
for ak, eids in alias.items():
    scores = [entry_score.get(eid) for eid in eids[:3] if eid in entry_score]
    scores = [s for s in scores if s is not None]
    if scores:
        alias_to_score[norm(ak)] = float(np.mean(scores))

# Map each KG narrator
gnn = {int(r["node_id"]): float(r["predicted_score"]) for _, r in pd.read_csv(GNN_PRED).iterrows()}
name_score = {}
kb_hits, gnn_hits = 0, 0

for nid_str, name in id_to_name.items():
    nkey = norm(name)
    nid = int(nid_str)
    if nkey in alias_to_score:
        name_score[nkey] = alias_to_score[nkey]
        kb_hits += 1
    else:
        name_score[nkey] = gnn.get(nid, 0.632)
        gnn_hits += 1

print(f"  KG narrators covered by GRADE_SCORE_MAP: {kb_hits:,} ({kb_hits/len(name_score)*100:.1f}%)")
print(f"  GNN fallback: {gnn_hits:,} ({gnn_hits/len(name_score)*100:.1f}%)")

# Score distribution
vals = list(name_score.values())
print(f"  Score range: [{min(vals):.4f}, {max(vals):.4f}], mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")

# ── Step 3: Build matn → multiple chains index (for ṭuruq) ──
print("\n[3] Building multi-chain index...")
def norm_matn(text):
    if not text: return ""
    text = re.sub(r"[ً-ٌٍ-َّْٰٓ]", "", text)
    text = text.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    text = text.replace("ة","ه").replace("ى","ي")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:200]

matn_to_chains = defaultdict(list)
for chunk in pd.read_csv(SANADSET, usecols=["Matn", "Sanad"], chunksize=100_000, dtype_backend="pyarrow"):
    for _, row in chunk.iterrows():
        m, s = row.get("Matn", ""), row.get("Sanad", "")
        if pd.isna(m) or pd.isna(s): continue
        mh = hashlib.md5(norm_matn(str(m)).encode()).hexdigest()
        try: c = json.loads(s.replace("'", '"'))
        except:
            try: c = ast.literal_eval(s)
            except: continue
        names = tuple(n.strip() for n in c if len(n.strip()) > 3)
        if len(names) >= 2: matn_to_chains[mh].append(list(names))

multi = sum(1 for v in matn_to_chains.values() if len(v) >= 2)
print(f"  Unique matns: {len(matn_to_chains):,}, with ≥2 chains: {multi:,}")

# ── Step 4: Evaluate against Kutub as-Sitta ──
print("\n[4] Evaluating against Kutub as-Sitta...")

def score_chain(names):
    scores = [name_score.get(norm(n), 0.632) for n in names]
    return float(min(scores))

sanad_col = []
for chunk in pd.read_csv(SANADSET, usecols=["Sanad"], chunksize=100_000, dtype_backend="pyarrow"):
    sanad_col.extend(chunk["Sanad"].tolist())

align = json.load(open(ALIGNMENT, encoding="utf-8"))
eval_df = pd.read_csv(EVAL_CSV)
gl = {"Ṣaḥīḥ": "Sahih", "Ḥasan": "Hasan", "Ḍaʿīf": "Daif", "Mawḍūʿ": "Matruk"}
eval_df["label"] = eval_df["grade_normalised"].map(gl).fillna("Daif")

grades = ["Sahih", "Hasan", "Daif", "Matruk"]
results = []

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
    book = str(row.get("book_name", ""))

    # Single-chain score (weakest-link)
    single = score_chain(names)

    # Ṭuruq: check other chains for same matn
    m_text = norm_matn(str(row.get("Arabic_Text", "")))
    mh = hashlib.md5(m_text.encode()).hexdigest() if m_text else ""
    all_chains = matn_to_chains.get(mh, [names])

    # Score all chains, take the best
    chain_scores = [score_chain(c) for c in all_chains[:5]]
    best = max(chain_scores)
    turuq_score = best if best > single and len(all_chains) >= 2 else single

    results.append({
        "actual": actual, "single": grade_from_score(single), "turuq": grade_from_score(turuq_score),
        "book": book, "s_score": round(single, 4), "t_score": round(turuq_score, 4),
        "n_chains": len(all_chains),
    })

df = pd.DataFrame(results)
acts = df["actual"].tolist()
s_list = df["single"].tolist()
t_list = df["turuq"].tolist()

print(f"\n{'=' * 65}")
print(f"  TRACK B AT SCALE — Kutub as-Sitta Results")
print(f"  n = {len(df):,} evaluation hadiths")
print(f"{'=' * 65}")

for label, plist in [("Single-chain (weakest-link)", s_list), ("Ṭuruq corroboration", t_list)]:
    cm = confusion_matrix(acts, plist, labels=grades)
    acc = accuracy_score(acts, plist)
    f1 = f1_score(acts, plist, average="macro", zero_division=0)
    k = cohen_kappa_score(acts, plist, weights="quadratic")
    print(f"\n  ── {label} ──")
    print(f"  {'':>12s} " + " ".join(f"{g:>8s}" for g in grades))
    for i, g in enumerate(grades):
        print(f"  {g:>8s}   " + " ".join(f"{cm[i][j]:>8d}" for j in range(4)))
    print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f} | κ: {k:.4f}")

    # Per-grade recall
    for g in grades:
        mask = [a == g for a in acts]
        if sum(mask) > 0:
            rec = sum(1 for i in range(len(acts)) if acts[i]==g and plist[i]==g) / sum(mask)
            print(f"    {g:10s} recall: {rec:.4f}")

# Summary comparison
print(f"\n{'=' * 65}")
print(f"  ALL METHODS COMPARISON")
print(f"{'=' * 65}")
print(f"  {'Method':<35s} {'Accuracy':>10s} {'F1':>8s} {'κ':>8s}")

methods_data = [
    ("GNN-only (mean-based)", 0.1047, 0.1013, 0.0218),
    ("Classifier + weakest-link", 0.0822, 0.1038, 0.0431),
    ("Track B at scale (single)", accuracy_score(acts, s_list),
     f1_score(acts, s_list, average="macro", zero_division=0),
     cohen_kappa_score(acts, s_list, weights="quadratic")),
    ("Track B + ṭuruq", accuracy_score(acts, t_list),
     f1_score(acts, t_list, average="macro", zero_division=0),
     cohen_kappa_score(acts, t_list, weights="quadratic")),
]

for name, a, f, k in methods_data:
    print(f"  {name:<35s} {a:>10.4f} {f:>8.4f} {k:>8.4f}")

# Show best Sahih recall
print(f"\n  Sahih recall comparison:")
for name, a, f, k in methods_data:
    sahih_recall = sum(1 for i in range(len(acts)) if acts[i]=="Sahih") / max(1, len(acts))
    print(f"    {name:<35s} {sahih_recall:.4f}")

# Save
df.to_csv(f"{OUT_DIR}/track_b_scale_results.csv", index=False)
print(f"\nSaved: {OUT_DIR}/track_b_scale_results.csv")
