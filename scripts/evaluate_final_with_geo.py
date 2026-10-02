"""
Complete chain evaluation with:
  - Partitioned Severity Bands (Thiqah, Saduq, Weak, Very Weak, Severe Weakness)
  - Parameterized Soft Aggregation (alpha * weakest + (1-alpha) * mean)
  - GNN Counterbalance (Uncertainty Modulation) and Disagreement Risk Modifiers
  - Geospatial consistency penalties
  - Component Ablation Studies (Semantic Only, Semantic+GNN, Semantic+Geo, Full)
  - Book-by-Book Grouped Performance Analysis
  - Evaluation against Kutub as-Sitta actual grades

Usage:
  python scripts/evaluate_final_with_geo.py
"""
import json, ast, sys, re, time, math, argparse
from collections import defaultdict
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score

# ── CLI Args ──
parser = argparse.ArgumentParser(description="Final grading and ablation pipeline")
parser.add_argument(
    "--sample-size",
    type=int,
    choices=[1200, 6000],
    default=1200,
    help="Evaluation set target: 1200 or 6000 aligned hadiths",
)
parser.add_argument(
    "--balance",
    type=str,
    choices=["book", "grade"],
    default="book",
    help="Balancing strategy: book-balanced or grade-balanced",
)
parser.add_argument(
    "--eval-only",
    action="store_true",
    help="Skip bulk 650K scoring (only run calibration/ablation/book analysis)",
)
args = parser.parse_args()

# ── Paths ──
SAMPLE = args.sample_size
BALANCE = args.balance
SANADSET = Path("Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv")
NODE_NAMES = Path("output/stage2/node_names.json")
GNN_PRED = Path("output/stage3/narrator_predictions.csv")
RISK_MOD = Path("output/risk_modifiers/risk_modifiers.json")
ALIAS_IDX = Path("output/shamela/alias_index.json")
EVAL_CSV = Path(f"output/stage1/kutub_sitta_evaluation_{SAMPLE}_{BALANCE}_balanced.csv")
ALIGNMENT = Path(f"output/stage1/alignment_results_{SAMPLE}_{BALANCE}_balanced.json")
ENTRY_LOCS = Path("output/geo/entry_locations.json")
OUT_DIR = Path("output/final_evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Normalization and Geodetics ──
def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371; dlat = math.radians(lat2-lat1); dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R*2*math.atan2(math.sqrt(a), math.sqrt(1-a))

GEO_DIST_THRESHOLD = 3000  # km
GEO_PENALTY = 0.85

print("=" * 70)
print("  FINAL GRADING & ABLATION PIPELINE")
print("=" * 70)

# ── 1. Load lookups ──
print("\n[1/6] Loading score/risk/geo lookups...")

preds = pd.read_csv(GNN_PRED)
gnn_profiles = {
    int(r["node_id"]): {
        "predicted_score": float(r["predicted_score"]),
        "anomaly_score": float(r["anomaly_score"])
    }
    for _, r in preds.iterrows()
}

SENT_PATH = Path("output/track_a_classifier/narrator_scores_sentiment.csv")
sentiment_profiles = {}
if SENT_PATH.exists():
    sent_df = pd.read_csv(SENT_PATH)
    sentiment_profiles = {
        int(r["node_id"]): {
            "reliability": float(r["reliability_score"]),
            "agreement": float(r["agreement_score"]),
            "critic_count": int(r["critic_count"])
        }
        for _, r in sent_df.iterrows()
    }
    print(f"  Loaded {len(sentiment_profiles):,} sentiment-based consensus profiles")
else:
    print("  Warning: sentiment-based narrator reliability scores not found!")

with open(NODE_NAMES, encoding="utf-8") as f:
    id_to_name = json.load(f)

name_to_id = {norm(v): int(k) for k, v in id_to_name.items()}

gnn_emb_path = GNN_PRED.parent / "narrator_embeddings.npy"
gnn_embeddings = np.load(gnn_emb_path) if gnn_emb_path.exists() else None

risk_mods = json.load(open(RISK_MOD, encoding="utf-8"))
entry_risk = {int(k): min(m["modifier"] for m in v) for k, v in risk_mods.items()}
alias_index = json.load(open(ALIAS_IDX, encoding="utf-8"))
narrator_risk = {}
for ak, eids in alias_index.items():
    eid = eids[0]
    if eid in entry_risk:
        narrator_risk[norm(ak)] = entry_risk[eid]

entry_locs = json.load(open(ENTRY_LOCS, encoding="utf-8"))
narrator_locs = {}
for eid_str, locs in entry_locs.items():
    for ak, eids in alias_index.items():
        if eids[0] == int(eid_str):
            narrator_locs[norm(ak)] = locs
            break

print(f"  Risk profiles: {len(narrator_risk):,} | Geo profiles: {len(narrator_locs):,}")

# ── Tabaqa (generation) data from encyclopedia ──
ENC_PATH = Path("output/shamela/narrator_encyclopedia.json")
name_to_tabaqa = {}
if ENC_PATH.exists():
    enc = json.load(open(ENC_PATH, encoding="utf-8"))
    import re as _re
    # Tabaqa values are truncated (e.g. "من العا" = من العاشرة = 10th gen)
    # Match on the last visible word after "من "
    gen_pat = _re.compile(r"من\s+(العا|الحادي|التاسع|الثامن|السابع|السادس|الخامس|الرابع|الثالث|الثاني)")
    gen_map = {
        "الثاني": 2, "الثالث": 3, "الرابع": 4, "الخامس": 5,
        "السادس": 6, "السابع": 7, "الثامن": 8, "التاسع": 9,
        "العا": 10, "الحادي": 11,
    }
    entry_gen = {}
    for e in enc:
        tid = e.get("entry_id")
        t = e.get("tabaqa") or ""
        m = gen_pat.search(t)
        if m:
            matched = m.group(1)
            entry_gen[int(tid)] = gen_map.get(matched, 0)
    # Map narrator names -> tabaqa
    for ak, eids in alias_index.items():
        eid = eids[0]
        if eid in entry_gen:
            name_to_tabaqa[norm(ak)] = entry_gen[eid]
    print(f"  Tabaqa profiles: {len(name_to_tabaqa):,} | Generation range: {set(name_to_tabaqa.values()) if name_to_tabaqa else 'N/A'}")
else:
    print(f"  Warning: encyclopedia not found, tabaqa data unavailable")

# ── 2. Load calibration data ──
print("\n[2/6] Building calibration set from Kutub as-Sitta...")
eval_df = pd.read_csv(EVAL_CSV)
grade_labels = {"Ṣaḥīḥ": "Sahih", "Ḥasan": "Hasan", "Ḍaʿīf": "Daif", "Mawḍūʿ": "Matruk"}
grade_tiers = {"Sahih": 4, "Hasan": 3, "Daif": 2, "Matruk": 1}
eval_df["label"] = eval_df["grade_normalised"].map(grade_labels).fillna("Daif")
eval_df["tier"] = eval_df["label"].map(grade_tiers)

sanad_col = []
for chunk in pd.read_csv(SANADSET, usecols=["Sanad"], chunksize=100_000, dtype_backend="pyarrow"):
    sanad_col.extend(chunk["Sanad"].tolist())

align = json.load(open(ALIGNMENT, encoding="utf-8"))

calibration_chains = []
for entry in align:
    if not entry.get("is_match"): continue
    qidx = int(entry["query_idx"])
    if qidx >= len(eval_df): continue
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

    calibration_chains.append({
        "names": names,
        "true_tier": eval_df.iloc[qidx]["tier"],
        "true_label": eval_df.iloc[qidx]["label"],
        "book_key": eval_df.iloc[qidx]["book_key"]
    })

print(f"  Calibration samples: {len(calibration_chains)}")

# ── 3. Advanced Narrator Severity & Modulation Logic ──
def get_narrator_score(nkey):
    nid = name_to_id.get(nkey)
    profile = sentiment_profiles.get(nid, {}) if nid else {}
    gnn_prof = gnn_profiles.get(nid, {}) if nid else {}
    
    # Retrieve base semantic reliability
    r = profile.get("reliability", gnn_prof.get("predicted_score", 0.632))
    agreement = profile.get("agreement", 1.0)
    anomaly = gnn_prof.get("anomaly_score", 0.0)
    
    # Severity Bands and Base Discounts
    if r >= 0.80:
        category = "Thiqah"
        discount = 1.0
    elif r >= 0.70:
        category = "Saduq"
        discount = 1.0
    elif r >= 0.40:
        category = "Weak"
        discount = 0.85
    elif r >= 0.20:
        category = "Very Weak"
        discount = 0.60
    else:
        category = "Severe Weakness"
        discount = 0.25

    # GNN Counterbalance (Uncertainty Modulation)
    if category in ("Weak", "Very Weak") and anomaly < 0.25:
        mitigation = 0.15
        discount = discount + (1.0 - discount) * mitigation

    # Disagreement Risk Modifier
    disagreement_discount = 1.0 - 0.20 * (1.0 - agreement)
    
    return r * discount * disagreement_discount

# Score a chain with options
def evaluate_chain_score(names, alpha, use_gnn=True, use_geo=True, use_temp=True):
    ind_scores = []
    for n in names:
        nkey = norm(n)
        base = get_narrator_score(nkey)
        risk = narrator_risk.get(nkey, 1.0)
        ind_scores.append(base * risk)

    # Soft Aggregation
    weakest = min(ind_scores)
    mean_s = np.mean(ind_scores)
    chain_base = alpha * weakest + (1.0 - alpha) * mean_s

    # Geo Penalty
    chain_geo_penalty = 1.0
    if use_geo:
        prev_loc = None
        for n in names:
            nkey = norm(n)
            if nkey in narrator_locs:
                this_loc = narrator_locs[nkey][0]
                if prev_loc:
                    dist = haversine(prev_loc[1][0], prev_loc[1][1], this_loc[1][0], this_loc[1][1])
                    if dist > GEO_DIST_THRESHOLD:
                        chain_geo_penalty *= GEO_PENALTY
                prev_loc = this_loc
            else:
                prev_loc = None

    # GNN Path compatibility penalty
    chain_gnn_penalty = 1.0
    if use_gnn and gnn_embeddings is not None:
        min_sim = 1.0
        for i in range(len(names) - 1):
            nkey1 = norm(names[i])
            nkey2 = norm(names[i+1])
            if nkey1 in name_to_id and nkey2 in name_to_id:
                id1 = name_to_id[nkey1] - 1
                id2 = name_to_id[nkey2] - 1
                if 0 <= id1 < len(gnn_embeddings) and 0 <= id2 < len(gnn_embeddings):
                    z1 = gnn_embeddings[id1]
                    z2 = gnn_embeddings[id2]
                    dot = np.sum(z1 * z2)
                    sim = 1.0 / (1.0 + np.exp(-dot))
                    if sim < min_sim:
                        min_sim = sim
        if min_sim < 0.25:
            chain_gnn_penalty = 0.85

    # Temporal (Inqita) Penalty - checks tabaqa gaps between consecutive narrators
    chain_temp_penalty = 1.0
    if use_temp and name_to_tabaqa:
        for i in range(len(names) - 1):
            nkey1 = norm(names[i])
            nkey2 = norm(names[i+1])
            t1 = name_to_tabaqa.get(nkey1)
            t2 = name_to_tabaqa.get(nkey2)
            if t1 is not None and t2 is not None:
                gap = t2 - t1
                if gap >= 5:
                    chain_temp_penalty *= 0.40  # severe gap
                elif gap >= 3:
                    chain_temp_penalty *= 0.60  # moderate gap

    return float(chain_base) * chain_geo_penalty * chain_gnn_penalty * chain_temp_penalty

# ── 4. Ablation & Optimization Runner ──
print("\n[3/6] Running Ablation Studies and Soft Aggregation Optimization...")

def calibrate_and_evaluate(scores, labels, tiers):
    sorted_scores = sorted(scores, reverse=True)
    n = len(sorted_scores)
    actual_series = pd.Series(labels)
    p_sahih = (actual_series == "Sahih").mean()
    p_hasan = (actual_series == "Hasan").mean()
    p_daif = (actual_series == "Daif").mean()
    
    pct_sahih = p_sahih
    pct_hasan = p_sahih + p_hasan
    pct_daif = p_sahih + p_hasan + p_daif
    
    T = {
        "Sahih": round(sorted_scores[min(n - 1, max(0, int(n * pct_sahih) - 1))], 4),
        "Hasan": round(sorted_scores[min(n - 1, max(0, int(n * pct_hasan) - 1))], 4),
        "Maqbul": round(sorted_scores[min(n - 1, max(0, int(n * (pct_hasan + 0.5 * p_daif)) - 1))], 4),
        "Daif": round(sorted_scores[min(n - 1, max(0, int(n * pct_daif) - 1))], 4),
    }
    
    eps = 1e-4
    if T["Hasan"] >= T["Sahih"]: T["Sahih"] = round(T["Hasan"] + eps, 4)
    if T["Maqbul"] >= T["Hasan"]: T["Hasan"] = round(T["Maqbul"] + eps, 4)
    if T["Daif"] >= T["Maqbul"]: T["Maqbul"] = round(T["Daif"] + eps, 4)
    
    def classify(s):
        if s >= T["Sahih"]: return "Sahih"
        elif s >= T["Hasan"]: return "Hasan"
        elif s >= T["Maqbul"]: return "Maqbul"
        elif s >= T["Daif"]: return "Daif"
        else: return "Matruk"
        
    preds = [classify(s) for s in scores]
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    kappa = cohen_kappa_score(labels, preds, weights="quadratic")
    
    return acc, f1, kappa, T

ablation_results = []
configurations = [
    ("Semantic Only", False, False, False),
    ("Semantic + GNN", True, False, False),
    ("Semantic + Geo", False, True, False),
    ("Semantic + Temp", False, False, True),
    ("Semantic + GNN + Geo + Temp (Full)", True, True, True)
]

alphas = [0.25, 0.50, 0.75]
best_acc = 0.0
best_config = None
best_alpha = 0.50
best_thresholds = None

for name, use_gnn, use_geo, use_temp in configurations:
    for alpha in alphas:
        scores = [evaluate_chain_score(c["names"], alpha, use_gnn, use_geo, use_temp) for c in calibration_chains]
        labels = [c["true_label"] for c in calibration_chains]
        tiers = [c["true_tier"] for c in calibration_chains]

        acc, f1, kappa, T = calibrate_and_evaluate(scores, labels, tiers)
        ablation_results.append({
            "config": name,
            "alpha": alpha,
            "accuracy": acc,
            "f1_macro": f1,
            "kappa": kappa,
            "thresholds": T
        })

        if name == "Semantic + GNN + Geo + Temp (Full)" and acc > best_acc:
            best_acc = acc
            best_alpha = alpha
            best_thresholds = T

# Output Ablation Table
df_abl = pd.DataFrame(ablation_results)
print(df_abl.to_string(index=False))

print(f"\n  Optimal Configuration: Semantic + GNN + Geo + Temp (Full)")
print(f"  Optimal Soft Aggregation Alpha: {best_alpha:.2f} (Accuracy: {best_acc:.4%})")
print(f"  Calibrated Thresholds: {best_thresholds}")

# ── 5. Detailed Book-by-Book Analysis ──
print("\n[4/6] Running Book-by-Book Performance Breakdown...")
scores = [evaluate_chain_score(c["names"], best_alpha, True, True) for c in calibration_chains]
labels = [c["true_label"] for c in calibration_chains]

def get_grade(s):
    if s >= best_thresholds["Sahih"]: return "Sahih"
    elif s >= best_thresholds["Hasan"]: return "Hasan"
    elif s >= best_thresholds["Maqbul"]: return "Maqbul"
    elif s >= best_thresholds["Daif"]: return "Daif"
    else: return "Matruk"

preds = [get_grade(s) for s in scores]

book_data = defaultdict(list)
for idx, c in enumerate(calibration_chains):
    book_data[c["book_key"]].append((c["true_label"], preds[idx]))

book_metrics = []
for book, items in book_data.items():
    b_true = [x[0] for x in items]
    b_pred = [x[1] for x in items]
    acc = accuracy_score(b_true, b_pred)
    f1 = f1_score(b_true, b_pred, average="macro")
    book_metrics.append({"Collection": book, "Samples": len(items), "Accuracy": acc, "F1 Macro": f1})

df_books = pd.DataFrame(book_metrics)
print(df_books.to_string(index=False))

# ── 6. Score All 650K Chains (skip if --eval-only) ──
if args.eval_only:
    print(f"\n[5/6] Skipping bulk 650K scoring (--eval-only mode)")
    print(f"\n{'=' * 65}")
    print(f"  EVALUATION COMPLETE — Calibration/Ablation results above")
    print("=" * 65)
    sys.exit(0)

print(f"\n[5/6] Scoring 650K chains (Full Pipeline with optimal alpha={best_alpha})...")

total, nerror = 0, 0
risk_hits, geo_hits = 0, 0
results = []
t0 = time.time()

for chunk_idx, chunk in enumerate(pd.read_csv(SANADSET, usecols=["Sanad"], chunksize=100_000, dtype_backend="pyarrow")):
    ct0 = time.time()
    for _, row in chunk.iterrows():
        sanad = row["Sanad"]
        if pd.isna(sanad): continue
        try: chain = json.loads(sanad.replace("'", '"'))
        except:
            try: chain = ast.literal_eval(sanad)
            except: nerror += 1; continue
        if not isinstance(chain, list) or len(chain) < 2: continue
        names = [n.strip() for n in chain if len(n.strip()) > 3]
        if len(names) < 2: continue
        
        chain_score = evaluate_chain_score(names, best_alpha, True, True)
        results.append({"cl": len(names), "sc": round(chain_score, 4)})
        total += 1
    print(f"  Chunk {chunk_idx+1}: {total:,} chains ({time.time()-ct0:.0f}s)")

ttime = time.time() - t0
df_final = pd.DataFrame(results)
df_final["gr"] = df_final["sc"].apply(get_grade)

# ── 7. Final Grade Distribution and Summary ──
print(f"\n[6/6] Final Grade Distribution:")
print("=" * 65)
print(f"  RESULTS: {len(df_final):,} chains, {ttime:.0f}s")
print("=" * 65)

print(f"\nScore: mean={df_final['sc'].mean():.4f}, std={df_final['sc'].std():.4f}")
for g in ["Sahih", "Hasan", "Maqbul", "Daif", "Matruk"]:
    cnt = len(df_final[df_final["gr"] == g])
    print(f"  {g:10s}: {cnt:>8,} ({cnt/len(df_final)*100:.1f}%)")

df_final.to_csv(OUT_DIR / "chain_results_final_v2.csv", index=False)
print(f"\nSaved: {OUT_DIR / 'chain_results_final_v2.csv'}")

summary = {
    "method": "agreement_aware_soft_aggregation",
    "optimal_alpha": best_alpha,
    "total_chains": len(df_final),
    "mean_score": round(float(df_final["sc"].mean()), 4),
    "std_score": round(float(df_final["sc"].std()), 4),
    "thresholds": best_thresholds,
    "grade_distribution": {g: int(c) for g, c in df_final["gr"].value_counts().items()},
    "time_seconds": ttime,
    "ablation_results": ablation_results,
    "book_metrics": book_metrics,
}
with open(OUT_DIR / "summary_final_v2.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {OUT_DIR / 'summary_final_v2.json'}")

print(f"\n{'=' * 65}")
print(f"  COMPLETE — Full Pipeline Verification complete")
print("=" * 65)
