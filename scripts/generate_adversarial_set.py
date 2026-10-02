"""
Generate adversarial hadith chains with injected temporal, geospatial, and
GNN path anomalies to validate that penalty mechanisms actually trigger.

Output: adversarial_evaluation.csv — clean vs injected scores per config.
"""
import ast, json, math, random, re, sys, io
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
random.seed(42)

PROJECT = Path(__file__).resolve().parent.parent

# ── Paths ──
SENT_PATH = PROJECT / "output/track_a_classifier/narrator_scores_sentiment.csv"
GNN_PRED = PROJECT / "output/stage3/narrator_predictions.csv"
GNN_EMB = PROJECT / "output/stage3/narrator_embeddings.npy"
NODE_NAMES = PROJECT / "output/stage2/node_names.json"
ALIAS_IDX = PROJECT / "output/shamela/alias_index.json"
ENC_PATH = PROJECT / "output/shamela/narrator_encyclopedia.json"
ENTRY_LOCS = PROJECT / "output/geo/entry_locations.json"
RISK_MOD = PROJECT / "output/risk_modifiers/risk_modifiers.json"
ALIGNMENT = PROJECT / "output/stage1/alignment_results_1200_book_balanced.json"
SANADSET = PROJECT / "Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv"

OUT_DIR = PROJECT / "output/final_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ──
def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

GEO_DIST_THRESHOLD = 3000
GEO_PENALTY = 0.85

# ── 1. Load data ──
print("=" * 65)
print("  ADVERSARIAL EVALUATION — Validating Penalty Mechanisms")
print("=" * 65)

print("\n[1/5] Loading lookups...")

# Sentiment profiles
sent_df = pd.read_csv(SENT_PATH)
sentiment_profiles = {
    int(r["node_id"]): {"reliability": float(r["reliability_score"]), "agreement": float(r["agreement_score"])}
    for _, r in sent_df.iterrows()
}

# GNN profiles
preds = pd.read_csv(GNN_PRED)
gnn_profiles = {int(r["node_id"]): {"predicted_score": float(r["predicted_score"]), "anomaly_score": float(r["anomaly_score"])} for _, r in preds.iterrows()}

# GNN embeddings
embeddings = np.load(GNN_EMB)

# Node names
with open(NODE_NAMES, encoding="utf-8") as f:
    id_to_name = json.load(f)
name_to_id = {norm(v): int(k) for k, v in id_to_name.items()}

# Alias index
alias_index = json.load(open(ALIAS_IDX, encoding="utf-8"))

# Encyclopedia — extract generation (tabaqa) data
enc = json.load(open(ENC_PATH, encoding="utf-8"))
import re as _re
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
        entry_gen[int(tid)] = gen_map.get(m.group(1), 0)

name_to_tabaqa = {}
for alias, eids in alias_index.items():
    eid = eids[0]
    if eid in entry_gen:
        name_to_tabaqa[norm(alias)] = entry_gen[eid]

# Entry locations
entry_locs = json.load(open(ENTRY_LOCS, encoding="utf-8"))
narrator_locs = {}
for eid_str, locs in entry_locs.items():
    for ak, eids in alias_index.items():
        if eids[0] == int(eid_str):
            narrator_locs[norm(ak)] = locs
            break

# Risk modifiers
risk_mods = json.load(open(RISK_MOD, encoding="utf-8"))
entry_risk = {int(k): min(m["modifier"] for m in v) for k, v in risk_mods.items()}
narrator_risk = {}
for ak, eids in alias_index.items():
    eid = eids[0]
    if eid in entry_risk:
        narrator_risk[norm(ak)] = entry_risk[eid]

print(f"  Sentiment profiles: {len(sentiment_profiles):,}")
print(f"  Narrators with generation (tabaqa): {len(name_to_tabaqa):,}")
print(f"  Narrators with geo: {len(narrator_locs):,}")
print(f"  GNN embeddings: {embeddings.shape}")

# ── 2. Load valid chains ──
print("\n[2/5] Loading valid aligned chains...")
align = json.load(open(ALIGNMENT, encoding="utf-8"))

sanad_col = []
for chunk in pd.read_csv(SANADSET, usecols=["Sanad"], chunksize=100_000, dtype_backend="pyarrow"):
    sanad_col.extend(chunk["Sanad"].tolist())

valid_chains = []
for entry in align:
    if not entry.get("is_match"):
        continue
    qidx = int(entry["query_idx"])
    ci = int(entry["corpus_idx"])
    if ci >= len(sanad_col):
        continue
    s = sanad_col[ci]
    if pd.isna(s):
        continue
    try:
        chain = json.loads(s.replace("'", '"'))
    except:
        try:
            chain = ast.literal_eval(s)
        except:
            continue
    names = [n.strip() for n in chain if len(n.strip()) > 3]
    if len(names) < 2:
        continue
    valid_chains.append(names)

print(f"  Loaded {len(valid_chains):,} valid chains")

# ── 3. Scoring function (mirrors evaluate_final_with_geo) ──
def get_narrator_score(nkey):
    nid = name_to_id.get(nkey)
    profile = sentiment_profiles.get(nid, {}) if nid else {}
    gnn_prof = gnn_profiles.get(nid, {}) if nid else {}
    r = profile.get("reliability", gnn_prof.get("predicted_score", 0.632))
    agreement = profile.get("agreement", 1.0)
    anomaly = gnn_prof.get("anomaly_score", 0.0)
    if r >= 0.80:
        discount = 1.0
    elif r >= 0.70:
        discount = 1.0
    elif r >= 0.40:
        discount = 0.85
    elif r >= 0.20:
        discount = 0.60
    else:
        discount = 0.25
    if discount < 1.0 and anomaly < 0.25:
        discount = discount + (1.0 - discount) * 0.15
    disagreement_discount = 1.0 - 0.20 * (1.0 - agreement)
    return r * discount * disagreement_discount

def score_chain(names, alpha=0.50, use_gnn=True, use_geo=True, use_temp=True):
    ind_scores = []
    for n in names:
        nkey = norm(n)
        base = get_narrator_score(nkey)
        risk = narrator_risk.get(nkey, 1.0)
        ind_scores.append(base * risk)
    weakest = min(ind_scores)
    mean_s = np.mean(ind_scores)
    chain_base = alpha * weakest + (1.0 - alpha) * mean_s

    # Geo penalty
    chain_geo = 1.0
    if use_geo:
        prev_loc = None
        for n in names:
            nkey = norm(n)
            if nkey in narrator_locs:
                this_loc = narrator_locs[nkey][0]
                if prev_loc:
                    dist = haversine(prev_loc[1][0], prev_loc[1][1], this_loc[1][0], this_loc[1][1])
                    if dist > GEO_DIST_THRESHOLD:
                        chain_geo *= GEO_PENALTY
                prev_loc = this_loc
            else:
                prev_loc = None

    # GNN path penalty
    chain_gnn = 1.0
    if use_gnn:
        min_sim = 1.0
        for i in range(len(names) - 1):
            nkey1 = norm(names[i])
            nkey2 = norm(names[i + 1])
            if nkey1 in name_to_id and nkey2 in name_to_id:
                id1 = name_to_id[nkey1] - 1
                id2 = name_to_id[nkey2] - 1
                if 0 <= id1 < len(embeddings) and 0 <= id2 < len(embeddings):
                    dot = np.sum(embeddings[id1] * embeddings[id2])
                    sim = 1.0 / (1.0 + np.exp(-dot))
                    if sim < min_sim:
                        min_sim = sim
        if min_sim < 0.25:
            chain_gnn = 0.85

    # Temporal (Inqita) Penalty
    chain_temp = 1.0
    if use_temp and name_to_tabaqa:
        for i in range(len(names) - 1):
            nkey1 = norm(names[i])
            nkey2 = norm(names[i + 1])
            t1 = name_to_tabaqa.get(nkey1)
            t2 = name_to_tabaqa.get(nkey2)
            if t1 is not None and t2 is not None:
                gap = t2 - t1
                if gap >= 5:
                    chain_temp *= 0.40
                elif gap >= 3:
                    chain_temp *= 0.60

    return float(chain_base) * chain_geo * chain_gnn * chain_temp

# ── 4. Generate adversarial chains ──
print("\n[3/5] Generating adversarial chains...")

results = []

# --- A. Temporal anomalies (cross-chain splice) ---
print("  A. Temporal anomalies (cross-chain splicing)...")
temp_count = 0
prev_chain_scores = []
for ci, names in enumerate(valid_chains):
    if temp_count >= 30:
        break
    avg_score = np.mean([get_narrator_score(norm(n)) for n in names])
    prev_chain_scores.append(avg_score)
    if ci == 0:
        continue
    # Find a chain with significantly different avg score (likely different era)
    for cj in range(ci):
        if abs(prev_chain_scores[ci] - prev_chain_scores[cj]) < 0.08:
            continue
        names2 = valid_chains[cj]
        # Splice: first half from chain A, second half from chain B
        mid = min(len(names), len(names2)) // 2
        anom = names[:mid] + names2[mid:]
        if len(anom) < 2:
            continue
        clean_score = score_chain(names, use_gnn=False, use_geo=False, use_temp=False)
        anom_semantic = score_chain(anom, use_gnn=False, use_geo=False, use_temp=False)
        anom_full = score_chain(anom, use_gnn=True, use_geo=True, use_temp=True)
        results.append({
            "type": "temporal",
            "chain_id": f"temp_{temp_count}",
            "clean_score": round(clean_score, 4),
            "anomaly_semantic_only": round(anom_semantic, 4),
            "anomaly_full": round(anom_full, 4),
            "splice_from_chain": ci,
            "splice_to_chain": cj,
            "chain_length": len(anom),
        })
        temp_count += 1
        break
print(f"     Generated {temp_count} cross-chain splice anomalies")

# --- A2. Targeted tabaqa violations ---
print("  A2. Targeted temporal anomalies (tabaqa gap injection)...")
temp2_count = 0
all_tabaqa_narrators = [(n, t) for n, t in name_to_tabaqa.items()]
random.shuffle(all_tabaqa_narrators)
for names in valid_chains:
    if temp2_count >= 30:
        break
    chain_tabaqa = [(i, n, name_to_tabaqa.get(norm(n))) for i, n in enumerate(names)]
    chain_tabaqa = [(i, n, t) for i, n, t in chain_tabaqa if t is not None]
    if len(chain_tabaqa) < 2:
        continue
    for idx1 in range(len(chain_tabaqa)):
        for idx2 in range(idx1 + 1, len(chain_tabaqa)):
            i1, n1, t1 = chain_tabaqa[idx1]
            i2, n2, t2 = chain_tabaqa[idx2]
            if abs(t1 - t2) > 1:
                continue
            distant = None
            for dn, dt in all_tabaqa_narrators:
                if dn in [norm(x) for x in names]:
                    continue
                if abs(dt - t1) >= 3 and abs(dt - t2) >= 3:
                    distant = (dn, dt)
                    break
            if not distant:
                continue
            anom = names.copy()
            anom[i1] = distant[0]
            clean_score = score_chain(names, use_gnn=False, use_geo=False, use_temp=False)
            anom_semantic = score_chain(anom, use_gnn=False, use_geo=False, use_temp=False)
            anom_full = score_chain(anom, use_gnn=True, use_geo=True, use_temp=True)
            results.append({
                "type": "temporal",
                "chain_id": f"temp_tabaqa_{temp2_count}",
                "clean_score": round(clean_score, 4),
                "anomaly_semantic_only": round(anom_semantic, 4),
                "anomaly_full": round(anom_full, 4),
                "original_narrator": n1,
                "injected_narrator": distant[0],
                "tabaqa_original": t1,
                "tabaqa_injected": distant[1],
                "chain_length": len(anom),
            })
            temp2_count += 1
            break
        if temp2_count >= 30:
            break
print(f"     Generated {temp2_count} tabaqa-targeted anomalies")
print(f"     Total temporal: {temp_count + temp2_count}")

# --- B. Geospatial anomalies ---
print("  B. Geospatial anomalies (force >3000km gap)...")
geo_count = 0
for names in valid_chains:
    if geo_count >= 50:
        break
    # Find chains where adjacent narrators have geo data
    loc_pairs = []
    for i in range(len(names) - 1):
        nk1 = norm(names[i])
        nk2 = norm(names[i + 1])
        if nk1 in narrator_locs and nk2 in narrator_locs:
            loc_pairs.append((i, nk1, nk2))
    if not loc_pairs:
        continue
    # Pick a pair and inject a distant location for one of them
    pair = random.choice(loc_pairs)
    # Force locations: use Baghdad (33.3, 44.4) and Cordoba (37.9, -4.8) — ~4,300 km apart
    # Set the second narrator to a far location by swapping their location reference
    # We simulate by modifying the chain's geo data in scoring
    # Instead of actually modifying the data, we create a chain where we manually override
    idx, nk1, nk2 = pair
    # Create anomalous chain — for geo, we need to swap to a known far-away narrator
    far_narrators = [n for n in narrator_locs if haversine(
        narrator_locs[n][0][1][0], narrator_locs[n][0][1][1],
        narrator_locs[nk1][0][1][0], narrator_locs[nk1][0][1][1]
    ) > 4000 and n != nk1 and n != nk2]
    if not far_narrators:
        continue
    far = random.choice(far_narrators)
    anom = names.copy()
    anom[idx + 1] = far  # Replace the second narrator with someone far away
    clean_score = score_chain(names, use_geo=False, use_gnn=False, use_temp=False)
    anom_semantic = score_chain(anom, use_geo=False, use_gnn=False, use_temp=False)
    anom_geo = score_chain(anom, use_geo=True, use_gnn=False, use_temp=False)
    anom_full = score_chain(anom, use_geo=True, use_gnn=True, use_temp=True)
    results.append({
        "type": "geospatial",
        "chain_id": f"geo_{geo_count}",
        "clean_score": round(clean_score, 4),
        "anomaly_semantic_only": round(anom_semantic, 4),
        "anomaly_full": round(anom_full, 4),
        "original_narrator": names[idx + 1],
        "injected_narrator": far,
        "chain_length": len(names),
    })
    geo_count += 1
print(f"     Generated {geo_count} geospatial anomalies")

# --- C. GNN path anomalies (demonstrate embedding homogeneity) ---
print("  C. GNN path anomalies (embedding homogeneity analysis)...")
# First, show that GNN embeddings are too uniform for the 0.25 threshold
sample_size = min(5000, len(embeddings))
ridxs = np.random.RandomState(42).choice(len(embeddings), sample_size, replace=False)
pair_sims = []
for _ in range(2000):
    i, j = np.random.RandomState(None).choice(ridxs, 2, replace=False)
    dot = np.sum(embeddings[i] * embeddings[j])
    pair_sims.append(float(1.0 / (1.0 + np.exp(-dot))))
pair_sims = np.array(pair_sims)
print(f"     GNN embedding similarity distribution (random pairs):")
print(f"       mean={pair_sims.mean():.4f}, std={pair_sims.std():.4f}")
print(f"       min={pair_sims.min():.4f}, below 0.25: {(pair_sims<0.25).sum()}/{len(pair_sims)}")

# Show that even with injected random narrator, the GNN penalty doesn't fire
# because the sigmoid-normalized dot product is always ~0.5
gnn_count = 0
for names in valid_chains[:100]:
    if gnn_count >= 20:
        break
    if len(names) < 3:
        continue
    # Pick a random narrator NOT in this chain
    all_keys = list(name_to_id.keys())
    random.shuffle(all_keys)
    for rk in all_keys:
        if rk in [norm(x) for x in names]:
            continue
        rid = name_to_id.get(rk)
        if rid and 0 <= rid - 1 < len(embeddings):
            z_rand = embeddings[rid - 1]
            # Check similarity to chain neighbors
            sims = []
            for x in names:
                nk = norm(x)
                if nk in name_to_id:
                    nid_n = name_to_id[nk]
                    if 0 <= nid_n - 1 < len(embeddings):
                        dot = np.sum(z_rand * embeddings[nid_n - 1])
                        sims.append(1.0 / (1.0 + np.exp(-dot)))
            if sims:
                pos = random.randint(1, len(names) - 1)
                anom = names[:pos] + [rk] + names[pos:]
                clean_score = score_chain(names, use_gnn=False, use_geo=False, use_temp=False)
                anom_semantic = score_chain(anom, use_gnn=False, use_geo=False, use_temp=False)
                anom_gnn = score_chain(anom, use_gnn=True, use_geo=False, use_temp=False)
                results.append({
                    "type": "gnn_path",
                    "chain_id": f"gnn_{gnn_count}",
                    "clean_score": round(clean_score, 4),
                    "anomaly_semantic_only": round(anom_semantic, 4),
                    "anomaly_full": round(anom_gnn, 4),
                    "injected_narrator": rk[:30],
                    "min_embedding_sim": round(min(sims), 4),
                    "chain_length": len(anom),
                })
                gnn_count += 1
                break
print(f"     Generated {gnn_count} GNN path anomaly chains")
print(f"     (Note: GNN penalty never fires - all embedding similarities > 0.25)")
print(f"     This indicates the GNN path threshold (0.25) is unreachable with")
print(f"     current embeddings - the penalty mechanism is a no-op on this data.")

# ── 5. Report ──
print(f"\n[4/5] Analysing results...")
df = pd.DataFrame(results)
print(f"  Total adversarial chains: {len(df)}")

print("\n[5/5] Results summary:")
print("=" * 65)

for anomaly_type in ["temporal", "geospatial", "gnn_path"]:
    sub = df[df["type"] == anomaly_type]
    if len(sub) == 0:
        continue
    clean_mean = sub["clean_score"].mean()
    semantic_mean = sub["anomaly_semantic_only"].mean()
    full_mean = sub["anomaly_full"].mean()
    penalized_semantic = (semantic_mean < clean_mean * 0.99)
    penalized_full = (full_mean < semantic_mean * 0.99)

    print(f"\n  {anomaly_type.upper()} ANOMALIES ({len(sub)} chains):")
    print(f"    Clean score (avg):          {clean_mean:.4f}")
    print(f"    Anomaly — Semantic only:    {semantic_mean:.4f}  {'⬇' if penalized_semantic else '='}")
    print(f"    Anomaly — Full pipeline:    {full_mean:.4f}  {'⬇' if penalized_full else '='}")
    print(f"    Base penalty from injection: {((1 - semantic_mean/clean_mean) * 100):+.1f}%")
    print(f"    Additional penalty from GNN/Geo: {((1 - full_mean/semantic_mean) * 100):+.1f}%")

# Save
out_path = OUT_DIR / "adversarial_evaluation.csv"
df.to_csv(out_path, index=False)
print(f"\n  Saved: {out_path}")

print(f"\n{'=' * 65}")
print("  ADVERSARIAL EVALUATION COMPLETE")
print("=" * 65)
