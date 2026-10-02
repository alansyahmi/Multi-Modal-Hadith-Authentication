"""Evaluate scoring pipeline across all 4 dataset parameters for alpha in {0.25, 0.50, 0.75}."""
import json
import ast
import sys
import re
import math
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ة","ه").replace("ى","ي").strip()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371; dlat = math.radians(lat2-lat1); dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R*2*math.atan2(math.sqrt(a), math.sqrt(1-a))

GEO_DIST_THRESHOLD = 3000
GEO_PENALTY = 0.85

def main():
    print("=" * 70)
    print("  EVALUATING ALL 4 DATASET PARAMETERS ACROSS ALPHA SWEEPS")
    print("=" * 70)

    SANADSET = Path("Dataset/Sanadset 650K Data on Hadith Narrators/Sanadset 650K Data on Hadith Narrators/sanadset.csv")
    NODE_NAMES = Path("output/stage2/node_names.json")
    GNN_PRED = Path("output/stage3/narrator_predictions.csv")
    RISK_MOD = Path("output/risk_modifiers/risk_modifiers.json")
    ALIAS_IDX = Path("output/shamela/alias_index.json")
    ENTRY_LOCS = Path("output/geo/entry_locations.json")
    STAGE1_DIR = Path("output/stage1")
    OUT_DIR = Path("output/final_evaluation")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load lookups
    preds = pd.read_csv(GNN_PRED)
    gnn_profiles = {int(r["node_id"]): {"predicted_score": float(r["predicted_score"]), "anomaly_score": float(r["anomaly_score"])} for _, r in preds.iterrows()}
    
    SENT_PATH = Path("output/track_a_classifier/narrator_scores_sentiment.csv")
    sentiment_profiles = {}
    if SENT_PATH.exists():
        sent_df = pd.read_csv(SENT_PATH)
        sentiment_profiles = {int(r["node_id"]): {"reliability": float(r["reliability_score"]), "agreement": float(r["agreement_score"])} for _, r in sent_df.iterrows()}

    id_to_name = json.load(open(NODE_NAMES, encoding="utf-8"))
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

    # Load sanad text column from Sanadset
    print("Loading Sanad column from Sanadset...")
    sanad_col = []
    for chunk in pd.read_csv(SANADSET, usecols=["Sanad"], chunksize=100_000, dtype_backend="pyarrow"):
        sanad_col.extend(chunk["Sanad"].tolist())

    def get_narrator_score(nkey):
        nid = name_to_id.get(nkey)
        profile = sentiment_profiles.get(nid, {}) if nid else {}
        gnn_prof = gnn_profiles.get(nid, {}) if nid else {}
        r = profile.get("reliability", gnn_prof.get("predicted_score", 0.632))
        agreement = profile.get("agreement", 1.0)
        anomaly = gnn_prof.get("anomaly_score", 0.0)
        
        if r >= 0.80: discount = 1.0
        elif r >= 0.70: discount = 1.0
        elif r >= 0.40: discount = 0.85
        elif r >= 0.20: discount = 0.60
        else: discount = 0.25

        if r < 0.70 and r >= 0.20 and anomaly < 0.25:
            discount = discount + (1.0 - discount) * 0.15

        disagreement_discount = 1.0 - 0.20 * (1.0 - agreement)
        return r * discount * disagreement_discount

    def evaluate_chain_score(names, alpha, use_gnn=True, use_geo=True):
        ind_scores = [get_narrator_score(norm(n)) * narrator_risk.get(norm(n), 1.0) for n in names]
        weakest = min(ind_scores)
        mean_s = np.mean(ind_scores)
        chain_base = alpha * weakest + (1.0 - alpha) * mean_s
        
        chain_geo_penalty = 1.0
        if use_geo:
            prev_loc = None
            for n in names:
                nkey = norm(n)
                if nkey in narrator_locs:
                    this_loc = narrator_locs[nkey][0]
                    if prev_loc:
                        dist = haversine(prev_loc[1][0], prev_loc[1][1], this_loc[1][0], this_loc[1][1])
                        if dist > GEO_DIST_THRESHOLD: chain_geo_penalty *= GEO_PENALTY
                    prev_loc = this_loc
                else: prev_loc = None
                    
        chain_gnn_penalty = 1.0
        if use_gnn and gnn_embeddings is not None:
            min_sim = 1.0
            for i in range(len(names) - 1):
                nkey1, nkey2 = norm(names[i]), norm(names[i+1])
                if nkey1 in name_to_id and nkey2 in name_to_id:
                    id1, id2 = name_to_id[nkey1] - 1, name_to_id[nkey2] - 1
                    if 0 <= id1 < len(gnn_embeddings) and 0 <= id2 < len(gnn_embeddings):
                        dot = np.sum(gnn_embeddings[id1] * gnn_embeddings[id2])
                        sim = 1.0 / (1.0 + np.exp(-dot))
                        if sim < min_sim: min_sim = sim
            if min_sim < 0.25: chain_gnn_penalty = 0.85
                
        return float(chain_base) * chain_geo_penalty * chain_gnn_penalty

    def calibrate_and_eval(scores, labels, tiers):
        sorted_scores = sorted(scores, reverse=True)
        N = len(scores)
        t_sahih = sorted_scores[int(N * 0.25)] if N > 0 else 0.70
        t_hasan = sorted_scores[int(N * 0.50)] if N > 0 else 0.55
        t_daif = sorted_scores[int(N * 0.75)] if N > 0 else 0.40

        preds_tier = []
        for s in scores:
            if s >= t_sahih: preds_tier.append(4)
            elif s >= t_hasan: preds_tier.append(3)
            elif s >= t_daif: preds_tier.append(2)
            else: preds_tier.append(1)

        acc = float(accuracy_score(tiers, preds_tier))
        f1 = float(f1_score(tiers, preds_tier, average="macro"))
        kappa = float(cohen_kappa_score(tiers, preds_tier, weights="quadratic"))
        return acc, f1, kappa

    params = ["1200_book_stratified", "1200_grade_balanced", "5000_book_stratified", "5000_grade_balanced"]
    final_summary = {}

    grade_labels = {"Ṣaḥīḥ": "Sahih", "Ḥasan": "Hasan", "Ḍaʿīf": "Daif", "Mawḍūʿ": "Matruk"}
    grade_tiers = {"Sahih": 4, "Hasan": 3, "Daif": 2, "Matruk": 1}

    for pname in params:
        eval_path = STAGE1_DIR / f"kutub_sitta_eval_{pname}.csv"
        align_path = STAGE1_DIR / f"alignment_results_{pname}.json"
        
        if not eval_path.exists() or not align_path.exists():
            print(f"⚠️ Missing files for {pname}, skipping.")
            continue

        eval_df = pd.read_csv(eval_path)
        eval_df["label"] = eval_df["grade_normalised"].map(grade_labels).fillna("Daif")
        eval_df["tier"] = eval_df["label"].map(grade_tiers)
        align = json.load(open(align_path, encoding="utf-8"))

        chains = []
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
            
            chains.append({
                "names": names,
                "true_tier": eval_df.iloc[aidx]["tier"],
                "true_label": eval_df.iloc[aidx]["label"]
            })

        print(f"\n[{pname}] Valid chains extracted: {len(chains)} / {len(align)}")
        
        alpha_results = {}
        for alpha in [0.25, 0.50, 0.75]:
            scores = [evaluate_chain_score(c["names"], alpha) for c in chains]
            tiers = [c["true_tier"] for c in chains]
            labels = [c["true_label"] for c in chains]
            
            acc, f1, kappa = calibrate_and_eval(scores, labels, tiers)
            alpha_results[str(alpha)] = {
                "accuracy": acc,
                "f1_macro": f1,
                "kappa": kappa
            }
            print(f"  alpha={alpha:.2f} -> Acc: {acc*100:.2f}%, F1: {f1*100:.2f}%, Kappa: {kappa:.4f}")

        final_summary[pname] = {
            "target_queries": len(eval_df),
            "valid_chains": len(chains),
            "alpha_sweeps": alpha_results
        }

    with open(OUT_DIR / "summary_all_4_params.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("EVALUATION OF ALL 4 PARAMETERS COMPLETE! Saved to summary_all_4_params.json")
    print("=" * 70)

if __name__ == "__main__":
    main()
