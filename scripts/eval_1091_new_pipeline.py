"""
Quick eval: sentiment (upgrade-only) + risk + geo.
GNN embeddings: not used — trained for regression, not link prediction.
"""
import sys, re, ast, json, math
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
sys.stdout.reconfigure(encoding="utf-8")

def norm(t):
    return re.sub(r"[ً-ٌٍ-َّْٰ]", "", t).replace("أ","ا").replace("إ","ا").replace("آ","a").replace("ة","ه").replace("ى","ي").strip()

print("=" * 60)
print("QUICK EVAL: Sentiment (upgrade-only) + Risk + Geo")
print("=" * 60)

# ── Sentiment scores ──
sent = pd.read_csv("output/track_a_classifier/narrator_scores_sentiment.csv")
sent_2plus = {r["node_id"]: r["reliability_score"] for _, r in sent.iterrows() if r["critic_count"] >= 2}
sent_any = {r["node_id"]: r["reliability_score"] for _, r in sent.iterrows() if r["critic_count"] >= 1}
print(f"  ≥2 critics (true jumhur): {len(sent_2plus):,}")
print(f"  ≥1 critic: {len(sent_any):,}")

# ── GNN predictions (baseline) ──
gnn = {r["node_id"]: r["predicted_score"] for _, r in pd.read_csv("output/stage3/narrator_predictions.csv").iterrows()}

# ── Name→score: sentiment as UPGRADE only, GNN as fallback ──
with open("output/stage2/node_names.json", encoding="utf-8") as f: id_to_name = json.load(f)
name_to_score = {}
for nid_str, name in id_to_name.items():
    nid = int(nid_str)
    gnn_s = float(gnn.get(nid, 0.632))
    sent_s = sent_any.get(nid)
    # Use sentiment ONLY if it gives a HIGHER score than GNN (upgrade only)
    if sent_s and sent_s > gnn_s:
        name_to_score[norm(name)] = sent_s
    else:
        name_to_score[norm(name)] = gnn_s
print(f"  Name→score: {len(name_to_score):,}")

# ── Risk + Geo ──
alias = json.load(open("output/shamela/alias_index.json", encoding="utf-8"))
rm = json.load(open("output/risk_modifiers/risk_modifiers.json", encoding="utf-8"))
er = {int(k): min(m["modifier"] for m in v) for k, v in rm.items()}
nr = {}
for ak, eids in alias.items():
    if eids[0] in er: nr[norm(ak)] = er[eids[0]]

entry_locs = json.load(open("output/geo/entry_locations.json", encoding="utf-8"))
nisba_city = json.load(open("output/geo/nisba_to_city.json", encoding="utf-8"))
city_coords = json.load(open("output/geo/city_coords.json", encoding="utf-8"))
NISBA_RE = re.compile(r"(ال[؀-ۿ]{2,20}[ي])")
nl = {}
for ak, eids in alias.items():
    eid_str = str(eids[0])
    if eid_str in entry_locs: nl[norm(ak)] = entry_locs[eid_str]
for nk in list(name_to_score.keys()):
    if nk not in nl:
        for m in NISBA_RE.finditer(nk):
            if m.group(1) in nisba_city:
                c = nisba_city[m.group(1)]
                nl[nk] = [(c, city_coords[c])]; break

def haversine(lat1, lon1, lat2, lon2):
    R = 6371; dlat = math.radians(lat2-lat1); dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R*2*math.atan2(math.sqrt(a), math.sqrt(1-a))

# ── Score chains ──
eval_df = pd.read_csv("output/final_evaluation/sanadset_1200.csv")
print(f"Eval: {len(eval_df)} hadiths")

T = (0.45, 0.425, 0.15)
actual, preds = [], []
for _, row in eval_df.iterrows():
    s = row["sanad"]
    try: ch = json.loads(s.replace("'", '"'))
    except:
        try: ch = ast.literal_eval(s)
        except: continue
    names = [n.strip() for n in ch if len(n.strip()) > 3]
    if len(names) < 2: continue

    scores, prev_loc = [], None
    for n in names:
        nk = norm(n)
        base = name_to_score.get(nk, 0.632)
        risk = nr.get(nk, 1.0)
        scores.append(base * risk)
        if nk in nl:
            loc = nl[nk][0]
            if prev_loc and haversine(prev_loc[1][0], prev_loc[1][1], loc[1][0], loc[1][1]) > 3000:
                scores[-1] *= 0.85
            prev_loc = loc
        else: prev_loc = None

    cs = (sorted(scores)[0] + sorted(scores)[1]) / 2 if len(scores) >= 2 else scores[0]
    g = "Sahih" if cs >= T[0] else "Hasan" if cs >= T[1] else "Daif" if cs >= T[2] else "Matruk"
    actual.append(row["grade"]); preds.append(g)

n = len(actual)
acc = accuracy_score(actual, preds)
f1 = f1_score(actual, preds, average="macro", zero_division=0)
k = cohen_kappa_score(actual, preds, weights="quadratic")

print(f"\nResults (n={n}):")
print(f"  Accuracy: {acc:.4f}")
print(f"  F1 Macro: {f1:.4f}")
print(f"  Cohen's κ: {k:.4f}")

grades = ["Sahih","Hasan","Daif","Matruk"]
print(f"\n{'':>10s} " + " ".join(f"{g:>8s}" for g in grades))
for a in grades:
    r = [sum(1 for i in range(n) if actual[i]==a and preds[i]==p) for p in grades]
    print(f"  {a:>8s} " + " ".join(f"{c:>8d}" for c in r))
for g in grades:
    c = sum(1 for i in range(n) if actual[i]==g and preds[i]==g)
    t = sum(1 for a in actual if a==g)
    print(f"  {g:>8s} recall: {c}/{t} ({c/max(t,1)*100:.1f}%)")
