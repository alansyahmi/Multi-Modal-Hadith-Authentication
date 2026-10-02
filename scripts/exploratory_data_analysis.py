"""
Exploratory Data Analysis — comprehensive three-dimension analysis.
Generates statistics and placeholder references for visualizations.
Saves numerical results to findings/ and prints summary to stdout.

Dimensions:
  1. Textual & Label Distribution (NLP)
  2. Graph Topology & Structural (KG/GNN)
  3. Spatiotemporal Metadata (Domain)
"""
import json, sys, io, math, re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "findings"
OUT_DIR.mkdir(exist_ok=True)

results = {}

print("=" * 68)
print("  EXPLORATORY DATA ANALYSIS  —  Three-Dimension Report")
print("=" * 68)

# ========================================================================
# DIMENSION 1 — TEXTUAL & LABEL DISTRIBUTION  (NLP)
# ========================================================================
print("\n" + "━" * 68)
print("  DIMENSION 1: TEXTUAL & LABEL DISTRIBUTION  (NLP)")
print("━" * 68)

# --- 1a. Class Imbalance Analysis  (from encyclopedia) ---
print("\n[1a] Grade classification imbalance…")
enc_path = PROJECT / "output/shamela/narrator_encyclopedia.json"
enc = json.load(open(enc_path, encoding="utf-8"))

gc = Counter()
for e in enc:
    gs = e.get("grades", [])
    for g in gs:
        if isinstance(g, dict):
            gc[g.get("grade", "")] += 1
        elif isinstance(g, str):
            gc[g] += 1

total_grades_assigned = sum(gc.values())
print(f"  Total grade-bearing entries: {total_grades_assigned:,}")

grade_map = {
    "ثقة": "Thiqah", "صدوق": "Saduq", "مقبول": "Maqbul",
    "ضعيف": "Daif", "متروك": "Matruk", "مجهول": "Majhul",
}
grade_buckets = defaultdict(int)
for g, c in gc.items():
    matched = False
    for arabic, eng in grade_map.items():
        if arabic in g:
            grade_buckets[eng] += c
            matched = True
            break
    if not matched:
        grade_buckets["Other"] += c

results["grade_distribution"] = {}
for label in ["Thiqah", "Saduq", "Maqbul", "Daif", "Matruk", "Majhul", "Other"]:
    cnt = grade_buckets.get(label, 0)
    pct = cnt / total_grades_assigned * 100
    results["grade_distribution"][label] = {"count": int(cnt), "pct": round(pct, 1)}
    print(f"    {label:15s}  {cnt:>6,}  ({pct:5.1f}%)")

# Class imbalance stats
positive = grade_buckets["Thiqah"] + grade_buckets["Saduq"] + grade_buckets["Maqbul"]
negative = grade_buckets["Daif"] + grade_buckets["Matruk"] + grade_buckets["Majhul"]
print(f"  Positive classes (Thiqah+Saduq+Maqbul): {positive:,} ({positive/total_grades_assigned*100:.1f}%)")
print(f"  Negative classes (Daif+Matruk+Majhul):  {negative:,} ({negative/total_grades_assigned*100:.1f}%)")
print(f"  Pos:Neg ratio ≈ {positive/negative:.1f}:1")
print(f"  [VIS]  →  Bar chart: grade category counts, log-scale y-axis")

# --- 1b. Biography text length variations ---
print("\n[1b] Biography text length variations…")
text_lengths = np.array([len(e.get("full_text", "")) for e in enc if e.get("full_text")])
word_est = text_lengths / 6.0

print(f"  Entries with text: {len(text_lengths):,}")
print(f"  Character length:  mean={text_lengths.mean():.0f}, median={np.median(text_lengths):.0f}")
print(f"    P10={np.percentile(text_lengths,10):.0f}, P90={np.percentile(text_lengths,90):.0f}")
print(f"  Estimated word count: mean={word_est.mean():.0f}, median={np.median(word_est):.0f}")
p25, p75 = np.percentile(text_lengths, 25), np.percentile(text_lengths, 75)
print(f"  IQR: {p75-p25:.0f} chars  (P25={p25:.0f}, P75={p75:.0f})")
very_short = int((text_lengths < 200).sum())
print(f"  Very short entries (<200 chars, alignment challenge): {very_short:,} ({very_short/len(text_lengths)*100:.1f}%)")
print(f"  [VIS]  →  Histogram: biography text lengths (chars), log-scale x-axis")

results["text_lengths"] = {
    "count": int(len(text_lengths)),
    "mean_chars": round(float(text_lengths.mean()), 1),
    "median_chars": round(float(np.median(text_lengths)), 1),
    "pct_very_short": round(very_short / len(text_lengths) * 100, 1),
}

# --- 1c. Alias density ---
print("\n[1c] Alias density…")
alias = json.load(open(PROJECT / "output/shamela/alias_index.json", encoding="utf-8"))
print(f"  Total alias forms in index: {len(alias):,}")

if isinstance(next(iter(alias.values())), list):
    alias_counts = [len(v) for v in alias.values()]
    avg = np.mean(alias_counts)
    med = np.median(alias_counts)
    mx = max(alias_counts)
    ge5 = sum(1 for c in alias_counts if c >= 5)
    ge20 = sum(1 for c in alias_counts if c >= 20)
    print(f"  Avg alternative forms per canonical narrator: {avg:.2f}, median={med:.1f}, max={mx:,}")
    print(f"  Narrators with ≥5 alternative forms: {ge5:,} ({ge5/len(alias)*100:.1f}%)")
    print(f"  Narrators with ≥20 alternative forms: {ge20:,} ({ge20/len(alias)*100:.1f}%)")
    print(f"  => Simple string-matching DB inadequate; Levenshtein fuzzy matching essential.")
    results["alias_density"] = {
        "total_forms": len(alias),
        "avg_per_narrator": round(float(avg), 2),
        "max_per_narrator": int(mx),
        "pct_ge5": round(ge5 / len(alias) * 100, 1),
    }
else:
    results["alias_density"] = {"total_forms": len(alias)}

print(f"  [VIS]  →  Histogram: number of alias forms per narrator, log x-axis")


# ========================================================================
# DIMENSION 2 — GRAPH TOPOLOGY & STRUCTURAL  (KG/GNN)
# ========================================================================
print("\n" + "━" * 68)
print("  DIMENSION 2: GRAPH TOPOLOGY & STRUCTURAL  (KG/GNN)")
print("━" * 68)

# Full graph stats from node_names.json
node_names = json.load(open(PROJECT / "output/stage2/node_names.json", encoding="utf-8"))
full_node_count = len(node_names)

# Graded subgraph stats from CSV
nodes_df = pd.read_csv(PROJECT / "output/stage2/graph_nodes.csv")
edges_df = pd.read_csv(PROJECT / "output/stage2/graph_edges.csv")
graded_count = len(nodes_df)
edge_count = len(edges_df)

print(f"\n[2a] Graph scale overview…")
print(f"  Full graph (from node_names.json):  {full_node_count:,} nodes")
print(f"  Graded subgraph (from stage2):      {graded_count:,} nodes / {edge_count:,} edges")
cold_start_nodes = full_node_count - graded_count
cold_pct = cold_start_nodes / full_node_count * 100
print(f"  Cold-start nodes (no grade):         {cold_start_nodes:,} ({cold_pct:.2f}%)")

# --- Degree distribution from graded subgraph ---
print(f"\n  Degree distribution (graded subgraph):")
degrees = nodes_df["in_degree"].fillna(0) + nodes_df["out_degree"].fillna(0)
print(f"    mean={degrees.mean():.2f}, median={degrees.median():.1f}, max={degrees.max():,}")
print(f"    P10={np.percentile(degrees,10):.1f}, P90={np.percentile(degrees,90):.1f}")

# Power-law fit
deg_counts = Counter(degrees)
deg_vals = np.array(sorted(d for d in deg_counts.keys() if d > 0))
deg_freq = np.array([deg_counts[d] for d in deg_vals])
mask = deg_freq >= 3
if mask.sum() > 5:
    logk = np.log(deg_vals[mask])
    logp = np.log(deg_freq[mask])
    coeffs = np.polyfit(logk, logp, 1)
    gamma = -coeffs[0]
    print(f"    Power-law exponent γ ≈ {gamma:.3f}  (log-log slope)")
else:
    gamma = 0.0
    print(f"    (insufficient data for power-law fit)")

top1p = max(1, int(len(degrees) * 0.01))
top_share = sum(sorted(degrees, reverse=True)[:top1p]) / max(sum(degrees), 1) * 100
print(f"    Top 1% nodes account for {top_share:.1f}% of total degree")
print(f"  [VIS]  →  Log-log degree distribution plot with fitted power-law line")

results["degree_distribution"] = {
    "full_nodes": full_node_count,
    "graded_nodes": graded_count,
    "edges": edge_count,
    "mean_degree": round(float(degrees.mean()), 2),
    "median_degree": round(float(degrees.median()), 1),
    "max_degree": int(degrees.max()),
    "gamma": round(float(gamma), 3),
    "top_1pct_share": round(float(top_share), 1),
    "cold_start_pct": round(cold_pct, 2),
}

# --- 2b. Attribute sparsity ---
print(f"\n[2b] Graph cold-start (attribute sparsity)…")
graded = nodes_df["grade"].notna() & (nodes_df["grade"] != "")
tabaqa_node = nodes_df["tabaqa"].notna() & (nodes_df["tabaqa"] != "")
death_node = nodes_df["death_info"].notna() & (nodes_df["death_info"] != "")

print(f"  Graded subgraph attribute coverage:")
print(f"    With grade:    {graded.sum():,} ({graded.sum()/graded_count*100:.2f}%)")
print(f"    With tabaqa:   {tabaqa_node.sum():,} ({tabaqa_node.sum()/graded_count*100:.2f}%)")
print(f"    With death:    {death_node.sum():,} ({death_node.sum()/graded_count*100:.2f}%)")

all_missing = (~graded) & (~tabaqa_node) & (~death_node)
print(f"    Missing ALL:   {all_missing.sum():,} ({all_missing.sum()/graded_count*100:.2f}%)")
print(f"  [VIS]  →  Stacked bar or Venn diagram: attribute coverage overlap")

# Cold-start in context of FULL graph
print(f"\n  Cold-start in full-graph context:")
print(f"    {cold_start_nodes:,} nodes ({cold_pct:.1f}%) have no grade — GNN must rely on neighbourhood embedding.")
print(f"    Only {int(graded.sum()):,} nodes ({graded.sum()/full_node_count*100:.2f}%) have explicit grade labels.")

results["cold_start"] = {
    "full_nodes": full_node_count,
    "graded_subgraph": graded_count,
    "cold_start_nodes": cold_start_nodes,
    "cold_start_pct": round(cold_pct, 2),
    "with_grade_in_subgraph": int(graded.sum()),
    "with_tabaqa_in_subgraph": int(tabaqa_node.sum()),
    "with_death_in_subgraph": int(death_node.sum()),
}


# ========================================================================
# DIMENSION 3 — SPATIOTEMPORAL METADATA  (Domain)
# ========================================================================
print("\n" + "━" * 68)
print("  DIMENSION 3: SPATIOTEMPORAL METADATA  (Domain)")
print("━" * 68)

# --- 3a. Tabaqa generation distribution ---
print("\n[3a] Tabaqa generation distribution…")
# Tabaqa patterns from graph_nodes (encyclopedia tabaqa field is the same data)
# Mapping from truncated Arabic ordinal patterns to generation numbers
tabaqa_gen_map = {
    "الثاني": 2, "الثالث": 3, "الرابع": 4, "الخامس": 5,
    "السادس": 6, "السابع": 7, "الثامن": 8, "التاسع": 9,
    "العا": 10,     # truncated "العاشرة"
    "الحادي": 11,
}
# Narrator generations (tabaqa via column in graph_nodes)
tabaqa_pops = Counter()
unparseable = 0
for _, row in nodes_df.iterrows():
    t = str(row.get("tabaqa", ""))
    if not t or t == "nan":
        continue
    matched = False
    for pat, gen in tabaqa_gen_map.items():
        if pat in t:
            tabaqa_pops[gen] += 1
            matched = True
            break
    if not matched:
        unparseable += 1

total_tabaqa = sum(tabaqa_pops.values()) + unparseable
print(f"  Nodes with tabaqa: {total_tabaqa:,}")

results["tabaqa_distribution"] = {}
for gen in range(2, 12):
    cnt = tabaqa_pops.get(gen, 0)
    pct = cnt / max(total_tabaqa, 1) * 100
    results["tabaqa_distribution"][gen] = {"count": cnt, "pct": round(pct, 1)}
    ce_start = 622 + (gen - 1) * 21
    ce_end = 622 + gen * 21
    bar = "█" * max(1, int(pct // 2))
    print(f"    Gen {gen:2d} (ca.{ce_start}–{ce_end} CE): {cnt:>5,} ({pct:4.1f}%)  {bar}")

if unparseable:
    print(f"    Unparseable:                {unparseable:,} ({unparseable/total_tabaqa*100:.1f}%)")
print(f"  [VIS]  →  Bar chart: narrator count per tabaqa generation (11 groups)")
print(f"  [VIS]  →  Timeline with key compiler lifetimes overlaid")

# --- 3b. Geographic density mapping ---
print("\n[3b] Geographic density mapping…")
geo = json.load(open(PROJECT / "output/geo/entry_locations.json", encoding="utf-8"))
print(f"  Total mapped municipal coordinates: {len(geo):,}")

# Geo structure is list of [city_name, [lat, lon]]
city_counter = Counter()
coord_counter = Counter()
for eid, loc_data in geo.items():
    if isinstance(loc_data, list) and len(loc_data) > 0 and isinstance(loc_data[0], list):
        city = loc_data[0][0]
        city_counter[city] += 1
        coord = tuple(loc_data[0][1]) if len(loc_data[0]) > 1 else None
        if coord:
            coord_counter[coord] += 1

print(f"  Unique named locations: {len(city_counter):,}")
print(f"  Unique coordinate pairs: {len(coord_counter):,}")
print(f"  Top 15 cities by narrator count:")
results["geo_top_cities"] = []
for city, cnt in city_counter.most_common(15):
    pct = cnt / len(geo) * 100
    results["geo_top_cities"].append({"city": city, "count": cnt, "pct": round(pct, 1)})
    bar = "█" * max(1, int(pct // 2))
    print(f"    {city:25s}: {cnt:>5,} ({pct:4.1f}%)  {bar}")

# Regional grouping
region_rules = {
    "Medina":    ["المدينة", "مدينة"],
    "Mecca":     ["مكة", "مك"],
    "Kufa":      ["الكوفة", "كوف"],
    "Basra":     ["البصرة", "بصر"],
    "Baghdad":   ["بغداد", "بغدا"],
    "Damascus":  ["دمشق"],
    "Egypt":     ["مصر", "القاهرة", "الفسطاط", "الإسكندر"],
    "Yemen":     ["اليمن", "صنعاء"],
    "Wasit":     ["واسط"],
    "Marw":      ["مرو"],
    "Nishapur":  ["نيسابور", "نيشابور"],
    "Rayy":      ["الري", "الرى"],
    "Jerusalem": ["القدس", "بيت المقدس", "أيلياء"],
    "Tunisia":   ["تونس", "القيروان"],
    "Spain":     ["الأندلس", "قرطبة", "إشبيلية"],
}
region_counts = Counter()
other_geo = 0
for city, cnt in city_counter.items():
    assigned = False
    for region, kws in region_rules.items():
        for kw in kws:
            if kw in city:
                region_counts[region] += cnt
                assigned = True
                break
        if assigned:
            break
    if not assigned:
        other_geo += cnt
region_counts["Other"] = other_geo

print(f"\n  Regional concentration (grouped):")
results["geo_regions"] = {}
for region, cnt in region_counts.most_common():
    pct = cnt / len(geo) * 100
    results["geo_regions"][region] = {"count": cnt, "pct": round(pct, 1)}
    bar = "█" * max(1, int(pct // 2))
    print(f"    {region:15s}: {cnt:>5,} ({pct:4.1f}%)  {bar}")
print(f"  [VIS]  →  Geo-map scatter plot: coordinate points over early Islamic world map")
print(f"  [VIS]  →  Heatmap or bubble chart: density per city/region")


# ========================================================================
# SAVE RESULTS
# ========================================================================
results_path = OUT_DIR / "eda_results.json"
json.dump(results, open(results_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"\n  Results saved to: {results_path}")

# Count total data points in results
total_stats = sum(len(v) if isinstance(v, dict) else 1 for v in results.values())
print(f"\n" + "=" * 68)
print(f"  EDA COMPLETE — {total_stats} statistics across 3 dimensions")
print("=" * 68)
