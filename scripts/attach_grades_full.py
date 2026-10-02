"""
Attach Ibn Hajar grades to the full-scale 262K-node KG.
Uses exact match first, then fuzzy match on unmatched only.
"""
import json, sys, numpy as np
from pathlib import Path
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rapidfuzz.fuzz import partial_ratio
from src.preprocessing.arabic_normalizer import normalise_hadith_text
from src.stage2.config import GRADE_SCORE_MAP, DEFAULT_GRADE_SCORE

# ── Paths ──
SRC_GRAPH = Path("output/stage2_full/pyg_graph_full.npz")
SRC_NAMES = Path("output/stage2_full/node_names.json")
GRADES_PATH = Path("output/stage2/narrator_grades.json")
DST_GRAPH = Path("output/stage2/pyg_graph_large_graded.npz")
DST_NAMES = Path("output/stage2/node_names.json")

# Load graph
print("Loading full KG...")
graph = np.load(SRC_GRAPH)
x = graph["x"]
edge_index = graph["edge_index"]
edge_attr = graph["edge_attr"]

# Load node names
with open(SRC_NAMES, encoding="utf-8") as f:
    id_to_name = json.load(f)

# Load narrator grades
with open(GRADES_PATH, encoding="utf-8") as f:
    grades_data = json.load(f)

print(f"Nodes: {len(id_to_name):,}")
print(f"Graded narrators: {len(grades_data):,}")

# ── Step 1: Build normalized lookup for grades ──
print("\nBuilding normalized grade lookup...")
grade_norm_map = {}
for gname, gd in grades_data.items():
    gnorm = normalise_hadith_text(gname, for_matching=True)
    # Keep the shortest name variant (canonical)
    if gnorm not in grade_norm_map or len(gname) < len(grade_norm_map[gnorm][0]):
        grade_norm_map[gnorm] = (gname, gd)

# ── Step 2: Match nodes to grades ──
print("Matching narrators to grades...")
attached = 0
grade_dist = Counter()
grade_norms_list = list(grade_norm_map.keys())
nodes_list = list(id_to_name.items())
unmatched_nodes = []

for node_id_str, name in nodes_list:
    node_id = int(node_id_str)
    idx = node_id - 1
    name_norm = normalise_hadith_text(name, for_matching=True)

    gd = None
    if name_norm in grade_norm_map:
        gd = grade_norm_map[name_norm][1]
    else:
        unmatched_nodes.append((node_id_str, name, name_norm, idx))

    if gd and gd.get("grade"):
        grade_label = gd["grade"]
        grade_score = GRADE_SCORE_MAP.get(grade_label, DEFAULT_GRADE_SCORE)
        attached += 1
        grade_dist[grade_label] += 1
        x[idx, 0] = grade_score
        x[idx, 5] = 1.0  # has_grade

print(f"  Exact matches: {attached:,}/{len(nodes_list):,}")

# ── Step 3: Fuzzy match remaining ──
if unmatched_nodes:
    print(f"\nFuzzy matching {len(unmatched_nodes):,} unmatched nodes...")
    fuzzy_matched = 0
    for node_id_str, name, name_norm, idx in unmatched_nodes:
        best = None
        best_score = 0
        for gnorm, (gname, gd) in grade_norm_map.items():
            score = partial_ratio(name_norm, gnorm)
            if score > best_score and score >= 85:
                best_score = score
                best = gd

        if best and best.get("grade"):
            grade_label = best["grade"]
            grade_score = GRADE_SCORE_MAP.get(grade_label, DEFAULT_GRADE_SCORE)
            fuzzy_matched += 1
            attached += 1
            grade_dist[grade_label] += 1
            x[idx, 0] = grade_score
            x[idx, 5] = 1.0

    print(f"  Fuzzy matches: {fuzzy_matched:,}")

print(f"\nTotal attached: {attached:,}/{len(id_to_name):,} ({attached/len(id_to_name)*100:.2f}%)")

# ── Step 4: Copy to stage2 and save ──
print("\nCopying to stage2 paths...")
np.savez(DST_GRAPH, x=x, edge_index=edge_index, edge_attr=edge_attr)
print(f"  Saved: {DST_GRAPH}")

# Copy node names too
with open(DST_NAMES, "w", encoding="utf-8") as f:
    json.dump(id_to_name, f, ensure_ascii=False)
print(f"  Saved: {DST_NAMES}")

print(f"\nGrade scores: min={x[:,0].min():.3f}, max={x[:,0].max():.3f}, mean={x[:,0].mean():.3f}")
print(f"Grade distribution:")
for g, c in grade_dist.most_common(10):
    print(f"  {g:20s}: {c:>6}")
