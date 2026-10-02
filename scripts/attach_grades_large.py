"""Attach Ibn Hajar grades to the large graph nodes."""
import json, sys, numpy as np
from pathlib import Path
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rapidfuzz.fuzz import partial_ratio
from src.preprocessing.arabic_normalizer import normalise_hadith_text
from src.stage2.config import GRADE_SCORE_MAP, DEFAULT_GRADE_SCORE

# Load graph
graph = np.load(r"output/stage2/pyg_graph_large.npz")
x = graph["x"]
edge_index = graph["edge_index"]
edge_attr = graph["edge_attr"]

# Load node names
with open(r"output/stage2/node_names.json", encoding="utf-8") as f:
    id_to_name = json.load(f)

# Load narrator grades
with open(r"output/stage2/narrator_grades.json", encoding="utf-8") as f:
    grades_data = json.load(f)

print(f"Nodes in graph: {len(id_to_name):,}")
print(f"Graded narrators in DB: {len(grades_data):,}")

# Match
attached = 0
grade_dist = Counter()

for node_id_str, name in id_to_name.items():
    node_id = int(node_id_str)
    idx = node_id - 1
    name_norm = normalise_hadith_text(name, for_matching=True)

    grade_label = ""
    grade_score = DEFAULT_GRADE_SCORE

    # Try exact match
    if name in grades_data:
        gd = grades_data[name]
    elif name_norm in grades_data:
        gd = grades_data[name_norm]
    else:
        # Fuzzy match
        best = None
        best_score = 0
        for gname, gd in grades_data.items():
            gname_norm = normalise_hadith_text(gname, for_matching=True)
            score = partial_ratio(name_norm, gname_norm)
            if score > best_score and score >= 85:
                best_score = score
                best = gd
        gd = best

    if gd and gd.get("grade"):
        grade_label = gd["grade"]
        grade_score = GRADE_SCORE_MAP.get(grade_label, DEFAULT_GRADE_SCORE)
        attached += 1
        grade_dist[grade_label] += 1
        x[idx, 0] = grade_score
        x[idx, 5] = 1.0  # has_grade
        if gd.get("death_info"):
            x[idx, 6] = 1.0
        if gd.get("tabaqa"):
            x[idx, 7] = 1.0

print(f"\nAttached grades: {attached:,}/{len(id_to_name):,} ({attached/len(id_to_name)*100:.1f}%)")

# Grade distribution
print(f"\nGrade distribution ({len(grade_dist)} types):")
for g, c in grade_dist.most_common(15):
    print(f"  {g:20s}: {c:>6}")

# Save updated graph
out = Path(r"output/stage2/pyg_graph_large_graded.npz")
np.savez(out, x=x, edge_index=edge_index, edge_attr=edge_attr)
print(f"\nSaved: {out}")
print(f"  x shape: {x.shape}")
print(f"  Grade scores: min={x[:,0].min():.3f}, max={x[:,0].max():.3f}, mean={x[:,0].mean():.3f}")
