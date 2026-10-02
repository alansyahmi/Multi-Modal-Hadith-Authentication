import json, numpy as np, sys
sys.stdout.reconfigure(encoding="utf-8")

f = open(r"c:\Projects\FYP Project\Report Drafts\_results_data.txt", "w", encoding="utf-8")

# Alignment
with open(r"c:\Projects\FYP Project\output\stage1\alignment_results.json", encoding="utf-8") as jf:
    data = json.load(jf)
matched = sum(1 for d in data if d.get("is_match"))
quar = sum(1 for d in data if d.get("quarantine"))
total = len(data)
f.write(f"Alignment: {total} queries, {matched} matched ({matched/total*100:.1f}%), {quar} quarantined\n")

# Graph
g = np.load(r"c:\Projects\FYP Project\output\stage2\pyg_graph_large.npz")
n_nodes = g["x"].shape[0]
n_edges = g["edge_index"].shape[1]
f.write(f"KG: {n_nodes} nodes, {n_edges} edges\n")

# Encyclopedia
with open(r"c:\Projects\FYP Project\output\shamela\narrator_encyclopedia.json", encoding="utf-8") as jf:
    enc = json.load(jf)
f.write(f"Encyclopedia: {len(enc)} entries\n")

# Matches
with open(r"c:\Projects\FYP Project\output\shamela\narrator_matches.json", encoding="utf-8") as jf:
    matches = json.load(jf)
exact = sum(1 for v in matches.values() if v.get("status","").startswith("exact"))
fuzzy = sum(1 for v in matches.values() if v.get("status","").startswith("fuzzy"))
nf = sum(1 for v in matches.values() if v.get("status") == "not_found")
f.write(f"Name matching: {exact} exact, {fuzzy} fuzzy, {nf} not found\n")
f.write(f"Match rate: {(exact+fuzzy)/(exact+fuzzy+nf)*100:.1f}%\n")

# GNN comparison
with open(r"c:\Projects\FYP Project\output\stage3\comparison_v2.json", encoding="utf-8") as jf:
    comp = json.load(jf)
f.write(f"\nGNN results (graph-only):\n")
for k, v in comp.get("graph_only", {}).items():
    if not isinstance(v, dict):
        f.write(f"  {k}: {v}\n")
f.write(f"\nGNN results (multimodal):\n")
for k, v in comp.get("multimodal", {}).items():
    if not isinstance(v, dict):
        f.write(f"  {k}: {v}\n")

f.close()
print("Written to Report Drafts/_results_data.txt")
