import json, numpy as np

# Graph metadata
with open(r"c:\Projects\FYP Project\output\stage2\graph_metadata.json", encoding="utf-8") as f:
    meta = json.load(f)
print("Graph metadata:", meta)

# KG stats
g = np.load(r"c:\Projects\FYP Project\output\stage2\pyg_graph_large.npz")
n_nodes = g["x"].shape[0]
n_edges = g["edge_index"].shape[1]
print(f"KG: {n_nodes} nodes, {n_edges} edges")

g2 = np.load(r"c:\Projects\FYP Project\output\stage2\pyg_graph_large_graded.npz")
print(f"Graded KG: {g2['x'].shape[0]} nodes")

# Encyclopedia
with open(r"c:\Projects\FYP Project\output\shamela\narrator_encyclopedia.json", encoding="utf-8") as f:
    enc = json.load(f)
print(f"Encyclopedia: {len(enc)} entries")

# Matches
with open(r"c:\Projects\FYP Project\output\shamela\narrator_matches.json", encoding="utf-8") as f:
    matches = json.load(f)
exact = sum(1 for v in matches.values() if v.get("status","").startswith("exact"))
fuzzy = sum(1 for v in matches.values() if v.get("status","").startswith("fuzzy"))
nf = sum(1 for v in matches.values() if v.get("status") == "not_found")
print(f"Name matching: {exact} exact, {fuzzy} fuzzy, {nf} not found")

# GNN eval results
with open(r"c:\Projects\FYP Project\output\stage3\comparison_v2.json", encoding="utf-8") as f:
    comp = json.load(f)
print(f"\nGNN comparison: {json.dumps(comp, indent=2)}")
