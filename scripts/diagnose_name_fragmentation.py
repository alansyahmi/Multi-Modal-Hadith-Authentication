"""Diagnose and fix narrator name fragmentation issues:
1. "Abihi" (his father) treated as a person
2. Abu Hurayra split into Abu / Abi / Aba + Hurayra fragments
3. Kunya prefixes not unified (Abu/Abi/Aba)
"""
import json, re, ast, sys
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

# ── Load data ──
df = pd.read_csv('output/stage1/sanadset_cleaned.csv', nrows=50000)
with open('output/stage2/node_names.json', encoding='utf-8') as f:
    id_to_name = json.load(f)

# ── 1. Find all "Abihi" (his father) occurrences in chains ──
print("="*60)
print("ISSUE 1: 'Abihi' (his father) treated as narrator")
print("="*60)

abihi_chains = []
for _, row in df.iterrows():
    sanad = str(row.get('Sanad', ''))
    try:
        chain = ast.literal_eval(sanad)
    except:
        nars = re.findall(r'<NAR>(.*?)</NAR>', sanad)
        chain = [n.strip() for n in nars] if nars else []

    for i, name in enumerate(chain):
        clean = re.sub(r'[ً-ٌٍ-َّْٰ]', '', name.strip())
        if clean in ['ابيه', 'أبيه', 'ابية', 'أبية']:
            abihi_chains.append({
                'name': name,
                'position': i,
                'chain': chain,
                'next_name': chain[i+1] if i+1 < len(chain) else None,
                'prev_name': chain[i-1] if i > 0 else None,
            })

print(f"  Occurrences: {len(abihi_chains)}")
for c in abihi_chains[:8]:
    print(f"  Chain: ... → {c['prev_name']} → [{c['name']}] → {c['next_name']} → ...")

# ── 2. Find Abu Hurayra variants ──
print(f"\n{'='*60}")
print("ISSUE 2: Abu Hurayra fragmented into variants")
print("="*60)

hurayra_vars = Counter()
for _, row in df.iterrows():
    sanad = str(row.get('Sanad', ''))
    try:
        chain = ast.literal_eval(sanad)
    except:
        nars = re.findall(r'<NAR>(.*?)</NAR>', sanad)
        chain = [n.strip() for n in nars] if nars else []

    for name in chain:
        if 'هريرة' in name or 'هريره' in name:
            hurayra_vars[name.strip()] += 1

print(f"  Unique variants: {len(hurayra_vars)}")
for variant, count in hurayra_vars.most_common(15):
    print(f"  [{variant:<40s}] appears {count:>5d} times")

# ── 3. Find stand-alone "Abu" / "Abi" fragments ──
print(f"\n{'='*60}")
print("ISSUE 3: Standalone 'Abu' / 'Abi' fragments")
print("="*60)

standalone = Counter()
for _, row in df.iterrows():
    sanad = str(row.get('Sanad', ''))
    try:
        chain = ast.literal_eval(sanad)
    except:
        nars = re.findall(r'<NAR>(.*?)</NAR>', sanad)
        chain = [n.strip() for n in nars] if nars else []

    for name in chain:
        clean = re.sub(r'[ً-ٌٍ-َّْٰ]', '', name.strip())
        if clean in ['ابو', 'أبو', 'ابي', 'أبي', 'ابا', 'أبا']:
            standalone[name.strip()] += 1

print(f"  Occurrences: {sum(standalone.values())}")
for variant, count in standalone.most_common(10):
    print(f"  [{variant}] appears {count} times")

# ── 4. Check predictions for these bad names ──
print(f"\n{'='*60}")
print("ISSUE 4: Predictions for bogus names")
print("="*60)

preds = pd.read_csv('output/stage3/narrator_predictions.csv')
# Build name→pred lookup
name_to_pred = {}
for _, row in preds.iterrows():
    nid = str(int(row['node_id']))
    if nid in id_to_name:
        name = id_to_name[nid]
        clean = re.sub(r'[ً-ٌٍ-َّْٰ]', '', name)
        name_to_pred[clean] = float(row['predicted_score'])

problem_names = ['ابيه', 'أبيه', 'ابي', 'أبي', 'وأبي هريرة', 'أبا هريرة']
for name in problem_names:
    if name in name_to_pred:
        print(f"  {name}: {name_to_pred[name]:.3f}")
    else:
        print(f"  {name}: NOT FOUND in predictions")

# ── Also check: what does the graph node for "Abihi" look like?
print(f"\n{'='*60}")
print("ISSUE 5: Graph node details for Abihi")
print("="*60)

import numpy as np
graph = np.load('output/stage2/pyg_graph_large_graded.npz')
x = graph['x']

for nid_str, name in id_to_name.items():
    clean = re.sub(r'[ً-ٌٍ-َّْٰ]', '', name.strip())
    if clean in ['ابيه', 'أبيه', 'ابي', 'أبي']:
        nid = int(nid_str) - 1
        print(f"  Node {nid_str}: [{name}]")
        print(f"    Features: grade={x[nid,0]:.3f} appearances={x[nid,1]:.3f} "
              f"in_deg={x[nid,2]:.3f} out_deg={x[nid,3]:.3f} "
              f"avg_pos={x[nid,4]:.3f}")
        if name_to_pred.get(clean):
            print(f"    Predicted: {name_to_pred[clean]:.3f}")
