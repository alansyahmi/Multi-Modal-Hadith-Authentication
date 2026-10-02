"""
Add geospatial/geographic mentions to Chapter 4 in v4 document.
"""
import docx
from docx.shared import Pt, Inches
from lxml import etree

SRC = r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v4.docx"
DST = r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v4.docx"

doc = docx.Document(SRC)
ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

# Find Chapter 4 paragraphs by scanning
ch4_start = None
for i, p in enumerate(doc.paragraphs):
    if p.text.strip() == "CHAPTER 4":
        ch4_start = i
        break

print(f"Chapter 4 starts at [{ch4_start}]")

# ── 1. Experiment 5: Graph Topology (para [418] area) ──
# Find paragraph: "Node features include log-transformed appearance count..."
for i in range(ch4_start, ch4_start + 70):
    if "log-transformed appearance count" in doc.paragraphs[i].text:
        exp5_idx = i
        print(f"Experiment 5 body at [{i}]")
        break

# Find Experiment 6 paragraph: "Three variants were compared..."
for i in range(ch4_start, ch4_start + 70):
    if "Three variants were compared" in doc.paragraphs[i].text:
        exp6_idx = i
        print(f"Experiment 6 body at [{i}]")
        break

# ── Edit Exp 5: add geographic region flag to node features list ──
p5 = doc.paragraphs[exp5_idx]
# Clear existing runs and rebuild with geographic mention
old_text = p5.text
new_text_5 = (
    "The graph is built by walking through Sanadset rows, pulling narrator names from each chain, and creating directed "
    "edges to represent teacher-student relationships (teacher → student). Node features include: log-transformed appearance "
    "count, in-degree, out-degree, average chain position, the Ibn Hajar grade score, and a binary flag indicating whether "
    "a known geographic region ({italic:nisba}) can be associated with the narrator for geospatial consistency verification. "
    "Edges carry positional data and frequency counts. As described in Section 3.4.2.1, the geographic flag enables the GNN "
    "to learn whether teacher-student pairs share a plausible regional overlap, serving as a structural check against "
    "anachronistic or geographically impossible transmission claims."
)

# Replace runs
for r in p5.runs:
    r._element.getparent().remove(r._element)

# Parse {italic:...} markup
import re
parts = re.split(r'\{italic:([^}]+)\}', new_text_5)
for i, segment in enumerate(parts):
    if not segment:
        continue
    is_italic = (i % 2 == 1)
    run = p5.add_run(segment)
    if is_italic:
        run.italic = True

print(f"Exp 5 updated: {new_text_5[:80]}...")

# ── Edit Exp 6: mention geographic flag among structural features ──
p6 = doc.paragraphs[exp6_idx]
old_text_6 = p6.text

# The existing text says "8 structural features" — but the node features list
# from Chapter 3 includes geographic flag. Let's update to note this.
new_text_6 = (
    "To answer the experimental question, a two-layer GCN is built in PyTorch Geometric. The model reads node features "
    "and outputs a continuous reliability score in [0, 1] for each narrator. Three variants were compared: (a) graph-only "
    "with 8 structural features (degree centrality, appearance count, chain position, grade score, and a binary geographic "
    "region flag); (b) graph + BERT with structural features plus the 128-dimensional PCA-reduced BERT embeddings; and "
    "(c) graph + lexicon with structural features plus 7 lexicon sentiment features for comparison."
)

for r in p6.runs:
    r._element.getparent().remove(r._element)
run = p6.add_run(new_text_6)

print(f"Exp 6 updated: {new_text_6[:80]}...")

# ── Save ──
doc.save(DST)
print(f"\nSaved to {DST}")
