"""Update LIST OF TABLES and LIST OF FIGURES in the full report DOCX."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from docx import Document
from docx.shared import Pt
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph

SRC = r"c:\Projects\FYP Project\Report Drafts\new_report_draft_md\fyp1_full_report.docx"
DST = r"c:\Projects\FYP Project\Report Drafts\new_report_draft_md\fyp1_full_report_v2.docx"

doc = Document(SRC)


def delete_para(para):
    p = para._element
    p.getparent().remove(p)


def insert_para_after(para, text, style_name="Normal"):
    new_p = OxmlElement("w:p")
    para._element.addnext(new_p)
    new_para = Paragraph(new_p, para.part)
    new_para.text = text
    new_para.style = doc.styles[style_name]
    return new_para


# ── Locate boundaries ──
lot_idx = lof_idx = ch1_idx = None
for i, p in enumerate(doc.paragraphs):
    t = p.text.strip()
    if "LIST OF TABLES" in t and "LIST OF FIGURES" not in t:
        lot_idx = i
    if "LIST OF FIGURES" in t:
        lof_idx = i
    if t.startswith("CHAPTER 1"):
        ch1_idx = i

print(f"LOT heading: para {lot_idx}")
print(f"LOF heading: para {lof_idx}")
print(f"Ch1 heading: para {ch1_idx}")

# ── Delete old LOT entries ──
to_delete = []
for i in range(lot_idx + 1, lof_idx):
    to_delete.append(doc.paragraphs[i])
for p in reversed(to_delete):
    delete_para(p)
print(f"Deleted {len(to_delete)} old LOT entries")

# ── Insert new LOT entries ──
lot_heading = doc.paragraphs[lot_idx]
tables = [
    ("Table 1:", "Example of Narrator Entity for the Companion Abu Hurayra", "13"),
    ("Table 2:", "Attributes in a Rich Narrator Node", "14"),
    ("Table 3:", "Example of Term Weighting", "16"),
    ("Table 4:", "Standardised Modifier Range for Risk Cases", "18"),
    ("Table 5:", "Purposes of Metrics Used in this Project", "19"),
    ("Table 6:", "Milestones and Deliverables by Week", "22"),
    ("Table 7:", "Tools, Libraries, and their Purposes by Category", "24"),
    ("Table 3.1:", "Dimensions of the Constructed Relational Knowledge Graph", "—"),
    ("Table 3.2:", "Semantic Score Mapping and Classical Arabic Synonyms for Biographical Tiers", "—"),
    ("Table 3.3:", "Risk Modifier Multipliers with Description", "—"),
    ("Table 5.1:", "Statistics of the Extracted Narrator Encyclopaedia", "—"),
    ("Table 5.2:", "Alignment Performance on Evaluation Sets", "—"),
    ("Table 5.3:", "Per-Class Classification Performance on Test Split", "—"),
    ("Table 5.4:", "Distribution of Resolved Narrator Names", "—"),
    ("Table 5.5:", "Structural Topology Comparison of Graph Variants", "—"),
    ("Table 5.6:", "GAE Link Prediction Performance", "—"),
    ("Table 5.7:", "Distribution of Narrator Reliability Scores across the 262K Graph", "—"),
    ("Table 5.8:", "Soft Aggregation Parameter Optimisation Results", "—"),
    ("Table 5.9:", "Scoring Pipeline Component Ablation Study Results", "—"),
    ("Table 5.10:", "Performance Breakdown across the Six Books", "—"),
    ("Table 5.11:", "Grade Recall on Balanced 3,022 Dataset", "—"),
    ("Table 5.12:", "Geospatial Violations across Datasets", "—"),
]
prev = lot_heading
for num, desc, pg in tables:
    full = f"{num} {desc}\t{pg}"
    prev = insert_para_after(prev, full)
    for run in prev.runs:
        run.font.size = Pt(11)
        run.font.name = "Tahoma"
print(f"Inserted {len(tables)} LOT entries")

# ── Re-find LOF and Ch1 after deletions ──
new_lof_idx = new_ch1_idx = None
for i, p in enumerate(doc.paragraphs):
    t = p.text.strip()
    if "LIST OF FIGURES" in t:
        new_lof_idx = i
    if t.startswith("CHAPTER 1"):
        new_ch1_idx = i

# ── Delete old LOF entries ──
to_delete = []
for i in range(new_lof_idx + 1, new_ch1_idx):
    to_delete.append(doc.paragraphs[i])
for p in reversed(to_delete):
    delete_para(p)
print(f"Deleted {len(to_delete)} old LOF entries")

# ── Insert new LOF entries ──
new_lof_heading = None
for p in doc.paragraphs:
    if "LIST OF FIGURES" in p.text.strip():
        new_lof_heading = p
        break

figures = [
    ("Figure 1.1:", "Graph Visualisation of a Sanad", "6"),
    ("Figure 2.1:", "Example of a Rich Narrator Node", "15"),
    ("Figure 2.2:", "Enriched Sanad Knowledge Graph Mock-up", "16"),
    ("Figure 2.3:", "The Multi-Modal Framework Flowchart Summarising the Methodology", "20"),
    ("Figure 2.4:", "The Flowchart that Summarises the Methodology of this Project", "21"),
    ("Figure 3.1:", "Bar Chart of Number of Entries by Explicit Grade", "—"),
    ("Figure 3.2:", "Biography Text Length Distribution", "—"),
    ("Figure 3.3:", "Alias Form Density per Canonical Narrator", "—"),
    ("Figure 3.4:", "Degree Distribution of the Graded Transmission Subgraph", "—"),
    ("Figure 3.5:", "Complementary CDF of Node Degree", "—"),
    ("Figure 3.6:", "Attribute Coverage in the Graded Subgraph", "—"),
    ("Figure 3.7:", "Narrator Distribution Across Tabaqa Generations", "—"),
    ("Figure 3.8:", "Narrators per Generation with Major Compiler Lifespans", "—"),
    ("Figure 3.9:", "Major Hadith Transmission Centres", "—"),
    ("Figure 3.10:", "Flowchart of the Multi-Modal Hadith Authenticity Verification Pipeline", "—"),
    ("Figure 3.11:", "Two-Tiered Algorithmic Alignment Architecture", "—"),
]
# Note: Fig 3.2 (Biography Text Length) and Fig 3.11 (Two-Tiered Alignment) resolve
# the numbering conflict in the original doc where both were labelled 3.2.
# The user should verify these numbers match their figure captions.

prev = new_lof_heading
for num, desc, pg in figures:
    full = f"{num} {desc}\t{pg}"
    prev = insert_para_after(prev, full)
    for run in prev.runs:
        run.font.size = Pt(11)
        run.font.name = "Tahoma"
print(f"Inserted {len(figures)} LOF entries")

# ── Save ──
doc.save(DST)
print(f"\nSaved: {DST}")
