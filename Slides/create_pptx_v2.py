#!/usr/bin/env python3
"""
FYP1 Defense Presentation — PPTX v2
Mirrors the interactive HTML presentation in content, layout, and data coverage.
36 slides covering all 6 chapters with updated content.
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ── Palette ──────────────────────────────────────────────
DARK_NAVY   = RGBColor(0x1B, 0x2A, 0x4A)
MEDIUM_BLUE = RGBColor(0x2D, 0x5F, 0x8A)
LIGHT_BLUE  = RGBColor(0x3A, 0x7C, 0xBD)
GOLD        = RGBColor(0xD4, 0xA8, 0x3C)
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
OFF_WHITE   = RGBColor(0xF5, 0xF0, 0xEB)
DARK_TEXT    = RGBColor(0x2C, 0x2C, 0x2C)
LIGHT_GRAY  = RGBColor(0x6B, 0x7B, 0x8D)
TABLE_HEAD  = RGBColor(0x1B, 0x2A, 0x4A)
TABLE_ALT   = RGBColor(0xE8, 0xEF, 0xF5)
GREEN_OK    = RGBColor(0x28, 0xA7, 0x45)

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
W = prs.slide_width
H = prs.slide_height

# ── Helpers ──────────────────────────────────────────────
def add_bg(slide, color=OFF_WHITE):
    bg = slide.background; fill = bg.fill; fill.solid(); fill.fore_color.rgb = color

def add_shape(slide, l, t, w, h, c):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    shp.fill.solid(); shp.fill.fore_color.rgb = c; shp.line.fill.background(); return shp

def tb(slide, l, t, w, h, text="", sz=18, c=DARK_TEXT, b=False, al=PP_ALIGN.LEFT, font_name="Calibri", ls=1.15):
    bx = slide.shapes.add_textbox(l, t, w, h)
    tf = bx.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = text
    p.font.size = Pt(sz); p.font.color.rgb = c; p.font.bold = b; p.font.name = font_name
    p.alignment = al; p.space_after = Pt(2)
    if ls: p.line_spacing = ls
    return tf

def rtb(slide, l, t, w, h):
    bx = slide.shapes.add_textbox(l, t, w, h)
    tf = bx.text_frame; tf.word_wrap = True; return tf

def ap(tf, text, sz=16, c=DARK_TEXT, b=False, al=PP_ALIGN.LEFT, font_name="Calibri", sa=4, ls=1.2):
    p = tf.add_paragraph(); p.text = text
    p.font.size = Pt(sz); p.font.color.rgb = c; p.font.bold = b; p.font.name = font_name
    p.alignment = al; p.space_after = Pt(sa)
    if ls: p.line_spacing = ls; return p

def bl(tf, items, sz=14, c=DARK_TEXT, ind=0, sa=3, ls=1.2):
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = it; p.font.size = Pt(sz); p.font.color.rgb = c; p.font.name = "Calibri"
        p.space_after = Pt(sa); p.level = ind
        if ls: p.line_spacing = ls

def sp(tf, text, sz=16, c=DARK_TEXT, b=False, al=PP_ALIGN.LEFT, font_name="Calibri", sa=4, ls=1.2):
    p = tf.paragraphs[0]; p.text = text
    p.font.size = Pt(sz); p.font.color.rgb = c; p.font.bold = b; p.font.name = font_name
    p.alignment = al; p.space_after = Pt(sa)
    if ls: p.line_spacing = ls; return p

# ── Section Header ───────────────────────────────────────
def sec(title, sub=""):
    s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s, DARK_NAVY)
    add_shape(s, Inches(1.2), Inches(3.3), Inches(10.9), Pt(6), GOLD)
    tb(s, Inches(1.2), Inches(1.8), Inches(10.9), Inches(1.5), title, 40, WHITE, True, PP_ALIGN.LEFT, "Calibri Light")
    if sub:
        tb(s, Inches(1.2), Inches(3.6), Inches(10.9), Inches(1.0), sub, 18, LIGHT_GRAY, font_name="Calibri Light")
    add_shape(s, Inches(1.2), Inches(6.5), Inches(10.9), Pt(1), LIGHT_BLUE)

def cs(title, sub=""):
    s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s, WHITE)
    add_shape(s, 0, 0, W, Inches(1.15), DARK_NAVY)
    add_shape(s, Inches(0.8), Inches(1.15), Inches(3.5), Pt(4), GOLD)
    tb(s, Inches(0.8), Inches(0.2), Inches(11.5), Inches(0.85), title, 28, WHITE, True, font_name="Calibri Light")
    if sub:
        tb(s, Inches(0.8), Inches(1.3), Inches(11.5), Inches(0.5), sub, 13, LIGHT_GRAY, font_name="Calibri")
    return s

def two_col(s, left_items, right_items, lt="", rt=""):
    lf = rtb(s, Inches(0.8), Inches(1.9), Inches(5.6), Inches(5.2))
    if lt: sp(lf, lt, 17, DARK_NAVY, True, sa=6)
    bl(lf, left_items, 13)
    rf = rtb(s, Inches(6.9), Inches(1.9), Inches(5.6), Inches(5.2))
    if rt: sp(rf, rt, 17, DARK_NAVY, True, sa=6)
    bl(rf, right_items, 13)

def tbl(s, l, t, w, h, headers, rows, cw=None):
    r = len(rows)+1; c = len(headers)
    ts = s.shapes.add_table(r, c, l, t, w, h)
    tab = ts.table
    if cw:
        for i, v in enumerate(cw): tab.columns[i].width = v
    for ri in range(r):
        for ci in range(c):
            cell = tab.cell(ri, ci)
            cell.text = headers[ci] if ri == 0 else rows[ri-1][ci] if ri-1 < len(rows) and ci < len(rows[ri-1]) else ""
            for para in cell.text_frame.paragraphs:
                para.font.size = Pt(11); para.font.name = "Calibri"; para.alignment = PP_ALIGN.CENTER
                if ri == 0: para.font.color.rgb = WHITE; para.font.bold = True
                else: para.font.color.rgb = DARK_TEXT
            if ri == 0: cell.fill.solid(); cell.fill.fore_color.rgb = TABLE_HEAD
            elif ri % 2 == 0: cell.fill.solid(); cell.fill.fore_color.rgb = TABLE_ALT
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    return tab

# ══════════════════════════════════════════════════════════
# SLIDE 1 — TITLE
# ══════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s, DARK_NAVY)
add_shape(s, Inches(1.0), Inches(1.5), Inches(11.3), Pt(5), GOLD)
tb(s, Inches(1.0), Inches(0.6), Inches(11.3), Inches(0.5), "FINAL YEAR PROJECT 1", 14, GOLD, True)
tb(s, Inches(1.0), Inches(1.8), Inches(11.3), Inches(1.8),
   "A Multi-Modal Framework for\nQuantifying Narrator Reliability\nin ʿIlm ar-Rijal", 36, WHITE, True, font_name="Calibri Light", ls=1.15)
tb(s, Inches(1.0), Inches(3.9), Inches(11.3), Inches(0.7),
   "Using Semantic Sentiment Analysis and Graph Neural Networks", 20, LIGHT_BLUE, font_name="Calibri Light")
add_shape(s, Inches(1.0), Inches(4.8), Inches(11.3), Pt(1), RGBColor(0x3A,0x5A,0x7A))
tb(s, Inches(1.0), Inches(5.1), Inches(5.5), Inches(0.5),
   "Alan Syahmi Bin Sanih @ Sani  |  BI23110344", 16, RGBColor(0xCC,0xCC,0xCC))
tb(s, Inches(1.0), Inches(5.6), Inches(5.5), Inches(0.5),
   "Supervisor: Dr. Khalifa Chekima", 16, RGBColor(0xCC,0xCC,0xCC))
tb(s, Inches(1.0), Inches(6.4), Inches(11.3), Inches(0.5),
   "Faculty of Computing and Informatics  |  Universiti Malaysia Sabah", 13, LIGHT_GRAY)

# ══════════════════════════════════════════════════════════
# SLIDE 2 — OUTLINE
# ══════════════════════════════════════════════════════════
s = cs("Presentation Outline")
tf = rtb(s, Inches(1.0), Inches(1.9), Inches(11.3), Inches(5.0))
items = [
    "1.  Introduction — Background, Problem Statements, Objectives & Scope",
    "2.  Literature Review — Hadith Authentication, GNN, Sentiment Quantisation",
    "3.  Project Methodology — Research Design, Pipeline Stages, Datasets (incl. EDA)",
    "4.  Experiment Design — 9 Experiments across 3 Stages",
    "5.  Preliminary Results — Key Findings & Performance Metrics",
    "6.  Conclusion — Limitations & Future Work (FYP2)",
]
for i, item in enumerate(items):
    if i == 0: sp(tf, item, 18, DARK_TEXT, ls=1.5, sa=10)
    else: ap(tf, item, 18, DARK_TEXT, ls=1.5, sa=10)

# ══════════════════════════════════════════════════════════
# CH1: INTRODUCTION
# ══════════════════════════════════════════════════════════
sec("CHAPTER 1", "Introduction")

# 1.1 Overview
s = cs("1.1  Overview", "The Challenge of Digital Hadith Authentication")
tf = rtb(s, Inches(0.8), Inches(1.9), Inches(5.5), Inches(5.0))
sp(tf, "Core Challenge", 18, DARK_NAVY, True, sa=6)
bl(tf, [
    "Hadith authentication via sanad criticism is a cornerstone of Islamic scholarship",
    "ʿIlm ar-Rijal evaluates narrators for moral probity (ʿadāla) & mnemonic precision (ḍabṭ)",
    "Two complementary datasets: Sanadset 650K (chains, no grades) + HuggingFace hadith_datasets (grades, no chains)",
    "The alignment bridges this gap — linking sanads to ground-truth grades",
    "Digital collections are too large for unaided human review — a scalability crisis",
], 14)
tf2 = rtb(s, Inches(6.9), Inches(1.9), Inches(5.5), Inches(5.0))
sp(tf2, "Three Technical Pillars", 18, DARK_NAVY, True, sa=6)
bl(tf2, [
    "Arabic NER — Transformer-based (AraBERT) extraction & disambiguation",
    "Graph Neural Networks — Modelling structural & temporal integrity of isnād chains",
    "Semantic Sentiment — Translating qualitative evaluations to continuous reliability index",
    "",
    "Vision: A decision-support tool grounded in classical Jarḥ wa Taʿdīl principles",
], 14)

# 1.2 Problem Background
s = cs("1.2  Problem Background", "Why Automation is Needed")
two_col(s, [
    "Traditional Bottlenecks:",
    "• Manual cross-referencing across 12+ volumes of biographical literature",
    "• Expert-dependent process — resistant to scale",
    "• Orthographic ambiguity in classical Arabic consonantal script",
    "• Minor name variations → narrator misidentification",
    "• Identifying a narrator ≠ evaluating their trustworthiness (Ahmad et al., 2025)",
], [
    "Why Now? — Enabling Factors:",
    "• Transformer models (AraBERT) achieve high accuracy for narrator disambiguation",
    "• GNN frameworks learn structural credibility from transmission networks",
    "• Feasibility confirmed: ML can assist isnād analysis (Usman, 2025)",
    "• Digital Hadith circulation outpaces verification capacity (Mosa, 2025)",
    "• Urgent: no existing framework combines NER + KG + GNN + sentiment scoring",
])

# 1.3 Problem Statements
s = cs("1.3  Problem Statements", "Two Critical Gaps in Current Research")
tf = rtb(s, Inches(0.8), Inches(1.9), Inches(5.5), Inches(5.0))
sp(tf, "PS1: Lack of Quantified Reliability Frameworks", 17, DARK_NAVY, True, sa=6)
bl(tf, [
    "Current NER models extract names but cannot judge narrator standing",
    "No existing framework maps Jarḥ wa Taʿdīl → standardised reliability index",
], 13)
tf2 = rtb(s, Inches(6.9), Inches(1.9), Inches(5.5), Inches(5.0))
sp(tf2, "PS2: Graph without Reliability Scores", 17, DARK_NAVY, True, sa=6)
bl(tf2, [
    "Transformer models extract chains as graphs (Mosa, 2025; Oumer, 2025)",
    "But stop at graph construction — no reliability scores assigned",
    "No use of actual Jarḥ wa Taʿdīl terms to judge trustworthiness",
    "This project adds quantitative reliability index + GNN structural analysis",
], 13)

# 1.4 Objectives
s = cs("1.4  Project Objectives")
tf = rtb(s, Inches(0.8), Inches(1.9), Inches(11.5), Inches(5.0))
sp(tf, "Three Primary Objectives", 20, DARK_NAVY, True, sa=10)
for lbl, txt in [("O1", "Develop an NLP classification model mapping qualitative scholarly judgments to a standardised quantitative reliability index (0.0 – 1.0)"),
                 ("O2", "Construct a Knowledge Graph of narrator relationships and investigate structural, temporal, and geographical features for chain plausibility"),
                 ("O3", "Evaluate framework accuracy by comparing automated scores against chain-level Hadith gradings (Ṣaḥīḥ, Ḥasan, Ḍaʿīf)")]:
    ap(tf, f"{lbl}:  {txt}", 15, DARK_TEXT, ls=1.3, sa=12)

# 1.5 Scope
s = cs("1.5  Project Scope", "Data Sources, Scale, Technical & Domain Boundaries")
two_col(s, [
    "Data Sources:",
    "• Shamela Digital Library (Taqrīb at-Tahdhīb, Tahdhīb al-Kamāl)",
    "• 15,448 narrator biographies from 36 volumes",
    "• Sanadset 650K — chains only, no grades",
    "• HuggingFace hadith_datasets — 6 graded CSVs from Kutub as-Sitta",
    "• Alignment links Sanadset chains → HuggingFace ground-truth grades",
    "",
    "Out of Scope: Matn analysis, Shīʿa isnād, real-time deployment, non-Arabic commentaries",
], [
    "Scale:",
    "• 262,497 narrator nodes | 987,036 directed edges",
    "• 4 dataset configs: 1,200 & 6,000 aligned Hadith",
    "• Book-balanced & grade-balanced evaluation",
    "",
    "Technical Scope:",
    "• Narrator extraction (NER from classical Arabic)",
    "• Entity disambiguation via Knowledge Graph",
    "• Semantic sentiment → continuous reliability scores",
    "• Geospatial & temporal verification",
    "Stack: Python, PyTorch Geometric, CAMeL Tools, AraBERT, NetworkX",
])

# ══════════════════════════════════════════════════════════
# CH2: LITERATURE REVIEW
# ══════════════════════════════════════════════════════════
sec("CHAPTER 2", "Literature Review")

# 2.1 Hadith Structure
s = cs("2.1  Hadith & Its Structure", "Sanad and Matn")
two_col(s, [
    "Anatomy of a Hadith:",
    "• Sanad (إسناد): The chain of transmission — \"who told whom\"",
    "• Matn (متن): The textual content — saying, action, or approval",
    "• Sanad = metadata layer: reliability, memory, lifespan per node",
    "",
    "Example (Mālik’s Muwaṭṭaʾ):",
    "\"...from Mālik, from Ibn Shihāb, from ʿAlī ibn Ḥusayn, that the Prophet ﷺ said...\"",
], [
    "Chain Visualisation:",
    "• Prophet ﷺ → Companion → Tābiʿī → Atbāʿ at-Tābiʿīn → ... → Compiler",
    "• Each node = narrator with known biography",
    "• Each edge = teacher–student transmission",
    "",
    "Significance:",
    "• Sunna provides operational parameters for Sharīʿa",
    "• Single unreliable narrator downgrades entire Hadith (Kamali, 2014)",
    "• One of oldest systematic source-criticism methodologies",
])

# 2.2–2.8 Combined
s = cs("2.2–2.8  Impact, Authentication & Technical Foundations")
two_col(s, [
    "Impact & Authentication:",
    "• Ṣaḥīḥ/Ḥasan → legally normative, 2nd source after Quran",
    "• Ḍaʿīf/Mawḍūʿ → risk of groundless practices",
    "• 5 conditions for Ṣaḥīḥ: uninterrupted chain, ʿadāla, ḍabṭ, no šāḏ, no ʿilla",
    "• This project addresses conditions 1–3 (sanad structure & reliability)",
    "",
    "ʿIlm ar-Rijal (Jarḥ wa Taʿdīl):",
    "• 5-tier: Thiqa → Ṣadūq → Maqbul → Ḍaʿīf → Matrūk",
    "• 36 volumes of biographical literature",
], [
    "Technical Foundations:",
    "• Arabic NER: CAMeL Tools + AraBERT achieve high F1",
    "• KGs enable entity disambiguation across alias variations",
    "• GNNs model structural integrity of isnād chains (Mahmoud et al., 2024)",
    "• Graph Autoencoders learn 64-dim embeddings via link prediction",
    "",
    "Semantic Sentiment Quantisation:",
    "• AraBERT maps 5 tiers → continuous [0.05 – 0.85]",
    "• Consensus normalisation improves agreement by ~12%",
    "",
    "Research Gap: No existing framework combines ALL elements",
    "in one integrated pipeline — this project is the first",
])

# ══════════════════════════════════════════════════════════
# CH3: METHODOLOGY
# ══════════════════════════════════════════════════════════
sec("CHAPTER 3", "Project Methodology")

# 3.1 Pipeline Architecture
s = cs("3.1  Pipeline Architecture", "Three-Stage Multi-Modal Framework")
for box_data in [
    (Inches(0.5), "STAGE 1: Data Ingestion & Alignment",
     "Sanadset 650K (chains, no grades) aligned with HuggingFace hadith_datasets (grades, no chains)\nTF-IDF + Levenshtein | 15,448 biographies from 36 volumes"),
    (Inches(4.8), "STAGE 2: Knowledge Graph",
     "262,497 narrator nodes | 987,036 directed edges\nAlias & ancestry matching for entity disambiguation"),
    (Inches(9.1), "STAGE 3: Scoring & Evaluation",
     "AraBERT sentiment → continuous scores | GAE link prediction\nSoft aggregation + geospatial/temporal adjustments"),
]:
    add_shape(s, Inches(box_data[0]), Inches(1.8), Inches(3.8), Inches(1.7), DARK_NAVY)
    add_shape(s, Inches(box_data[0]+3.4), Inches(2.3), Inches(0.4), Inches(0.5), GOLD)
    tb(s, Inches(box_data[0]+0.2), Inches(1.9), Inches(3.4), Inches(0.35), box_data[1], 11, GOLD, True)
    tb(s, Inches(box_data[0]+0.2), Inches(2.3), Inches(3.4), Inches(1.0), box_data[2], 11, WHITE, ls=1.2)

two_col(s, [
    "Key Data Sources:",
    "• Tahdhīb al-Kamāl (al-Mizzī) — comprehensive biographies",
    "• Taqrīb at-Tahdhīb (Ibn Hajar) — concise graded summaries",
    "• Sanadset 650K — chains only (no grades)",
    "• HuggingFace hadith_datasets — 6 graded CSVs (grades, no chains)",
], [
    "Technology Stack:",
    "• PyTorch + PyTorch Geometric — GNN framework",
    "• CAMeL Tools + AraBERT — Arabic NLP",
    "• NetworkX + igraph — graph analytics",
    "• Pandas + NumPy + scikit-learn — evaluation",
], rt="", lt="")

# 3.2 Dataset Characteristics — Part 1
s = cs("3.2  Dataset Corpus Characteristics (1/2)", "Pre-Pipeline: Two Complementary Datasets")
# Left: Sanadset table + HuggingFace table
headers_s = ["Property", "Value"]
rows_s = [
    ["Total Hadith", "650,986"],
    ["Unique Books", "956"],
    ["Columns", "Hadith, Book, Num_hadith, Matn, Sanad, Sanad_Length"],
    ["Mean Sanad Length", "4.6 narrators"],
    ["Grade Labels?", "None"],
]
tbl(s, Inches(0.5), Inches(1.9), Inches(5.8), Inches(1.8), headers_s, rows_s)

headers_h = ["Explicit Grade", "Count", "%"]
rows_h = [
    ["nan (Bukhārī + Muslim; Ṣaḥīḥ by consensus)", "15,682", "46.5%"],
    ["Ṣaḥīḥ (Darussalam)", "8,883", "26.3%"],
    ["Ṣaḥīḥ (Al-Albānī)", "3,179", "9.4%"],
    ["Ḥasan (Darussalam)", "2,174", "6.4%"],
    ["Ḍaʿīf (Darussalam + variants)", "2,063", "6.1%"],
    ["Ḍaʿīf (Al-Albānī)", "730", "2.2%"],
    ["Ḥasan (Al-Albānī) + Ḥasan Ṣaḥīḥ", "599", "1.8%"],
    ["Other (Mawḍūʿ, Šāḏ, Munkar, etc.)", "428", "1.3%"],
]
tbl(s, Inches(0.5), Inches(4.0), Inches(5.8), Inches(3.0), headers_h, rows_h)

tf = rtb(s, Inches(6.9), Inches(1.9), Inches(5.5), Inches(1.5))
sp(tf, "Pre-Pipeline Gap", 17, DARK_NAVY, True, sa=6)
bl(tf, [
    "Sanadset 650K: has chains but no grades",
    "HuggingFace hadith_datasets: has grades but no structured sanads",
    "Alignment (TF-IDF + Levenshtein) bridges this gap",
], 13)

tf2 = rtb(s, Inches(6.9), Inches(3.5), Inches(5.5), Inches(2.0))
sp(tf2, "5-Tier Jumhūr Mapping (post-alignment)", 15, DARK_NAVY, True, sa=6)
tbl(s, Inches(6.9), Inches(4.0), Inches(5.5), Inches(2.0),
    ["Tier", "Label", "Score"],
    [["1 (Highest)", "Thiqah (ثقة)", "0.85"],
     ["2", "Ṣadūq (صدوق)", "0.70"],
     ["3 (Middle)", "Maqbul (مقبول)", "0.55"],
     ["4", "Ḍaʿīf (ضعيف)", "0.25"],
     ["5 (Lowest)", "Matrūk (متروك)", "0.05"]])

tb(s, Inches(6.9), Inches(6.2), Inches(5.5), Inches(0.8),
   "Total HF: 33,738 Hadith across 6 books  |  Graded: 18,056  |  Middle tier = Maqbul (not Lā Baʼs)",
   11, LIGHT_GRAY)

# 3.2 Dataset Characteristics — Part 2 (EDA Deep Dive)
s = cs("3.2  Exploratory Data Analysis (2/2)", "NLP, Graph Topology & Cold-Start Deep Dive")

# Left column: NLP + Grade distribution tables
tbl(s, Inches(0.5), Inches(1.9), Inches(5.8), Inches(1.8),
    ["NLP Metric", "Value"],
    [["Total biographies", "15,448"],
     ["Mean text length", "1,098 chars"],
     ["Median text length", "333 chars (skewed)"],
     ["Very short (<100 chars)", "43.3%"],
     ["Total alias forms", "24,545"],
     ["Avg aliases per narrator", "4.02"],
     ["Max aliases (one narrator)", "1,848"]])

tbl(s, Inches(0.5), Inches(4.0), Inches(5.8), Inches(2.2),
    ["Shamela Grade Tier", "Count", "%"],
    [["Thiqah", "3,409", "28.4%"],
     ["Ṣadūq", "2,607", "21.7%"],
     ["Maqbul", "2,088", "17.4%"],
     ["Majhūl (unknown)", "1,157", "9.6%"],
     ["Ḍaʿīf", "851", "7.1%"],
     ["Matrūk", "215", "1.8%"],
     ["Other / unclassified", "1,692", "14.1%"]])

# Right column: Graph topology + Cold-start + Attribute coverage
tbl(s, Inches(6.9), Inches(1.9), Inches(5.5), Inches(1.8),
    ["Graph Metric", "Graded Subgraph", "Full Graph"],
    [["Nodes", "2,172", "262,497"],
     ["Edges", "3,541", "987,036"],
     ["Mean degree", "4.12", "7.5"],
     ["Power-law γ", "1.612", "~2.3"],
     ["Top 1% share", "19.8% edges", "—"]])

tbl(s, Inches(6.9), Inches(4.0), Inches(5.5), Inches(1.2),
    ["Attribute Coverage", "Narrators", "%"],
    [["Grade label", "1,987", "91.5%"],
     ["Ṭabaqa (generation)", "1,915", "88.2%"],
     ["Death year", "1,386", "63.8%"]])

tb(s, Inches(6.9), Inches(5.4), Inches(5.5), Inches(1.5),
   "Cold-Start Problem: 99.17% of 262K narrator nodes have no direct grade label.\n"
   "Only 2,172 nodes (0.83%) are in the graded subgraph.\n"
   "GNN must propagate reliability signals from extremely sparse supervision.\n"
   "Scale-free property (γ≈1.61–2.3): few highly-connected hub narrators dominate.\n"
   "Self-supervised GAE pre-training (link prediction) is critical.",
   11, DARK_TEXT, ls=1.3)

# 3.3–3.4 Stages 1 & 2
s = cs("3.3 – 3.4  Stage 1 & 2: Data Ingestion & Knowledge Graph")
two_col(s, [
    "Stage 1: Data Ingestion & Preprocessing",
    "• Regex parser extracts narrator blocks from 36 volumes",
    "• Structured: name, kunya, nisba, ṭabaqa, grade evaluations",
    "• Two-Tier Alignment (WHY: Sanadset has no grades, HF has no sanads)",
    "•   Tier 1: TF-IDF reduces search space by ~85%",
    "•   Tier 2: Levenshtein exact chain verification",
    "• Output: Narrator Encyclopaedia + aligned dataset (chains + grades)",
    "• Result: 15,448 biographies, 6,000 aligned Hadith",
], [
    "Stage 2: Knowledge Graph Construction",
    "• Nodes: 262,497 unique narrators with attributes",
    "• Edges: 987,036 directed teacher→student transmissions",
    "• Algorithmic Name Mapping: root-based consonantal skeletons",
    "• Levenshtein threshold ≤2 for near-identical names",
    "• Resolves avg 3.2 alias forms per canonical narrator",
    "",
    "4 Dataset Configurations:",
    "• A: 1,200 book-balanced | B: 1,200 grade-balanced",
    "• C: 6,000 book-balanced | D: 6,000 grade-balanced (3,022 unique grades)",
])

# 3.5 Stage 3
s = cs("3.5  Stage 3: Probabilistic Trust Scoring", "Semantic Sentiment + GAE + Soft Aggregation + Risk Modifiers")
two_col(s, [
    "Semantic Sentiment Scores:",
    "• AraBERT classifies → 5-tier Jumhūr labels",
    "• Mapping: Thiqa=0.85, Ṣadūq=0.70, Maqbul=0.55, Ḍaʿīf=0.25, Matrūk=0.05",
    "• Consensus averaging across biographers reduces noise ~12%",
    "",
    "Chain-Level Scoring:",
    "• S_base = α · min(scores) + (1−α) · mean(scores)",
    "• S_chain = S_base × ∏(risk modifiers)",
], [
    "Risk Modifiers:",
    "• Tadlīs (concealment): 0.85",
    "• Unknown narrator: 0.70",
    "• Geospatial gap: 0.80",
    "• Temporal gap: 0.75",
    "",
    "GAE Architecture:",
    "• 2-layer GCN encoder (256 → 64 dims)",
    "• Inner-product decoder, self-supervised (mask 15% edges)",
    "• Output: 64-dim embeddings for structural uncertainty modulation",
])

# Tabaqa Table
s = cs("Ṭabaqa Generation Distribution", "Narrator Distribution Across Generations")
tbl(s, Inches(0.8), Inches(2.0), Inches(11.5), Inches(2.5),
    ["Generation", "Period", "Narrators", "Key Centres"],
    [["Ṣaḥāba (Companions)", "7th C.", "~5,800", "Medina / Mecca"],
     ["Tābiʿūn (Successors)", "8th C.", "~22,000", "Medina / Kufa"],
     ["Atbāʿ at-Tābiʿīn", "9th C.", "~40,000", "Baghdad"],
     ["Later 3rd–5th C.", "9–11th C.", "~100,000+", "Widespread"],
     ["Post-5th C.", "11th+ C.", "~95,000", "Global"]])

# ══════════════════════════════════════════════════════════
# CH4: EXPERIMENT DESIGN
# ══════════════════════════════════════════════════════════
sec("CHAPTER 4", "Experiment Design")

# Stage 1 Experiments
s = cs("4.2  Stage 1 Experiments", "Data Ingestion & Two-Tier Alignment")
two_col(s, [
    "Exp 1: Biographical Text Parsing",
    "• Goal: Extract structured biographies from 36 volumes",
    "• Method: Regex for name, kunya, nisba, ṭabaqa, grade",
    "• Target: >90% extraction recall",
    "",
    "Exp 2: Two-Tier Alignment Framework",
    "• MOTIVATION: Sanadset has chains no grades; HF has grades no sanads",
    "• Goal: Link HF query Hadith to matching Sanadset sanad",
    "• Tier 1: TF-IDF (reduces search space ~85%)",
    "• Tier 2: Levenshtein exact chain verification",
], [
    "Datasets Used:",
    "• A: 1,200 book-balanced (200 per book from HF)",
    "• B: 1,200 grade-balanced",
    "• C: 6,000 book-balanced (1,000 per book)",
    "• D: 6,000 grade-balanced (3,022 unique grades)",
    "",
    "All configs: 6 books from HuggingFace hadith_datasets",
    "Each aligned: sanad (Sanadset) + grade (HuggingFace)",
])

# Stage 2 Experiments
s = cs("4.3  Stage 2 Experiments", "Entity Linking & Graph Construction")
two_col(s, [
    "Exp 3: Narrator Reliability Classification",
    "• Multi-class tier prediction from biography text",
    "• Fine-tuned AraBERT vs. TF-IDF + SVM baseline",
    "",
    "Exp 4: Narrator Disambiguation",
    "• Resolve alias variations → canonical narrator IDs",
    "• Root-based consonantal skeleton + Levenshtein",
    "",
    "Exp 5: KG Construction & Topology",
    "• Build full KG from aligned sanads",
    "• Analyse degree dist., density, components",
], [
    "Evaluation Metrics:",
    "• Classification: Macro/Weighted F1, per-class PR",
    "• Disambiguation: Accuracy, alias resolution ratio",
    "• Graph: |V|, |E|, avg degree, clustering coefficient",
    "",
    "Baselines:",
    "• SVM + TF-IDF (traditional ML)",
    "• Random graph (same |V|,|E| for topology comparison)",
    "• Exact string match (disambiguation baseline)",
    "• node2vec (embedding baseline for GAE)",
])

# Stage 3 Experiments
s = cs("4.4  Stage 3 Experiments", "Scoring, GNN, Aggregation & Ablation")
two_col(s, [
    "Exp 6: GAE Link Prediction",
    "• Mask 15% edges, self-supervised reconstruction",
    "• 2-layer GCN (256→64), inner-product decoder",
    "• Metrics: ROC-AUC, Average Precision",
    "",
    "Exp 7: Sentiment Quantisation",
    "• Map 5-tier labels → continuous [0.05–0.85]",
    "• AraBERT vs. human labels, calibration analysis",
], [
    "Exp 8 & 9: Aggregation & Multi-Book",
    "• Grid search for optimal α in soft aggregation",
    "• Component ablation: semantic vs. +geo vs. +temporal vs. +GNN",
    "• Per-book accuracy across all 6 canonical collections",
    "• Grade-wise recall on 3,022 grade-balanced set",
    "",
    "Evaluation: Accuracy, Cohen’s κ (vs. HF grades), per-book recall",
])

# ══════════════════════════════════════════════════════════
# CH5: PRELIMINARY RESULTS
# ══════════════════════════════════════════════════════════
sec("CHAPTER 5", "Preliminary Results")

# 5.1 Stage 1 Results
s = cs("5.1  Stage 1 Results", "Data Ingestion & Alignment")
two_col(s, [
    "Exp 1: Biographical Parsing",
    "• Total biographies extracted: 15,448",
    "• Recall: >95% on held-out pages",
    "• Field accuracy: Name 93% | Grade 89% | Nisba 91%",
    "• 36 volumes fully parsed",
    "",
    "Exp 2: Alignment Performance",
    "• Precision: ~94% on 500-Hadith evaluation set",
    "• Significance: each Hadith now has BOTH sanad (Sanadset) AND grade (HF)",
], [
    "Alignment Quality per Book:",
    "• Bukhārī: 96% | Muslim: 95% | Abī Dāwūd: 93%",
    "• Tirmiḏī: 92% | Nasāʾī: 93% | Ibn Mājah: 91%",
    "",
    "Tier 1 (TF-IDF): reduces search space by 85%",
    "Tier 2 (Levenshtein): 96% exact-match accuracy",
])

# 5.2 Stage 2 Results
s = cs("5.2  Stage 2 Results", "Classification, Disambiguation & KG Topology")
two_col(s, [
    "Exp 3: AraBERT Classification",
    "• Macro F1: 0.82 (vs. SVM baseline 0.67)",
    "• Per-class: Thiqa 0.89, Ṣadūq 0.78, Maqbul 0.74",
    "•                 Ḍaʿīf 0.71, Matrūk 0.83",
    "• Transformer provides +15% over SVM baseline",
    "",
    "Exp 4: Disambiguation",
    "• 94.2% accuracy on held-out set",
    "• Avg 3.2 alias forms resolved per narrator",
], [
    "Exp 5: KG Topology",
    "• 262,497 nodes | 987,036 directed edges",
    "• Avg degree: 7.5 | Scale-free (γ≈2.3)",
    "• Main component: 93% of nodes connected",
    "",
    "Name Mapping Accuracy:",
    "• Exact string: 78.3%",
    "• Root-based (consonantal): 91.7%",
    "• + Levenshtein (≤2): 94.2%",
])

# GAE Results
s = cs("Exp 6: GAE Link Prediction", "Graph Autoencoder Performance")
two_col(s, [
    "GAE Architecture:",
    "• Encoder: 2-layer GCN (256→64 bottleneck)",
    "• Decoder: Inner product of latent vectors",
    "• Training: Self-supervised (mask 15% edges)",
    "• Adam, lr=0.01, 200 epochs, 1:1 neg sampling",
    "",
    "Results:",
    "• ROC-AUC: 92.39%",
    "• Avg Precision: 94.04%",
    "• vs. node2vec: +3.5% ROC-AUC",
    "• 68% missing links in top-20% predictions",
], [
    "Interpretation:",
    "• GNNs learn meaningful structural representations from isnād networks",
    "• 64-dim embeddings capture teacher–student patterns beyond adjacency",
    "• Higher centrality nodes → more reliable predictions",
    "• Embeddings used for GNN uncertainty modulation of sentiment scores",
])

# Sentiment Results
s = cs("Exp 7: Sentiment Quantisation", "Score Distribution & Classification Calibration")
tbl(s, Inches(0.8), Inches(1.9), Inches(11.5), Inches(2.0),
    ["Grade Tier", "% of Corpus", "F1", "Precision", "Recall"],
    [["Thiqah", "38%", "0.89", "0.91", "0.87"],
     ["Ṣadūq", "22%", "0.78", "0.76", "0.80"],
     ["Maqbul", "15%", "0.74", "0.72", "0.76"],
     ["Ḍaʿīf", "18%", "0.71", "0.68", "0.74"],
     ["Matrūk", "7%", "0.83", "0.85", "0.82"]])
tb(s, Inches(0.8), Inches(4.2), Inches(11.5), Inches(1.5),
   "Consensus normalisation improved inter-annotator agreement by ~12% (60% raw → 72% normalised).\n"
   "Continuous score mapping enables fine-grained differentiation within tiers.\n"
   "262,497 narrator scores (only 2,172 have direct biography labels — others inferred via GNN propagation).",
   13, DARK_TEXT, ls=1.4)

# Ablation Study
s = cs("Exp 8: Ablation Study", "Contribution of Each Modality")
tbl(s, Inches(0.8), Inches(1.9), Inches(11.5), Inches(2.0),
    ["Configuration", "Accuracy", "Cohen’s κ", "Ṣaḥīḥ Recall", "Ḍaʿīf Recall"],
    [["Semantic-only (baseline)", "59.8%", "0.45", "68%", "61%"],
     ["+ Geospatial", "62.4%", "0.50", "70%", "64%"],
     ["+ Geospatial + Temporal", "63.7%", "0.52", "71%", "65%"],
     ["+ All + GNN", "64.2%", "0.53", "72%", "65%"],
     ["Full (all + tuning)", "65.1%", "0.55", "72%", "67%"]])
tb(s, Inches(0.8), Inches(4.2), Inches(11.5), Inches(1.2),
   "Geospatial provides the largest individual gain (+2.6%). GNN adds marginal accuracy (+0.5–1%)\n"
   "but improves interpretability. Full pipeline achieves 65.1% — consistent with sanad-only ceiling.",
   13, DARK_TEXT, ls=1.3)

# Multi-Book Breakdown
s = cs("Exp 9: Multi-Book Performance", "Book-Level Accuracy & Grade Recall")
two_col(s, [
    "Accuracy by Book:",
    "• Ṣaḥīḥ al-Bukhārī: 68% (highest)",
    "• Ṣaḥīḥ Muslim: 65%",
    "• Sunan Abī Dāwūd: 60%",
    "• Jāmiʿ at-Tirmiḏī: 58%",
    "• Sunan al-Nasāʾī: 59%",
    "• Sunan Ibn Mājah: 54% (lowest)",
], [
    "Grade Recall (3,022 balanced set):",
    "• Ṣaḥīḥ: 72%",
    "• Ḥasan: 48% (hardest — subjective boundaries)",
    "• Ḍaʿīf: 65%",
    "",
    "Bukhārī highest: most rigorously graded.",
    "Ibn Mājah lowest: known disputes in grading.",
    "Framework is conservative (over-penalises → Ḍaʿīf).",
])

# Discussion & Ceiling
s = cs("5.4  Discussion & Theoretical Ceiling", "Why ~65% Accuracy?")
two_col(s, [
    "Sanad-Only Ceiling:",
    "• Sanad-only analysis ceiling ~65–70%",
    "• Grade also depends on matn (textual criticism)",
    "• Hidden defects (ʿilal) only detectable via matn comparison",
    "• Some Ḍaʿīf grades due to matn, not sanad",
    "",
    "Implication: NOT a limitation of ML but an epistemological",
    "constraint of sanad-only analysis. The framework performs",
    "at its theoretical maximum for the input domain.",
], [
    "Geospatial & Temporal Impact:",
    "• ~12% of chains have geospatial violations",
    "• Violations correlate strongly with Ḍaʿīf",
    "• Temporal verification catches ~8% more inconsistencies",
    "• GNN: marginal accuracy gain but better interpretability",
    "",
    "Achieving higher accuracy requires adding matn analysis",
    "— planned for FYP2.",
])

# ══════════════════════════════════════════════════════════
# CH6: CONCLUSION
# ══════════════════════════════════════════════════════════
sec("CHAPTER 6", "Conclusion")

# Summary
s = cs("Summary of Contributions")
tf = rtb(s, Inches(0.8), Inches(1.9), Inches(11.5), Inches(5.0))
sp(tf, "What Was Achieved", 20, DARK_NAVY, True, sa=8)
for a in [
    "First computational framework combining NER + KG + GNN + sentiment scoring for ʿIlm ar-Rijal",
    "Alignment pipeline bridges two complementary datasets: Sanadset 650K (chains) + HuggingFace (grades)",
    "15,448 biographies parsed → 262,497 narrator KG with 987K transmission edges",
    "AraBERT-based Jumhūr sentiment pipeline: 5-tier classical evaluations → continuous [0.05–0.85]",
    "GAE link prediction: 92.39% ROC-AUC — validates structural learning from isnād networks",
    "Reproducible, inspectable pipeline — every score traceable to source text and chain position",
    "Empirical confirmation of sanad-only theoretical ceiling (~65% accuracy)",
]:
    ap(tf, "✓  " + a, 15, DARK_TEXT, sa=8, ls=1.3)

# Limitations
s = cs("6.1  Key Limitations")
two_col(s, [
    "Dataset:",
    "• Only 15K biographies graded (of 262K nodes)",
    "• Grade imbalance: Thiqa 38% vs. Matrūk 7%",
    "• HF coverage: only 6 books — many chains lack matching grade",
    "• Geospatial: only ~40% of narrators have confirmed locations",
    "",
    "Methodological:",
    "• Sanad-only: cannot capture matn-based defects",
    "• Single-school (Sunni Jarḥ wa Taʿdīl)",
    "• Batch pipeline only — no real-time deployment",
], [
    "Structural Insights:",
    "• Sanad-only ceiling ~65–70% is an inherent epistemic limit",
    "• Geospatial violations = strongest individual penalty signal",
    "• GNN adds marginal accuracy but improves interpretability",
    "• Consensus aggregation reduces label noise ~12%",
    "• Cold-start (99.17% unlabeled) necessitates self-supervised pre-training",
])

# Future Work
s = cs("6.2  Future Work (FYP2)")
two_col(s, [
    "Technical:",
    "• Matn integration — textual analysis for full dual-track (80%+ target)",
    "• Multi-language: Urdu & Persian commentaries",
    "• Real-time API deployment",
    "• Scale to full 600K+ Sanadset",
    "• Cross-school: include Shīʿa rijāl evaluation",
], [
    "Validation & Timeline:",
    "• Expert review with qualified muhaddithūn",
    "• Expand ground-truth: 10K+ manual annotations",
    "• Sensitivity analysis per Jumhūr tier shift",
    "",
    "FYP2 Phases (24 weeks):",
    "• W1–6: Matn integration + dataset expansion",
    "• W7–14: Model improvement + cross-school comparison",
    "• W15–20: Expert validation + sensitivity analysis",
    "• W21–24: Web deployment + documentation",
])

# ══════════════════════════════════════════════════════════
# THANK YOU
# ══════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s, DARK_NAVY)
add_shape(s, Inches(1.0), Inches(3.0), Inches(11.3), Pt(5), GOLD)
tb(s, Inches(1.0), Inches(1.8), Inches(11.3), Inches(1.2),
   "Thank You", 48, WHITE, True, PP_ALIGN.CENTER, "Calibri Light")
tb(s, Inches(1.0), Inches(3.3), Inches(11.3), Inches(1.0),
   "Questions & Discussion", 26, GOLD, al=PP_ALIGN.CENTER, font_name="Calibri Light")
tb(s, Inches(1.0), Inches(4.8), Inches(11.3), Inches(1.5),
   "Alan Syahmi Bin Sanih @ Sani  |  BI23110344\n"
   "Supervisor: Dr. Khalifa Chekima\n"
   "Faculty of Computing and Informatics  |  Universiti Malaysia Sabah",
   16, RGBColor(0xAA,0xBB,0xCC), al=PP_ALIGN.CENTER, ls=1.4)
tb(s, Inches(3.0), Inches(6.3), Inches(7.3), Inches(0.5),
   "خَيْرُ النَّاسِ أَنْفَعُهُمْ لِلنَّاسِ",
   18, GOLD, al=PP_ALIGN.CENTER)

# ── SAVE ─────────────────────────────────────────────────
out = str(Path(__file__).resolve().parent / "FYP1_Presentation_v2.pptx")
prs.save(out)
print(f"Saved: {out}")
print(f"Total slides: {len(prs.slides)}")
