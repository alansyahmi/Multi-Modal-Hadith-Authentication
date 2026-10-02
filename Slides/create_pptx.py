#!/usr/bin/env python3
"""
Generate FYP1 Defense Presentation (.pptx)
Based on: fyp1_full_report_v2_edited.docx
Topic: Multi-Modal Framework for Quantifying Narrator Reliability in ʿIlm ar-Rijal
       using Semantic Sentiment Analysis and Graph Neural Networks
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import copy

# ── Colour Palette ──────────────────────────────────────────────
DARK_NAVY   = RGBColor(0x1B, 0x2A, 0x4A)   # headings, backgrounds
MEDIUM_BLUE = RGBColor(0x2D, 0x5F, 0x8A)   # accents
LIGHT_BLUE  = RGBColor(0x3A, 0x7C, 0xBD)   # sub-accents
GOLD        = RGBColor(0xD4, 0xA8, 0x3C)   # highlights
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
OFF_WHITE   = RGBColor(0xF5, 0xF0, 0xEB)   # slide bg tint
DARK_TEXT    = RGBColor(0x2C, 0x2C, 0x2C)
LIGHT_GRAY  = RGBColor(0x6B, 0x7B, 0x8D)
DARK_GRAY   = RGBColor(0x4A, 0x4A, 0x4A)
TABLE_HEAD  = RGBColor(0x1B, 0x2A, 0x4A)
TABLE_ALT   = RGBColor(0xE8, 0xEF, 0xF5)

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
W = prs.slide_width
H = prs.slide_height

# ── Helpers ──────────────────────────────────────────────────────

def add_bg(slide, color=OFF_WHITE):
    """Fill slide background."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_shape(slide, left, top, width, height, fill_color, alpha=None):
    """Add a filled rectangle."""
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill_color
    shp.line.fill.background()
    return shp

def add_textbox(slide, left, top, width, height, text="", font_size=18,
                color=DARK_TEXT, bold=False, alignment=PP_ALIGN.LEFT,
                font_name="Calibri", line_spacing=1.15):
    """Add a text box with a single paragraph."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    p.space_after = Pt(2)
    if line_spacing:
        p.line_spacing = line_spacing
    return tf

def add_rich_textbox(slide, left, top, width, height):
    """Add an empty text box and return text_frame for manual building."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    return tf

def add_paragraph(tf, text, size=16, color=DARK_TEXT, bold=False,
                  alignment=PP_ALIGN.LEFT, font_name="Calibri",
                  space_after=4, bullet=False, indent_level=0,
                  line_spacing=1.2):
    """Append a paragraph to an existing text frame."""
    p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    p.space_after = Pt(space_after)
    p.level = indent_level
    if line_spacing:
        p.line_spacing = line_spacing
    if bullet:
        p.level = indent_level
    return p

def set_first_para(tf, text, size=16, color=DARK_TEXT, bold=False,
                   alignment=PP_ALIGN.LEFT, font_name="Calibri",
                   space_after=4, line_spacing=1.2):
    """Set text on the first paragraph (already exists)."""
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    p.space_after = Pt(space_after)
    if line_spacing:
        p.line_spacing = line_spacing
    return p

def add_bullet_list(tf, items, size=15, color=DARK_TEXT, bold_first=False,
                    indent=0, space_after=3, line_spacing=1.2):
    """Add multiple bullet-point paragraphs."""
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.font.name = "Calibri"
        p.space_after = Pt(space_after)
        p.level = indent
        if line_spacing:
            p.line_spacing = line_spacing
        if bold_first and i == 0:
            p.font.bold = True

def make_section_header(title_text, subtitle_text=""):
    """Dark navy slide with gold accent bar — section divider."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    add_bg(slide, DARK_NAVY)
    # gold bar
    add_shape(slide, Inches(1.2), Inches(3.3), Inches(10.9), Pt(6), GOLD)
    # title
    add_textbox(slide, Inches(1.2), Inches(1.8), Inches(10.9), Inches(1.5),
                title_text, font_size=40, color=WHITE, bold=True,
                alignment=PP_ALIGN.LEFT, font_name="Calibri Light")
    # subtitle
    if subtitle_text:
        add_textbox(slide, Inches(1.2), Inches(3.6), Inches(10.9), Inches(1.0),
                    subtitle_text, font_size=18, color=LIGHT_GRAY,
                    alignment=PP_ALIGN.LEFT, font_name="Calibri Light")
    # bottom accent line
    add_shape(slide, Inches(1.2), Inches(6.5), Inches(10.9), Pt(1), LIGHT_BLUE)
    return slide

def make_content_slide(title_text, subtitle_text=""):
    """White slide with navy header bar and gold accent."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    # top bar
    add_shape(slide, 0, 0, W, Inches(1.15), DARK_NAVY)
    # gold underline
    add_shape(slide, Inches(0.8), Inches(1.15), Inches(3.5), Pt(4), GOLD)
    # title
    add_textbox(slide, Inches(0.8), Inches(0.2), Inches(11.5), Inches(0.85),
                title_text, font_size=30, color=WHITE, bold=True,
                font_name="Calibri Light")
    if subtitle_text:
        add_textbox(slide, Inches(0.8), Inches(1.3), Inches(11.5), Inches(0.5),
                    subtitle_text, font_size=14, color=LIGHT_GRAY,
                    font_name="Calibri", bold=False, alignment=PP_ALIGN.LEFT)
    return slide

def make_two_col(slide, left_items, right_items, left_title="", right_title=""):
    """Create two-column bullet layout."""
    left_tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(5.6), Inches(5.2))
    if left_title:
        set_first_para(left_tf, left_title, size=18, color=DARK_NAVY, bold=True, space_after=6)
    add_bullet_list(left_tf, left_items, size=14, indent=0)

    right_tf = add_rich_textbox(slide, Inches(6.9), Inches(1.9), Inches(5.6), Inches(5.2))
    if right_title:
        set_first_para(right_tf, right_title, size=18, color=DARK_NAVY, bold=True, space_after=6)
    add_bullet_list(right_tf, right_items, size=14, indent=0)

def add_table(slide, left, top, width, height, rows, cols, data, col_widths=None,
              header_color=TABLE_HEAD, alt_color=TABLE_ALT):
    """Add a styled table."""
    table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table
    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = w
    for r in range(rows):
        for c in range(cols):
            cell = table.cell(r, c)
            cell.text = data[r][c] if r < len(data) and c < len(data[r]) else ""
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(12)
                paragraph.font.name = "Calibri"
                paragraph.alignment = PP_ALIGN.CENTER
                if r == 0:
                    paragraph.font.color.rgb = WHITE
                    paragraph.font.bold = True
                else:
                    paragraph.font.color.rgb = DARK_TEXT
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = header_color
            elif r % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = alt_color
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    return table

def add_simple_table(slide, left, top, width, height, headers, rows_data):
    """Convenience: headers list + list-of-lists rows_data."""
    rows = len(rows_data) + 1
    cols = len(headers)
    data = [headers] + rows_data
    return add_table(slide, left, top, width, height, rows, cols, data)


# ════════════════════════════════════════════════════════════════
# SLIDE 1 — TITLE
# ════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, DARK_NAVY)
add_shape(slide, Inches(1.0), Inches(1.5), Inches(11.3), Pt(5), GOLD)
add_textbox(slide, Inches(1.0), Inches(0.6), Inches(11.3), Inches(0.5),
            "FINAL YEAR PROJECT 1", font_size=14, color=GOLD, bold=True,
            font_name="Calibri")
add_textbox(slide, Inches(1.0), Inches(1.8), Inches(11.3), Inches(1.8),
            "A Multi-Modal Framework for\nQuantifying Narrator Reliability\nin ʿIlm ar-Rijal",
            font_size=38, color=WHITE, bold=True, font_name="Calibri Light",
            line_spacing=1.15)
add_textbox(slide, Inches(1.0), Inches(3.9), Inches(11.3), Inches(0.7),
            "Using Semantic Sentiment Analysis and Graph Neural Networks",
            font_size=20, color=LIGHT_BLUE, font_name="Calibri Light")
add_shape(slide, Inches(1.0), Inches(4.8), Inches(11.3), Pt(1), RGBColor(0x3A, 0x5A, 0x7A))
add_textbox(slide, Inches(1.0), Inches(5.1), Inches(5.5), Inches(0.5),
            "Alan Syahmi Bin Sanih @ Sani  |  BI23110344",
            font_size=16, color=RGBColor(0xCC, 0xCC, 0xCC), font_name="Calibri")
add_textbox(slide, Inches(1.0), Inches(5.6), Inches(5.5), Inches(0.5),
            "Supervisor: Dr. Khalifa Chekima",
            font_size=16, color=RGBColor(0xCC, 0xCC, 0xCC), font_name="Calibri")
add_textbox(slide, Inches(1.0), Inches(6.4), Inches(11.3), Inches(0.5),
            "Faculty of Computing and Informatics  |  Universiti Malaysia Sabah",
            font_size=13, color=LIGHT_GRAY, font_name="Calibri")

# ════════════════════════════════════════════════════════════════
# SLIDE 2 — OUTLINE
# ════════════════════════════════════════════════════════════════
slide = make_content_slide("Presentation Outline")
items = [
    "1.  Introduction — Background, Problem Statements, Objectives & Scope",
    "2.  Literature Review — Hadith Authentication, GNN, Sentiment Quantisation",
    "3.  Project Methodology — Research Design, Pipeline Stages, Datasets",
    "4.  Experiment Design — 9 Experiments across 3 Stages",
    "5.  Preliminary Results — Key Findings & Performance Metrics",
    "6.  Conclusion — Limitations & Future Work (FYP2)",
]
tf = add_rich_textbox(slide, Inches(1.0), Inches(1.9), Inches(11.3), Inches(5.0))
set_first_para(tf, items[0], size=18, color=DARK_TEXT, space_after=10, line_spacing=1.5)
add_paragraph(tf, items[1], size=18, color=DARK_TEXT, space_after=10, line_spacing=1.5)
add_paragraph(tf, items[2], size=18, color=DARK_TEXT, space_after=10, line_spacing=1.5)
add_paragraph(tf, items[3], size=18, color=DARK_TEXT, space_after=10, line_spacing=1.5)
add_paragraph(tf, items[4], size=18, color=DARK_TEXT, space_after=10, line_spacing=1.5)
add_paragraph(tf, items[5], size=18, color=DARK_TEXT, space_after=10, line_spacing=1.5)


# ════════════════════════════════════════════════════════════════
# CHAPTER 1: INTRODUCTION
# ════════════════════════════════════════════════════════════════
make_section_header("CHAPTER 1", "Introduction")

# 1.1 Overview
slide = make_content_slide("1.1  Overview", "The Challenge of Digital Hadith Authentication")
tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(11.5), Inches(5.0))
set_first_para(tf, "Core Idea", size=20, color=DARK_NAVY, bold=True, space_after=8)
add_bullet_list(tf, [
    "Hadith authentication via sanad (chain of narration) criticism is a cornerstone of Islamic scholarship",
    "ʿIlm ar-Rijal evaluates narrators for moral probity (ʿadāla) & mnemonic precision (ḍabṭ)",
    "Digital collections are now too large for unaided human review — a scalability crisis",
    "This project builds a pipeline that automates narrator reliability scoring by combining:",
], size=15, indent=0, space_after=4)
add_paragraph(tf, "", size=8, space_after=2)  # spacer
# sub-items
add_paragraph(tf, "▸  Arabic NER (AraBERT) for precise narrator extraction", size=15, color=DARK_NAVY, space_after=3)
add_paragraph(tf, "▸  Graph Neural Networks (GAE) for isnād structural modelling", size=15, color=DARK_NAVY, space_after=3)
add_paragraph(tf, "▸  Semantic Sentiment Quantisation for continuous reliability scoring", size=15, color=DARK_NAVY, space_after=6)
add_paragraph(tf, "", size=8, space_after=2)
add_paragraph(tf, "Goal: A reproducible, inspectable system grounded in classical Jarḥ wa Taʿdīl principles",
              size=15, color=MEDIUM_BLUE, bold=True, space_after=2)

# 1.2 Problem Background
slide = make_content_slide("1.2  Problem Background", "Why Automation is Needed")
tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(tf, "Traditional Bottlenecks", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf, [
    "Manual cross-referencing across 12+ volumes of biographical literature",
    "Expert-dependent process — resistant to scale",
    "Orthographic ambiguity in classical Arabic consonantal script",
    "Minor name variations → narrator misidentification",
], size=14, indent=0, space_after=4)

rtf = add_rich_textbox(slide, Inches(6.9), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(rtf, "Why Now?", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(rtf, [
    "Transformer models (AraBERT) achieve high accuracy in narrator disambiguation",
    "GNN frameworks can learn structural credibility from transmission networks",
    "Feasibility confirmed: ML can assist isnād analysis (Usman, 2025; Mahmoud et al., 2022)",
    "Urgent need as digital Hadith circulation outpaces verification capacity (Mosa, 2025)",
], size=14, indent=0, space_after=4)

# 1.3 Problem Statements
slide = make_content_slide("1.3  Problem Statements", "Two Critical Gaps in Current Research")
tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(11.5), Inches(2.2))
set_first_para(tf, "PS1:  Lack of Quantified Reliability Frameworks", size=20, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf, [
    "Current NER models can extract & disambiguate narrator names but cannot judge their standing",
    "No existing framework maps classical Jarḥ wa Taʿdīl assessments to a standardised reliability index",
    "Identifying a narrator ≠ evaluating their trustworthiness (Ahmad et al., 2025)",
], size=15, indent=0, space_after=6)

tf2 = add_rich_textbox(slide, Inches(0.8), Inches(4.2), Inches(11.5), Inches(2.5))
set_first_para(tf2, "PS2:  Graph Construction without Reliability Scoring", size=20, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf2, [
    "Transformer models can extract narration chains as graphs (Mosa, 2025; Oumer, 2025)",
    "But existing work stops at graph construction — no reliability scores are assigned",
    "No use of actual Jarḥ wa Taʿdīl criticism terms to judge narrator trustworthiness",
    "This research goes beyond: adds quantitative reliability index + GNN-based structural analysis",
], size=15, indent=0, space_after=4)

# 1.4 Objectives
slide = make_content_slide("1.4  Project Objectives")
tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(11.5), Inches(5.0))
set_first_para(tf, "Three Primary Objectives", size=22, color=DARK_NAVY, bold=True, space_after=10)

objs = [
    ("O1", "Develop an NLP classification model that maps qualitative scholarly judgments (Jarḥ wa Taʿdīl) to a standardised quantitative reliability index (0.0 – 1.0)"),
    ("O2", "Construct a Knowledge Graph of narrator relationships and investigate structural, temporal, and geographical features as contributing factors to transmission chain plausibility"),
    ("O3", "Evaluate the framework's accuracy by comparing automated reliability scores against chain-level Hadith gradings (Ṣaḥīḥ, Ḥasan, Ḍaʿīf)"),
]
for label, text in objs:
    p = tf.add_paragraph()
    p.text = f"{label}:  {text}"
    p.font.size = Pt(16)
    p.font.color.rgb = DARK_TEXT
    p.font.name = "Calibri"
    p.space_after = Pt(14)
    p.line_spacing = 1.3
    # bold the label
    run = p.runs[0]
    run.font.bold = True
    run.font.color.rgb = GOLD

# 1.5 Scope
slide = make_content_slide("1.5  Project Scope", "Data Sources, Scale, Technical & Domain Boundaries")
tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(tf, "Data Sources", size=17, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf, [
    "Shamela Digital Library (Taqrīb at-Tahdhīb, Tahdhīb al-Kamāl)",
    "15,448 narrator biographies extracted from 36 volumes",
    "Sanadset 650K corpus for Hadith alignment",
    "Six Canonical Hadith Collections (Kutub as-Sitta)",
], size=14, indent=0, space_after=6)
add_paragraph(tf, "", size=6, space_after=2)
add_paragraph(tf, "Scale", size=17, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf, [
    "262,497 narrator nodes | 987,036 directed edges",
    "4 dataset configurations: 1,200 & 6,000 aligned Hadith",
    "Book-balanced & grade-balanced evaluation",
], size=14, indent=0, space_after=4)

rtf = add_rich_textbox(slide, Inches(6.9), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(rtf, "Technical Scope", size=17, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(rtf, [
    "Narrator extraction (NER from classical Arabic)",
    "Entity disambiguation via Knowledge Graph",
    "Semantic sentiment → continuous reliability scores",
    "Geospatial & temporal verification",
    "Python + CAMeL Tools + PyArabic + AraBERT/MARBERT",
], size=14, indent=0, space_after=6)
add_paragraph(rtf, "", size=6, space_after=2)
add_paragraph(rtf, "Out of Scope", size=17, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(rtf, [
    "Matn (textual content) analysis",
    "Shīʿa isnād methodologies",
    "Real-time/web deployment",
    "Non-Arabic Hadith commentaries",
], size=14, indent=0, space_after=4)


# ════════════════════════════════════════════════════════════════
# CHAPTER 2: LITERATURE REVIEW
# ════════════════════════════════════════════════════════════════
make_section_header("CHAPTER 2", "Literature Review")

# 2.1 Hadith Structure
slide = make_content_slide("2.1  Hadith & Its Structure", "Sanad and Matn")
tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(11.5), Inches(2.0))
set_first_para(tf, "A Hadith consists of two parts:", size=18, color=DARK_NAVY, bold=True, space_after=8)
add_bullet_list(tf, [
    "Sanad (إسناد): The chain of transmission — \"who told whom\" — traced back to the Prophet ﷺ",
    "Matn (متن): The textual content — the actual saying, action, or approval being reported",
], size=15, indent=0, space_after=6)
add_paragraph(tf, "", size=6, space_after=2)

tf2 = add_rich_textbox(slide, Inches(0.8), Inches(3.6), Inches(11.5), Inches(3.0))
set_first_para(tf2, "Example:", size=18, color=DARK_NAVY, bold=True, space_after=4)
add_paragraph(tf2, "", size=4, space_after=2)
add_paragraph(tf2,
    '"And it was related to me (Yaḥyā) from Mālik, from Ibn Shihāb, from ʿAlī ibn Ḥusayn,\n'
    'that the Messenger of Allāh ﷺ said: "Part of the excellence of a man\'s Islam\n'
    'is that he leaves what does not concern him.""',
    size=14, color=MEDIUM_BLUE, bold=False, line_spacing=1.4, space_after=8)
add_paragraph(tf2, "", size=4, space_after=2)
add_paragraph(tf2,
    "Sanad: Yaḥyā ← Mālik ← Ibn Shihāb ← ʿAlī ← Prophet ﷺ\n"
    "Matn: \"Part of the excellence of a man's Islam is that he leaves what does not concern him.\"",
    size=14, color=DARK_TEXT, line_spacing=1.4, space_after=2)

# 2.2 Impact of Hadith Grading
slide = make_content_slide("2.2  Impact of Hadith Grading", "Why Accuracy Matters")
tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(tf, "Religious Significance", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf, [
    "Ṣaḥīḥ (sound) & Ḥasan (good) → legally normative, 2nd most authoritative source after Quran",
    "Ḍaʿīf (weak) or Mawḍūʿ (fabricated) → risk of introducing groundless practices",
    "1.9 billion Muslims affected by rulings on worship, commerce, family law & ethics",
    "Single unreliable narrator can downgrade an entire Hadith (Kamali, 2014)",
], size=14, indent=0, space_after=4)

rtf = add_rich_textbox(slide, Inches(6.9), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(rtf, "Broader Significance", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(rtf, [
    "Oldest systematic peer-review & source-criticism methodology in human history",
    "Classical muhaddithūn developed chain-of-trust verification centuries before modern citation standards",
    "Interdisciplinary relevance: historians, linguists, information scientists (Ghriss, 2025)",
    "Computational tools applied here extend beyond Islamic studies (A'thoina & Hasballah, 2025)",
], size=14, indent=0, space_after=4)

# 2.3-2.4 Combined: Authentication & ʿIlm ar-Rijal
slide = make_content_slide("2.3 – 2.4  Hadith Authentication & ʿIlm ar-Rijal")
tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(tf, "Authentication Process", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf, [
    "Dual-track: sanad criticism + matn criticism",
    "5 conditions for Ṣaḥīḥ: uninterrupted chain, upright narrators, precise memory, no šāḏ (irregularity), no ʿilla (hidden defect)",
    "This project focuses on sanad criticism only",
], size=14, indent=0, space_after=6)
add_paragraph(tf, "", size=6, space_after=2)
add_paragraph(tf, "Chain Visualisation", size=16, color=DARK_NAVY, bold=True, space_after=4)
add_bullet_list(tf, [
    "Prophet ﷺ → Companion → Tābiʿī → Atbāʿ at-Tābiʿīn → ... → Compiler",
    "Each node = a narrator with known biography",
    "Each edge = a teacher–student transmission relationship",
], size=14, indent=0, space_after=4)

rtf = add_rich_textbox(slide, Inches(6.9), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(rtf, "ʿIlm ar-Rijal (Jarḥ wa Taʿdīl)", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(rtf, [
    "Literally \"Criticism & Commendation\" — the science of narrator evaluation",
    "5-tier classification system (from Ṣaḥīḥ down to Matrūk)",
    "Assesses: moral probity (ʿadāla) + mnemonic precision (ḍabṭ)",
    "Corpus: 36 volumes of Tahdhīb al-Kamāl + Taqrīb at-Tahdhīb",
    "Qualitative judgments → this project's continuous score (0.0 – 1.0)",
], size=14, indent=0, space_after=4)

# 2.5-2.8: Technical Literature
slide = make_content_slide("2.5 – 2.8  Technical Foundations")
tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(tf, "Arabic NER & Knowledge Graphs", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf, [
    "Arabic NER: CAMeL Tools, AraBERT achieve high F1 for narrator extraction",
    "Orthographic ambiguity remains a key challenge (El Ganadi et al., 2025)",
    "Knowledge Graphs enable entity disambiguation across alias variations",
    "Graph structure captures teacher–student relationships spatially",
], size=14, indent=0, space_after=8)
add_paragraph(tf, "", size=6, space_after=2)
add_paragraph(tf, "Graph Neural Networks", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf, [
    "GNNs model structural integrity of isnād chains (Mahmoud et al., 2024)",
    "Graph Autoencoders learn node embeddings via self-supervised link prediction",
    "Captures relative position & credibility within transmission network",
], size=14, indent=0, space_after=4)

rtf = add_rich_textbox(slide, Inches(6.9), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(rtf, "Semantic Sentiment Quantisation", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(rtf, [
    "Translates qualitative scholarly evaluations (e.g., \"Thiqa\", \"Ṣadūq\") to continuous scores",
    "AraBERT-based classification maps 5 tiers to [0.05 – 0.85]",
    "Consensus agreement normalisation across multiple authorities",
], size=14, indent=0, space_after=8)
add_paragraph(rtf, "", size=6, space_after=2)
add_paragraph(rtf, "The Research Gap", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(rtf, [
    "No existing framework combines: NER extraction + KG construction + GNN analysis + sentiment scoring",
    "Prior work stops at graph construction (Mosa, 2025; Oumer, 2025)",
    "This project is first to add quantitative reliability index + multi-modal synthesis",
], size=14, indent=0, space_after=4)


# ════════════════════════════════════════════════════════════════
# CHAPTER 3: PROJECT METHODOLOGY
# ════════════════════════════════════════════════════════════════
make_section_header("CHAPTER 3", "Project Methodology")

# 3.1 Flowchart Overview
slide = make_content_slide("3.1  Pipeline Architecture", "Three-Stage Multi-Modal Framework")
# Box for Stage 1
add_shape(slide, Inches(0.6), Inches(1.8), Inches(3.8), Inches(1.6), DARK_NAVY)
add_textbox(slide, Inches(0.8), Inches(1.9), Inches(3.4), Inches(0.4),
            "STAGE 1", font_size=11, color=GOLD, bold=True)
add_textbox(slide, Inches(0.8), Inches(2.2), Inches(3.4), Inches(1.0),
            "Data Ingestion & Preprocessing\n• Regex parser: 15,448 biographies\n• TF-IDF + Levenshtein alignment\n• 6 Canonical Books cross-referenced",
            font_size=12, color=WHITE, line_spacing=1.2)

# Arrow
add_textbox(slide, Inches(4.5), Inches(2.3), Inches(0.5), Inches(0.5),
            "▸", font_size=28, color=GOLD, bold=True)

# Box for Stage 2
add_shape(slide, Inches(5.0), Inches(1.8), Inches(3.8), Inches(1.6), MEDIUM_BLUE)
add_textbox(slide, Inches(5.2), Inches(1.9), Inches(3.4), Inches(0.4),
            "STAGE 2", font_size=11, color=GOLD, bold=True)
add_textbox(slide, Inches(5.2), Inches(2.2), Inches(3.4), Inches(1.0),
            "Knowledge Graph Construction\n• 262,497 narrator nodes\n• 987,036 directed edges\n• Alias & ancestry matching",
            font_size=12, color=WHITE, line_spacing=1.2)

# Arrow
add_textbox(slide, Inches(8.9), Inches(2.3), Inches(0.5), Inches(0.5),
            "▸", font_size=28, color=GOLD, bold=True)

# Box for Stage 3
add_shape(slide, Inches(9.4), Inches(1.8), Inches(3.3), Inches(1.6), LIGHT_BLUE)
add_textbox(slide, Inches(9.6), Inches(1.9), Inches(2.9), Inches(0.4),
            "STAGE 3", font_size=11, color=GOLD, bold=True)
add_textbox(slide, Inches(9.6), Inches(2.2), Inches(2.9), Inches(1.0),
            "Scoring & Evaluation\n• AraBERT sentiment → scores\n• GAE link prediction\n• Soft aggregation → chain scores",
            font_size=12, color=WHITE, line_spacing=1.2)

# Bottom details - Two columns
tf = add_rich_textbox(slide, Inches(0.6), Inches(3.8), Inches(5.8), Inches(3.2))
set_first_para(tf, "Data Sources", size=17, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf, [
    "Tahdhīb al-Kamāl (al-Mizzī) — comprehensive narrator biographies",
    "Taqrīb at-Tahdhīb (Ibn Hajar) — concise graded summaries",
    "Sanadset 650K — large-scale sanad corpus for alignment",
    "Six Canonical Books — ground-truth Hadith grades for evaluation",
], size=13, indent=0, space_after=4)

rtf = add_rich_textbox(slide, Inches(6.8), Inches(3.8), Inches(5.8), Inches(3.2))
set_first_para(rtf, "Key Technologies", size=17, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(rtf, [
    "Python + PyTorch + PyTorch Geometric (GNN framework)",
    "CAMeL Tools + AraBERT (Arabic NLP)",
    "NetworkX + igraph (graph analytics)",
    "Pandas + NumPy + scikit-learn (evaluation)",
], size=13, indent=0, space_after=4)

# 3.2 Dataset Characteristics
slide = make_content_slide("3.2  Dataset Corpus Characteristics", "NLP, Graph, and Spatiotemporal Dimensions")

tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(3.7), Inches(5.0))
set_first_para(tf, "NLP Dimension", size=16, color=DARK_NAVY, bold=True, space_after=4)
add_bullet_list(tf, [
    "15,448 biographies from 36 volumes",
    "5-tier grade distribution (Ṣaḥīḥ → Matrūk)",
    "Avg bio length: ~250–500 tokens",
    "Grade labels extracted via regex",
], size=13, indent=0, space_after=6)

tf2 = add_rich_textbox(slide, Inches(4.8), Inches(1.9), Inches(3.7), Inches(5.0))
set_first_para(tf2, "KG / GNN Dimension", size=16, color=DARK_NAVY, bold=True, space_after=4)
add_bullet_list(tf2, [
    "262,497 narrator nodes",
    "987,036 directed transmission edges",
    "64-dim embeddings via GAE (self-supervised)",
    "ROC-AUC 92.39%, AP 94.04%",
    "Scale-free degree distribution (power-law)",
], size=13, indent=0, space_after=6)

tf3 = add_rich_textbox(slide, Inches(8.8), Inches(1.9), Inches(3.7), Inches(5.0))
set_first_para(tf3, "Spatiotemporal", size=16, color=DARK_NAVY, bold=True, space_after=4)
add_bullet_list(tf3, [
    "Ṭabaqa generations mapped (22 layers)",
    "Major transmission centres: Medina, Mecca, Kufa, Basra, Baghdad, Damascus",
    "Geospatial violation detection",
    "Compiler lifespan alignment verification",
], size=13, indent=0, space_after=4)

# 3.3 Stage 1: Data Ingestion
slide = make_content_slide("3.3  Stage 1: Data Ingestion & Preprocessing")
tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(tf, "Biographical Text Parsing", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf, [
    "Regex-based parser extracts individual narrator blocks",
    "Structured fields: name, kunya, nisba, ṭabaqa, grade evaluations",
    "36 volumes → 15,448 structured biographies",
    "Output: Narrator Encyclopaedia (JSON + CSV)",
], size=14, indent=0, space_after=8)
add_paragraph(tf, "", size=6, space_after=2)
add_paragraph(tf, "Two-Tier Alignment Framework", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf, [
    "Tier 1: TF-IDF vectorisation for heuristic search space reduction",
    "Tier 2: Levenshtein distance for exact sanad-level verification",
    "Cross-references query Hadith against Sanadset 650K corpus",
], size=14, indent=0, space_after=4)

rtf = add_rich_textbox(slide, Inches(6.9), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(rtf, "Key Results:", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(rtf, [
    "15,448 biographies extracted (high recall >95%)",
    "6,000 Hadith successfully aligned across 6 books",
    "Alignment precision: ~94% on evaluation set",
    "Output feeds directly into KG construction pipeline",
], size=14, indent=0, space_after=4)

# 3.4 Stage 2
slide = make_content_slide("3.4  Stage 2: Knowledge Graph Construction")
tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(tf, "Relational Graph Dimensions", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf, [
    "Nodes: 262,497 unique narrators",
    "Edges: 987,036 directed teacher→student transmissions",
    "Attributes: name variants, nisba, ṭabaqa, lifespan, reliability tier",
    "Alias mapping: resolves kunya, patronymics, laqab to canonical ID",
], size=14, indent=0, space_after=8)
add_paragraph(tf, "", size=6, space_after=2)
add_paragraph(tf, "Algorithmic Name Mapping", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf, [
    "Root-based matching (consonantal skeleton)",
    "Levenshtein distance threshold ≤ 2 for near-identical names",
    "Cross-references with biographical metadata for disambiguation",
    "Resolves an average of 3.2 alias forms per canonical narrator",
], size=14, indent=0, space_after=4)

# 3.5 Stage 3
slide = make_content_slide("3.5  Stage 3: Probabilistic Trust Scoring", "Semantic Sentiment + GAE + Soft Aggregation")
tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(3.7), Inches(5.0))
set_first_para(tf, "Sentiment Scoring", size=16, color=DARK_NAVY, bold=True, space_after=4)
add_bullet_list(tf, [
    "AraBERT classifies → 5-tier Jumhūr labels",
    "Mapping: Thiqa=0.85, Ṣadūq=0.70, Lā Baʾs=0.55, Ḍaʿīf=0.25, Matrūk=0.05",
    "Consensus averaging across multiple biographers",
    "Continuous scores replace discrete labels",
], size=12, indent=0, space_after=4)

tf2 = add_rich_textbox(slide, Inches(4.8), Inches(1.9), Inches(3.7), Inches(5.0))
set_first_para(tf2, "GAE Representation", size=16, color=DARK_NAVY, bold=True, space_after=4)
add_bullet_list(tf2, [
    "2-layer Graph Autoencoder",
    "64-dimensional node embeddings",
    "Self-supervised link prediction",
    "Decoder: inner-product of latent vectors",
    "Uncertainty modulation of sentiment scores",
], size=12, indent=0, space_after=4)

tf3 = add_rich_textbox(slide, Inches(8.8), Inches(1.9), Inches(3.7), Inches(5.0))
set_first_para(tf3, "Chain-Level Scoring", size=16, color=DARK_NAVY, bold=True, space_after=4)
add_bullet_list(tf3, [
    "Soft aggregation: weighted mean + weakest-link",
    "Parameter α controls aggregation shape",
    "Risk modifiers: tadlīs, unknowns, geospatial gaps",
    "Final: S_chain = S_base × ∏(risk_modifiers)",
    "Compared against ground-truth Hadith grade",
], size=12, indent=0, space_after=4)

# Tabaqa table
slide = make_content_slide("Ṭabaqa Generation Distribution", "Narrator Distribution Across Generations")
headers = ["Generation", "Period", "Narrators", "Key Centres"]
data = [
    ["Ṣaḥāba", "7th C.", "~5,800", "Medina / Mecca"],
    ["Tābiʿūn", "8th C.", "~22,000", "Medina / Kufa"],
    ["Atbāʿ at-Tābiʿīn", "9th C.", "~40,000", "Baghdad"],
    ["Later 3rd–5th C.", "9–11th C.", "~100,000+", "Widespread"],
    ["Post-5th C.", "11th+ C.", "~95,000", "Global"],
]
add_simple_table(slide, Inches(0.8), Inches(2.0), Inches(11.5), Inches(3.0),
                 headers, data)

# ════════════════════════════════════════════════════════════════
# CHAPTER 4: EXPERIMENT DESIGN
# ════════════════════════════════════════════════════════════════
make_section_header("CHAPTER 4", "Experiment Design")

# 4.1-4.2
slide = make_content_slide("4.1 – 4.2  Stage 1 Experiments", "Data Ingestion & Two-Tier Alignment")
tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(tf, "Exp 1: Biographical Text Parsing", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf, [
    "Goal: Extract narrator biographies from 36 volumes",
    "Method: Regex patterns for structured fields",
    "Metrics: extraction recall, field accuracy",
    "Target: >90% extraction recall on evaluation set",
], size=14, indent=0, space_after=8)
add_paragraph(tf, "", size=6, space_after=2)
add_paragraph(tf, "Exp 2: Two-Tier Alignment Framework", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf, [
    "Goal: Cross-reference query Hadith against Sanadset 650K",
    "Tier 1: TF-IDF for efficient candidate retrieval",
    "Tier 2: Levenshtein for exact chain verification",
    "Evaluation: precision@k on held-out alignment set",
], size=14, indent=0, space_after=4)

rtf = add_rich_textbox(slide, Inches(6.9), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(rtf, "Datasets Used:", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(rtf, [
    "Config A: 1,200 Hadith (200 per book, balanced)",
    "Config B: 1,200 Hadith (grade-balanced)",
    "Config C: 6,000 Hadith (1,000 per book, balanced)",
    "Config D: 6,000 Hadith (grade-balanced, 3,022 unique grades)",
    "All configs: book-balanced across 6 books, grade-balanced for Ṣaḥīḥ/Ḥasan/Ḍaʿīf",
], size=14, indent=0, space_after=4)

# 4.3 Stage 2
slide = make_content_slide("4.3  Stage 2 Experiments", "Entity Linking & Graph Construction")
tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(tf, "Exp 3: Narrator Reliability Classification", size=17, color=DARK_NAVY, bold=True, space_after=4)
add_bullet_list(tf, [
    "Multi-class tier prediction from biography text",
    "Train/val/test split on extracted biographies",
    "Baseline: TF-IDF + SVM | AraBERT fine-tuned model",
], size=13, indent=0, space_after=8)
add_paragraph(tf, "", size=4, space_after=1)
add_paragraph(tf, "Exp 4: Narrator Disambiguation", size=17, color=DARK_NAVY, bold=True, space_after=4)
add_bullet_list(tf, [
    "Resolve alias variations → canonical narrator IDs",
    "Name matcher using root consonantal skeletons",
    "Evaluation: disambiguation accuracy on held-out set",
], size=13, indent=0, space_after=8)
add_paragraph(tf, "", size=4, space_after=1)
add_paragraph(tf, "Exp 5: KG Construction & Topology", size=17, color=DARK_NAVY, bold=True, space_after=4)
add_bullet_list(tf, [
    "Build full KG with edges from aligned sanads",
    "Analyse: degree distribution, density, connected components",
    "Compare to random graph baselines",
], size=13, indent=0, space_after=4)

rtf = add_rich_textbox(slide, Inches(6.9), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(rtf, "Key Metrics Tracked:", size=17, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(rtf, [
    "Classification F1 (macro, weighted, per-class)",
    "Disambiguation accuracy",
    "Graph: |V|, |E|, avg degree, clustering coefficient",
    "Connected component analysis",
    "Alias resolution ratio",
    "Comparison: random graph vs. scale-free fit",
], size=13, indent=0, space_after=4)

# 4.4 Stage 3
slide = make_content_slide("4.4  Stage 3 Experiments", "Scoring, GNN & Aggregation")
tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(tf, "Exp 6: GAE Link Prediction", size=17, color=DARK_NAVY, bold=True, space_after=4)
add_bullet_list(tf, [
    "Self-supervised: mask 15% edges, predict missing links",
    "Metrics: ROC-AUC, Average Precision",
    "2-layer GCN encoder, inner-product decoder",
    "64-dim embeddings → downstream scoring",
], size=13, indent=0, space_after=8)
add_paragraph(tf, "", size=4, space_after=1)
add_paragraph(tf, "Exp 7: Sentiment Quantisation", size=17, color=DARK_NAVY, bold=True, space_after=4)
add_bullet_list(tf, [
    "Map 5-tier Jumhūr labels → continuous [0.05–0.85]",
    "AraBERT classification vs. human-annotated labels",
    "Score distribution analysis & calibration",
], size=13, indent=0, space_after=8)
add_paragraph(tf, "", size=4, space_after=1)
add_paragraph(tf, "Exp 8 & 9: Chain-Level Aggregation & Ablation", size=17, color=DARK_NAVY, bold=True, space_after=4)
add_bullet_list(tf, [
    "Exp 8: Optimise α in soft aggregation, component ablation study",
    "Exp 9: Multi-book breakdown — per-book accuracy analysis",
], size=13, indent=0, space_after=4)

rtf = add_rich_textbox(slide, Inches(6.9), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(rtf, "Evaluation Metrics:", size=17, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(rtf, [
    "Classification accuracy (% correct grade)",
    "Cohen's κ (agreement with ground truth)",
    "ROC-AUC for link prediction",
    "Per-book & per-grade recall breakdown",
    "Geospatial violation impact analysis",
    "Ablation: semantic-only vs. full multi-modal",
], size=13, indent=0, space_after=4)


# ════════════════════════════════════════════════════════════════
# CHAPTER 5: PRELIMINARY RESULTS
# ════════════════════════════════════════════════════════════════
make_section_header("CHAPTER 5", "Preliminary Results")

# 5.1 Stage 1 Results
slide = make_content_slide("5.1  Stage 1 Results", "Data Ingestion & Alignment Performance")
tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(5.5), Inches(3.0))
set_first_para(tf, "Exp 1: Biographical Parsing", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf, [
    "Total biographies extracted: 15,448",
    "Recall: >95% on held-out evaluation pages",
    "Field accuracy: 93% for name, 89% for grade, 91% for nisba",
    "Coverage: 36 volumes fully parsed",
    "Output: structured JSON encyclopaedia",
], size=14, indent=0, space_after=4)

rtf = add_rich_textbox(slide, Inches(6.9), Inches(1.9), Inches(5.5), Inches(3.0))
set_first_para(rtf, "Exp 2: Alignment Framework", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(rtf, [
    "Alignment precision: ~94% on 500 Hadith evaluation set",
    "Tier 1 (TF-IDF) reduces search space by 85%",
    "Tier 2 (Levenshtein) achieves 96% exact-match accuracy",
    "6,000 Hadith aligned across 4 configs (1,200 + 6,000 variants)",
], size=14, indent=0, space_after=4)

# 5.2 Stage 2 Results
slide = make_content_slide("5.2  Stage 2 Results", "Classification, Disambiguation & KG Topology")
tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(5.5), Inches(3.5))
set_first_para(tf, "Exp 3: Reliability Classification", size=17, color=DARK_NAVY, bold=True, space_after=4)
add_bullet_list(tf, [
    "AraBERT F1 (macro): 0.82 on 5-tier classification",
    "Best per-class: Thiqa (0.89) | Weakest: Ḍaʿīf (0.71)",
    "SVM baseline F1: 0.67 — transformer provides +15%",
], size=13, indent=0, space_after=8)
add_paragraph(tf, "", size=4, space_after=1)
add_paragraph(tf, "Exp 4: Disambiguation", size=17, color=DARK_NAVY, bold=True, space_after=4)
add_bullet_list(tf, [
    "94.2% disambiguation accuracy on held-out set",
    "Avg 3.2 alias forms resolved per canonical narrator",
    "Root-based matching handles orthographic variation",
], size=13, indent=0, space_after=4)

rtf = add_rich_textbox(slide, Inches(6.9), Inches(1.9), Inches(5.5), Inches(3.5))
set_first_para(rtf, "Exp 5: KG Construction", size=17, color=DARK_NAVY, bold=True, space_after=4)
add_bullet_list(rtf, [
    "Nodes: 262,497 | Edges: 987,036",
    "Scale-free degree distribution (power-law exponent ≈ 2.3)",
    "Avg degree: 7.5 edges per node",
    "Main connected component: 93% of nodes",
    "Compared to: random graph of same size (avg deg ~3.4)",
], size=13, indent=0, space_after=4)

# 5.3 Stage 3 Results
slide = make_content_slide("5.3  Stage 3 Results", "GAE, Sentiment Scoring & Chain Aggregation")
tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(3.7), Inches(5.0))
set_first_para(tf, "Exp 6: GAE Link Prediction", size=16, color=DARK_NAVY, bold=True, space_after=4)
add_bullet_list(tf, [
    "ROC-AUC: 92.39%",
    "Avg Precision: 94.04%",
    "64-dim embeddings capture structural roles",
    "Higher centrality nodes → more reliable predictions",
], size=12, indent=0, space_after=6)

tf2 = add_rich_textbox(slide, Inches(4.8), Inches(1.9), Inches(3.7), Inches(5.0))
set_first_para(tf2, "Exp 7: Sentiment Scoring", size=16, color=DARK_NAVY, bold=True, space_after=4)
add_bullet_list(tf2, [
    "Score distribution over 262K nodes",
    "Thiqa: 38% | Ṣadūq: 22% | Lā Baʾs: 15%",
    "Ḍaʿīf: 18% | Matrūk: 7%",
    "Consensus normalisation: 72% inter-annotator agreement",
], size=12, indent=0, space_after=6)

tf3 = add_rich_textbox(slide, Inches(8.8), Inches(1.9), Inches(3.7), Inches(5.0))
set_first_para(tf3, "Exp 8: Chain-Level Results", size=16, color=DARK_NAVY, bold=True, space_after=4)
add_bullet_list(tf3, [
    "Semantic-only: ~60% accuracy, κ ≈ 0.45",
    "+ Geospatial: +3-5% on structurally-challenging chains",
    "+ Temporal: +2-4% on cross-generation chains",
    "Full pipeline (all modalities): 62-65% accuracy",
    "Ablation confirms: geospatial > temporal > GNN for accuracy",
], size=12, indent=0, space_after=6)

# Exp 9: Multi-Book Breakdown
slide = make_content_slide("5.3  Exp 9: Multi-Book Performance Breakdown")
tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(tf, "Book-Level Accuracy", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf, [
    "Ṣaḥīḥ al-Bukhārī: 68% accuracy (highest) — most rigorously graded collection",
    "Ṣaḥīḥ Muslim: 65% accuracy",
    "Sunan Abī Dāwūd: 60% accuracy",
    "Jāmiʿ at-Tirmiḏī: 58% accuracy",
    "Sunan al-Nasāʾī: 59% accuracy",
    "Sunan Ibn Mājah: 54% accuracy (lowest) — known grading disputes",
], size=14, indent=0, space_after=8)
add_paragraph(tf, "", size=6, space_after=2)
add_paragraph(tf, "Grade Recall (3,022 Grade-Balanced Set):", size=17, color=DARK_NAVY, bold=True, space_after=4)
add_bullet_list(tf, [
    "Ṣaḥīḥ recall: 72% | Ḥasan recall: 48% | Ḍaʿīf recall: 65%",
    "Ḥasan is the hardest grade — sits between Ṣaḥīḥ and Ḍaʿīf with subjective boundaries",
    "Framework tends to over-penalise → more conservative (classifies as Ḍaʿīf)",
], size=14, indent=0, space_after=4)

# Ablation Study
slide = make_content_slide("Ablation Study Summary", "Contribution of Each Modality to Accuracy")
headers = ["Configuration", "Accuracy", "Cohen's κ", "Ṣaḥīḥ Recall", "Ḍaʿīf Recall"]
data = [
    ["Semantic-only (baseline)", "59.8%", "0.45", "68%", "61%"],
    ["+ Geospatial", "62.4%", "0.50", "70%", "64%"],
    ["+ Geospatial + Temporal", "63.7%", "0.52", "71%", "65%"],
    ["+ Geospatial + Temporal + GNN", "64.2%", "0.53", "72%", "65%"],
    ["Full (all + tuning)", "65.1%", "0.55", "72%", "67%"],
]
add_simple_table(slide, Inches(0.8), Inches(2.0), Inches(11.5), Inches(3.0),
                 headers, data)

# GAE Link Prediction Results
slide = make_content_slide("GAE Link Prediction Performance", "Graph Autoencoder Results")
tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(tf, "GAE Architecture:", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf, [
    "Encoder: 2-layer Graph Convolutional Network (GCN)",
    "Hidden dimensions: 256 → 64 (bottleneck)",
    "Decoder: Inner product of latent node vectors",
    "Training: Self-supervised — mask 15% of edges, predict existence",
    "Optimiser: Adam, lr=0.01, 200 epochs",
    "Negative sampling: 1:1 ratio",
], size=14, indent=0, space_after=6)

rtf = add_rich_textbox(slide, Inches(6.9), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(rtf, "Results:", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(rtf, [
    "Validation ROC-AUC: 92.39%",
    "Validation AP: 94.04%",
    "68% of missing links in top-20% of predictions",
    "Embeddings used for: GNN uncertainty modulation of sentiment scores",
    "Comparison vs. node2vec: GAE +3.5% ROC-AUC",
], size=14, indent=0, space_after=4)

# Discussion: Sanad-Only Ceiling
slide = make_content_slide("5.4  Discussion & Theoretical Ceiling", "Why ~65% Accuracy?")
tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(tf, "The Sanad-Only Ceiling", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf, [
    "Sanad-only analysis has an inherent accuracy ceiling ~65-70%",
    "Grade determination also depends on matn analysis (textual criticism)",
    "Hidden defects (ʿilal) often only detectable via matn comparison",
    "Some Ḍaʿīf grades are due to matn issues, not sanad issues",
], size=14, indent=0, space_after=8)
add_paragraph(tf, "", size=6, space_after=2)
add_paragraph(tf, "Geospatial & Temporal Impact", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf, [
    "~12% of chains contain at least one geospatial violation",
    "These violations correlate strongly with Ḍaʿīf classifications",
    "Temporal overlap verification catches ~8% additional inconsistencies",
    "GNN embeddings provide marginal gain (+1-2%) over manual features",
], size=14, indent=0, space_after=4)

rtf = add_rich_textbox(slide, Inches(6.9), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(rtf, "Key Insight", size=18, color=GOLD, bold=True, space_after=6)
add_paragraph(rtf, "", size=4, space_after=1)
add_paragraph(rtf,
    "Results are consistent with classical Hadith theory:\n"
    "a reliable sanad is necessary but not sufficient for\n"
    "a Ṣaḥīḥ grade. The framework correctly identifies\n"
    "structurally sound chains but cannot evaluate matn\n"
    "quality — hence the ceiling.",
    size=15, color=MEDIUM_BLUE, line_spacing=1.5, space_after=6)
add_paragraph(rtf, "", size=4, space_after=1)
add_paragraph(rtf,
    "Implication: This is NOT a limitation of the ML approach\n"
    "but an epistemological constraint of sanad-only analysis.",
    size=15, color=DARK_TEXT, bold=True, line_spacing=1.4, space_after=2)


# ════════════════════════════════════════════════════════════════
# CHAPTER 6: CONCLUSION
# ════════════════════════════════════════════════════════════════
make_section_header("CHAPTER 6", "Conclusion")

# Summary
slide = make_content_slide("6.  Summary of Contributions")
tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(11.5), Inches(5.0))
set_first_para(tf, "What Was Achieved", size=22, color=DARK_NAVY, bold=True, space_after=8)

achievements = [
    "✓  First computational framework combining NER + KG + GNN + sentiment scoring for ʿIlm ar-Rijal",
    "✓  15,448 biographies parsed → 262,497 narrator KG with 987K transmission edges",
    "✓  AraBERT-based Jumhūr sentiment pipeline mapping classical tiers → continuous [0.05 – 0.85]",
    "✓  GAE link prediction: 92.39% ROC-AUC — validates structural learning approach",
    "✓  Reproducible, inspectable pipeline — every score traceable to source text and chain position",
    "✓  Empirical confirmation of sanad-only theoretical ceiling (~65% accuracy)",
]
for a in achievements:
    add_paragraph(tf, a, size=16, color=DARK_TEXT, space_after=8, line_spacing=1.3)

# 6.1 Limitations
slide = make_content_slide("6.1  Key Limitations & Structural Insights")
tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(tf, "Dataset Limitations", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf, [
    "Only 15K biographies graded (of 262K nodes) — sparse supervision",
    "Grade imbalance: Thiqa heavily over-represented",
    "Sanad alignment relies on exact spelling — orthographic drift reduces coverage",
    "Geospatial data: only ~40% of narrators have confirmed locations",
], size=14, indent=0, space_after=8)
add_paragraph(tf, "", size=6, space_after=2)
add_paragraph(tf, "Methodological Limitations", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf, [
    "Sanad-only: cannot capture matn-based defects",
    "Single-school focus (Sunni Jarḥ wa Taʿdīl)",
    "No real-time deployment — batch pipeline only",
    "AraBERT struggles with 9th century classical Arabic orthography",
], size=14, indent=0, space_after=4)

rtf = add_rich_textbox(slide, Inches(6.9), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(rtf, "Structural Insights", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(rtf, [
    "Sanad-only accuracy ceiling is ∼65-70% — inherent epistemic limit",
    "Geospatial violations are the strongest individual penalty signal",
    "GNN embeddings add marginal accuracy but improve interpretability",
    "Consensus aggregation across biographers reduces label noise by 12%",
], size=14, indent=0, space_after=4)

# 6.2 Future Work
slide = make_content_slide("6.2  Recommendations for Future Work (FYP2)")
tf = add_rich_textbox(slide, Inches(0.8), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(tf, "Technical Improvements", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf, [
    "Matn integration: add textual content analysis for full dual-track evaluation",
    "Multi-language expansion: include Urdu & Persian Hadith commentaries",
    "Real-time API: deploy scoring pipeline as web service",
    "Larger pre-training: scale to all 600K+ Sanadset entries",
    "Cross-school comparison: include Shīʿa rijāl evaluation methods",
], size=14, indent=0, space_after=8)
add_paragraph(tf, "", size=6, space_after=2)
add_paragraph(tf, "Scholarly Validation", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(tf, [
    "Expert review: have muhaddithūn evaluate edge cases",
    "Expand ground-truth: manual annotation of 10K+ additional chains",
    "Sensitivity analysis: quantify impact of each Jumhūr tier shift on final grade",
], size=14, indent=0, space_after=4)

rtf = add_rich_textbox(slide, Inches(6.9), Inches(1.9), Inches(5.5), Inches(5.0))
set_first_para(rtf, "Planned FYP2 Timeline", size=18, color=DARK_NAVY, bold=True, space_after=6)
add_bullet_list(rtf, [
    "Phase 1: Matn integration & dataset expansion (Weeks 1-6)",
    "Phase 2: Model improvement & cross-school comparison (Weeks 7-14)",
    "Phase 3: Expert validation & sensitivity analysis (Weeks 15-20)",
    "Phase 4: Web deployment & documentation (Weeks 21-24)",
], size=14, indent=0, space_after=4)

# ════════════════════════════════════════════════════════════════
# CLOSING SLIDE
# ════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, DARK_NAVY)
add_shape(slide, Inches(1.0), Inches(3.0), Inches(11.3), Pt(5), GOLD)
add_textbox(slide, Inches(1.0), Inches(1.8), Inches(11.3), Inches(1.2),
            "Thank You", font_size=48, color=WHITE, bold=True,
            alignment=PP_ALIGN.CENTER, font_name="Calibri Light")
add_textbox(slide, Inches(1.0), Inches(3.3), Inches(11.3), Inches(1.0),
            "Questions & Discussion", font_size=26, color=GOLD,
            alignment=PP_ALIGN.CENTER, font_name="Calibri Light")
add_shape(slide, Inches(1.0), Inches(4.5), Inches(11.3), Pt(1), RGBColor(0x3A, 0x5A, 0x7A))
add_textbox(slide, Inches(1.0), Inches(4.8), Inches(11.3), Inches(1.5),
            "Alan Syahmi Bin Sanih @ Sani  |  BI23110344\n"
            "Supervisor: Dr. Khalifa Chekima\n"
            "Faculty of Computing and Informatics  |  Universiti Malaysia Sabah",
            font_size=16, color=RGBColor(0xAA, 0xBB, 0xCC),
            alignment=PP_ALIGN.CENTER, font_name="Calibri", line_spacing=1.4)
add_textbox(slide, Inches(3.0), Inches(6.3), Inches(7.3), Inches(0.5),
            "خَيْرُ النَّاسِ أَنْفَعُهُمْ لِلنَّاسِ",
            font_size=18, color=GOLD, alignment=PP_ALIGN.CENTER, font_name="Calibri")

# ── SAVE ────────────────────────────────────────────────────────
output_path = str(Path(__file__).resolve().parent / "FYP1_Presentation.pptx")
prs.save(output_path)
print(f"Presentation saved to: {output_path}")
print(f"Total slides: {len(prs.slides)}")
