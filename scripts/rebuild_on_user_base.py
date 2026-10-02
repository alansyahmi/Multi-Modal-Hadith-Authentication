"""
Rebuild from user's v4_ch5.docx as base.
- Preserves user's exact wording/formatting changes
- Applies UK English spelling
- Fixes geospatial FYP2 framing in 5.4.2
- Rebuilds italic runs properly (Arabic terms, transliterated terms)
- Adds geospatial mention to Chapter 4
"""
import re, docx
from docx.shared import Inches
from lxml import etree

SRC = r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v4_ch5.docx"
DST = r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v4.docx"

doc = docx.Document(SRC)

# ── UK English spelling map ──
UK_MAP = [
    ("normalize", "normalise"),
    ("Normalize", "Normalise"),
    ("normalization", "normalisation"),
    ("Normalization", "Normalisation"),
    ("characterize", "characterise"),
    ("characterized", "characterised"),
    ("organization", "organisation"),
    ("recognize", "recognise"),
    ("recognized", "recognised"),
    ("utilize", "utilise"),
    ("utilized", "utilised"),
    ("vocalize", "vocalise"),
    ("vocalized", "vocalised"),
    ("unvocalized", "unvocalised"),
    ("analyse", "analyse"),  # already UK, skip
    ("neighborhood", "neighbourhood"),
    ("Neighborhood", "Neighbourhood"),
]

# ── Arabic/foreign terms that should be italic ──
# We auto-detect: Arabic script AND words with special transliteration chars
# (underdots, macrons, ʿ/ʾ). Build regex for matching.

# Known multi-word terms (must match before single-word terms)
KNOWN_ITALIC = [
    "Tahdhīb al-Kamāl", "Taqrīb at-Tahdhīb",
    "Jarḥ wa Taʿdīl", "Kutub as-Sitta",
    "Maktaba Shamela", "ʿIlm ar-Rijal",
    "alif maqṣūra", "tāʾ marbūṭa",
    "kunya-nasab", "nasab-kunya",
    "Ibn Hajar",  # Proper name — NOT italic
]

# Single-word terms
SINGLE_ITALIC = [
    "tarājim", "Tahdhīb", "Taqrīb",
    "sanad", "isnād", "matn", "matns",
    "thiqa", "ṣadūq", "ḍaʿīf",
    "Ṣaḥīḥ", "Jarḥ", "Taʿdīl",
    "tadlīs", "ikhtilāṭ",
    "ism", "nasab", "kunya", "nisba",
    "Shamela",  # part of Maktaba Shamela — only italic if standalone
]

# ── Helper: check if a word/segment contains transliteration chars ──
TRANS_CHARS = set("ṣḍḥṭẓāīūʿʾ")

def has_transliteration(s):
    """Check if string contains Arabic transliteration characters."""
    return any(c in TRANS_CHARS for c in s)

def has_arabic_script(s):
    """Check if string contains Arabic Unicode characters."""
    return bool(re.search(r'[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]', s))

def should_italicize(segment):
    """Determine if a text segment should be italicized."""
    s = segment.strip()
    if not s:
        return False
    # Arabic script → italic
    if has_arabic_script(s):
        return True
    # Transliteration chars → italic
    if has_transliteration(s):
        # Exception: proper names like "Ibn Hajar" contain ʿ but shouldn't be italic
        if "Ibn Hajar" in segment or "ibn" in segment.lower() or "bin" in segment.lower():
            # Only italicize if it's a standalone term, not a name
            if s in SINGLE_ITALIC:
                return True
            return False
        return True
    # Check known terms list
    for term in SINGLE_ITALIC:
        if s == term or s == term.lower():
            return True
    return False

# ── UK spelling fix ──
def fix_uk(text):
    for us, uk in UK_MAP:
        text = text.replace(us, uk)
    return text

# ── Find chapter boundaries ──
def find_ch(name):
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip() == name and p.style.name == "Heading 1":
            return i
    return None

ch5 = find_ch("CHAPTER 5")
ref = find_ch("REFERENCES")
ch4 = find_ch("CHAPTER 4")

print(f"Ch4=[{ch4}], Ch5=[{ch5}], REF=[{ref}]")

# ═══ FIX 1: UK English + Rebuild italic runs in Chapter 5 ═══
for i in range(ch5, ref):
    p = doc.paragraphs[i]
    if p.style.name != "Normal" or not p.text.strip():
        continue

    original_text = p.text

    # Apply UK spelling
    fixed_text = fix_uk(original_text)

    # ── Special text fixes ──
    # Fix 5.4.2 geospatial in FYP2 list
    if "(b) adding geospatial consistency verification" in fixed_text:
        fixed_text = fixed_text.replace(
            "(b) adding geospatial consistency verification, and (c) exploring",
            "and (b) exploring"
        )
        print(f"  [{i}] Fixed geospatial in FYP2 list")

    # If text didn't change and no italic needed, skip
    # Rebuild runs for any paragraph with Arabic/transliteration chars
    needs_rebuild = (fixed_text != original_text) or has_transliteration(fixed_text) or has_arabic_script(fixed_text)

    if not needs_rebuild:
        continue

    # ── Rebuild paragraph runs ──
    # Strategy: split text at words that should be italic, keeping delimiters
    text = fixed_text

    # Simple approach: split into words/wsp/non-wsp, italicize words that match
    # Split on word boundaries
    parts = re.split(r'(\s+)', text)  # splits preserving whitespace

    # Remove existing runs
    for r in p.runs:
        r._element.getparent().remove(r._element)

    # Build new runs, merging consecutive segments with same italic status
    new_run_data = []  # list of (text, italic)
    for part in parts:
        if not part:
            continue
        is_italic = should_italicize(part)
        # Check if it's part of a multi-word known term
        if not is_italic and part.strip():
            # Check if it's a multi-word known term
            for term in KNOWN_ITALIC:
                if term in part:
                    is_italic = True
                    break
        new_run_data.append((part, is_italic))

    # Merge consecutive runs with same italic status
    merged = []
    for seg_text, seg_italic in new_run_data:
        if merged and merged[-1][1] == seg_italic:
            merged[-1] = (merged[-1][0] + seg_text, seg_italic)
        else:
            merged.append((seg_text, seg_italic))

    # Create runs
    for seg_text, seg_italic in merged:
        run = p.add_run(seg_text)
        if seg_italic:
            run.italic = True

    if fixed_text != original_text or merged:
        print(f"  [{i}] Rebuilt runs" + (" (UK fix)" if fixed_text != original_text else ""))

# ═══ FIX 2: Geospatial mentions in Chapter 4 ═══
# Find Experiment 5 body paragraph
for i in range(ch4, ch5):
    p = doc.paragraphs[i]
    if "log-transformed appearance count" in p.text and "in-degree" in p.text:
        old = p.text
        if "geographic" not in old:
            # Replace node features list with geospatial mention
            # The original says: "log-transformed appearance count, in-degree, out-degree, average chain position, and the grade score"
            new = old.replace(
                "log-transformed appearance count, in-degree, out-degree, average chain position, and the grade score",
                "log-transformed appearance count, in-degree, out-degree, average chain position, the Ibn Hajar grade score, and a binary flag indicating whether a known geographic region (nisba) can be associated with the narrator for geospatial consistency verification"
            )
            # Rebuild runs
            if new != old:
                for r in p.runs:
                    r._element.getparent().remove(r._element)
                # Parse {italic:...} style — but we need to insert markers first
                # nisba should be italic
                new_marked = new.replace("(nisba)", "({italic:nisba})")
                parts = re.split(r'\{italic:([^}]+)\}', new_marked)
                for j, segment in enumerate(parts):
                    if not segment:
                        continue
                    is_italic = (j % 2 == 1)
                    run = p.add_run(segment)
                    if is_italic:
                        run.italic = True
                print(f"  [{i}] Added geospatial to Experiment 5")

# Find Experiment 6 body paragraph
for i in range(ch4, ch5):
    p = doc.paragraphs[i]
    if "Three variants were compared" in p.text:
        old = p.text
        if "geographic region flag" not in old and "8 structural features" in old:
            # Replace the 8 structural features description
            new = old.replace(
                "8 structural features",
                "8 structural features (degree centrality, appearance count, chain position, grade score, and a binary geographic region flag)"
            )
            if new != old:
                for r in p.runs:
                    r._element.getparent().remove(r._element)
                run = p.add_run(new)
                print(f"  [{i}] Added geographic flag to Experiment 6")
        break

# ═══ Save ═══
doc.save(DST)
print(f"\nSaved to {DST}")
