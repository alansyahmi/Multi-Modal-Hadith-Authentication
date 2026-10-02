"""
Final rebuild: user's v4_ch5 as base + UK English + proper italic runs + geospatial fixes.
"""
import re, docx
from docx.shared import Inches

SRC = r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v4_ch5.docx"
DST = r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v4.docx"

doc = docx.Document(SRC)

# === UK English mapping ===
UK_MAP = [
    ("normalize", "normalise"), ("Normalize", "Normalise"),
    ("normalization", "normalisation"), ("Normalization", "Normalisation"),
    ("vocalized", "vocalised"), ("unvocalized", "unvocalised"),
    ("characterize", "characterise"), ("characterized", "characterised"),
    ("organization", "organisation"), ("recognize", "recognise"),
    ("neighborhood", "neighbourhood"),
]

# === Multi-word terms that must be italicised as a unit ===
MW_TERMS = [
    "Kutub as-Sitta", "Tahdhīb al-Kamāl", "Taqrīb at-Tahdhīb",
    "Jarḥ wa Taʿdīl", "Maktaba Shamela",
    "ʿIlm ar-Rijal", "alif maqṣūra", "tāʾ marbūṭa",
]
MW_RE = re.compile('|'.join(re.escape(t) for t in MW_TERMS))

# === Single tokens that should be italic ===
TRANS_LETTERS = set("ṣḍḥṭẓāīūʿʵ")

def is_transliterated(w):
    return any(c in TRANS_LETTERS for c in w)

def is_arabic_script(w):
    return bool(re.search(r'[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]', w))

# === Build a set of single italic tokens from MW_TERMS ===
SINGLE_ITALIC = {
    "tarājim", "Tahdhīb", "Taqrīb",
    "sanad", "isnād", "matn", "matns",
    "thiqa", "ṣadūq", "ḍaʿīf",
    "Ṣaḥīḥ", "Jarḥ", "Taʿdīl",
    "tadlīs", "ikhtilāṭ",
    "ism", "nasab", "kunya", "nisba",
    "Shamela",
}

def rebuild_runs(p, text):
    """Replace paragraph runs with properly italicised runs from plain text."""
    # Remove old runs
    for r in list(p.runs):
        r._element.getparent().remove(r._element)

    # Mark multi-word known terms
    marked = MW_RE.sub(lambda m: f'{{italic:{m.group(0)}}}', text)

    # Split into segments
    raw_segs = re.split(r'(\{italic:[^}]+\})', marked)
    segs = []  # list of (text, is_italic)

    for seg in raw_segs:
        if not seg:
            continue
        if seg.startswith('{italic:') and seg.endswith('}'):
            segs.append((seg[8:-1], True))
            continue
        # Split unmarked segment at boundaries between italic-worthy spans
        # Use a simple char-by-char or token approach
        tokens = re.split(r'(\s+)', seg)
        for t in tokens:
            if not t:
                continue
            stripped = t.strip()
            if not stripped:
                segs.append((t, False))
            else:
                is_it = (is_arabic_script(stripped)
                         or is_transliterated(stripped)
                         or stripped in SINGLE_ITALIC
                         or stripped.lower() in SINGLE_ITALIC)
                segs.append((t, is_it))

    # Merge adjacent same-format
    merged = []
    for txt, it in segs:
        if merged and merged[-1][1] == it:
            merged[-1] = (merged[-1][0] + txt, it)
        else:
            merged.append((txt, it))

    for txt, it in merged:
        run = p.add_run(txt)
        if it:
            run.italic = True

    return merged


# === Locate chapters ===
def find_ch(name):
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip() == name and p.style.name == "Heading 1":
            return i
    return None

ch4 = find_ch("CHAPTER 4")
ch5 = find_ch("CHAPTER 5")
ref = find_ch("REFERENCES")
print(f"Ch4=[{ch4}]  Ch5=[{ch5}]  REF=[{ref}]")

# ═══ Ch5: UK English + geospatial fix + italic runs ═══
for i in range(ch5, ref):
    p = doc.paragraphs[i]
    if p.style.name != "Normal" or not p.text.strip():
        continue

    text = p.text

    # UK English
    for us, uk in UK_MAP:
        text = text.replace(us, uk)

    # Fix 5.4.2: remove geospatial from FYP2 list
    if "(b) adding geospatial consistency verification" in text:
        text = text.replace(
            "(b) adding geospatial consistency verification, and (c) exploring",
            "and (b) exploring"
        )
        print(f"  [{i}] Fixed geospatial in FYP2 list")

    # Fix 5.5: remove "Fourth, geospatial validation ... FYP2"
    if "Fourth, geospatial validation using location-based edge verification is not yet implemented and requires Geopy gazetteer mapping in FYP2" in text:
        # Remove the entire "Fourth..." sentence including the period
        text = text.replace(
            "Fourth, geospatial validation using location-based edge verification is not yet implemented and requires Geopy gazetteer mapping in FYP2. ",
            ""
        ).replace(
            "Fourth, geospatial validation using location-based edge verification is not yet implemented and requires Geopy gazetteer mapping in FYP2.",
            ""
        )
        print(f"  [{i}] Removed 'Fourth: geospatial is FYP2' sentence")

    # Rebuild italic runs if text changed or contains italic-worthy content
    if text != p.text or is_arabic_script(text) or is_transliterated(text):
        rebuild_runs(p, text)
        if text != p.text:
            print(f"  [{i}] Fixed text + rebuilt runs")
        else:
            print(f"  [{i}] Rebuilt italic runs")

# ═══ Ch4: add geospatial mentions ═══
# Experiment 5
for i in range(ch4, ch5):
    p = doc.paragraphs[i]
    if "log-transformed appearance count" in p.text and "in-degree" in p.text:
        old = p.text
        if "geographic" not in old:
            new = old.replace(
                "log-transformed appearance count, in-degree, out-degree, average chain position, and the grade score",
                "log-transformed appearance count, in-degree, out-degree, average chain position, the Ibn Hajar grade score, and a binary flag indicating whether a known geographic region ({italic:nisba}) can be associated with the narrator for geospatial consistency verification"
            )
            if new != old:
                for r in list(p.runs):
                    r._element.getparent().remove(r._element)
                parts = re.split(r'\{italic:([^}]+)\}', new)
                for j, seg in enumerate(parts):
                    if not seg:
                        continue
                    run = p.add_run(seg)
                    if j % 2 == 1:
                        run.italic = True
                print(f"  [{i}] Updated Experiment 5 with geospatial")

# Experiment 6
for i in range(ch4, ch5):
    p = doc.paragraphs[i]
    if "Three variants were compared" in p.text and "8 structural features" in p.text:
        old = p.text
        if "geographic region flag" not in old:
            new = old.replace(
                "8 structural features",
                "8 structural features (degree centrality, appearance count, chain position, grade score, and a binary geographic region flag)"
            )
            if new != old:
                for r in list(p.runs):
                    r._element.getparent().remove(r._element)
                p.add_run(new)
                print(f"  [{i}] Updated Experiment 6 with geographic flag")
        break

doc.save(DST)
print(f"\nSaved to {DST}")
