"""
Reformat Chapter 4 to use proper transliteration matching v4 style.
Reads your v1 (with your edits), applies formatting, saves as v3.
"""
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import re

# Transliteration mapping: plain ASCII → v4 style with special characters
TRANSLIT = {
    # Core terms
    "Hadith": "Ḥadīth",
    "hadith": "ḥadīth",
    "sanad": "sanad",
    "Sanad": "Sanad",
    "isnad": "isnād",
    "Isnad": "Isnād",
    "matn": "matn",
    "Matn": "Matn",
    "rijal": "rijāl",
    "Rijal": "Rijāl",
    "jarh wa ta'dil": "jarḥ wa taʿdīl",
    "Jarh wa Ta'dil": "Jarḥ wa Taʿdīl",
    "jarh wa tadil": "jarḥ wa taʿdīl",
    "Jarh wa Tadil": "Jarḥ wa Taʿdīl",
    "jarh": "jarḥ",
    "Jarh": "Jarḥ",
    "ta'dil": "taʿdīl",
    "Ta'dil": "Taʿdīl",
    "tadil": "taʿdīl",
    "Tadil": "Taʿdīl",
    "tadlis": "tadlīs",
    "Tadlis": "Tadlīs",
    "ikhtilat": "ikhtilāṭ",
    "Ikhtilat": "Ikhtilāṭ",
    "turuq": "ṭuruq",
    "Turuq": "Ṭuruq",
    "tabaqa": "ṭabaqa",
    "Tabaqa": "Ṭabaqa",
    "tabaqat": "ṭabaqāt",
    "ingita": "inqiṭāʿ",
    "shudhudh": "shudhūdh",
    "illah": "ʿillah",
    "rihlah": "riḥlah",
    "muhaddithun": "muḥaddithūn",
    "Muhaddithun": "Muḥaddithūn",

    # Grade terms
    "Sahih": "Ṣaḥīḥ",
    "sahih": "ṣaḥīḥ",
    "Daif": "Ḍaʿīf",
    "Da'if": "Ḍaʿīf",
    "daif": "ḍaʿīf",
    "da'if": "ḍaʿīf",
    "Hasan": "Ḥasan",
    "hasan": "ḥasan",
    "Matruk": "Matrūk",
    "matruk": "matrūk",
    "Maqbul": "Maqbūl",
    "maqbul": "maqbūl",
    "Saduq": "Ṣadūq",
    "saduq": "ṣadūq",
    "Thiqah": "Thiqah",
    "thiqah": "thiqah",
    "kadhib": "kadhdhāb",
    "Kadhib": "Kadhdhāb",
    "jahala": "jahālah",
    "Jahala": "Jahālah",

    # Book names
    "Tahdhib al-Kamal": "Tahdhīb al-Kamāl",
    "Tahdhib": "Tahdhīb",
    "Taqrib al-Tahdhib": "Taqrīb al-Tahdhīb",
    "Taqrib": "Taqrīb",
    "Kutub as-Sitta": "Kutub as-Sittah",
    "Muwatta": "Muwaṭṭaʾ",
    "Sanadset": "Sanadset",

    # People
    "Mizzi": "Mizzī",
    "Ibn Hajar": "Ibn Ḥajar",
    "Asqalani": "ʿAsqalānī",
    "Dhahabi": "Dhahabī",
    "Bukhari": "Bukhārī",
    "Muslim": "Muslim",
    "Tirmidhi": "Tirmidhī",
    "Nasai": "Nasāʾī",
    "Dawud": "Dāwūd",
    "Majah": "Mājah",

    # Geographic terms
    "Kufa": "Kūfa",
    "Basra": "Baṣra",
    "Baghdad": "Baghdād",
    "nisba": "nisbah",
    "Nisba": "Nisbah",
}

# Sort by length descending so longer terms match first
sorted_terms = sorted(TRANSLIT.items(), key=lambda x: -len(x[0]))

def apply_translit(text):
    """Replace plain ASCII terms with v4-style transliteration."""
    for ascii_term, v4_term in sorted_terms:
        # Case-insensitive replacement for multi-word terms
        pattern = re.compile(re.escape(ascii_term), re.IGNORECASE)
        text = pattern.sub(v4_term, text)
    return text

def format_paragraph(para):
    """Apply transliteration + italicize foreign terms in one pass."""
    text = para.text
    if not text: return False

    # Apply transliteration
    new_text = apply_translit(text)
    if new_text == text: return False

    # Rebuild paragraph with formatting
    # Clear existing runs
    for run in para.runs:
        run.text = ""

    # Add the transliterated text as one run, italicized if it contains foreign terms
    # Foreign term detection: if any v4-style special char is present, italicize
    has_foreign = any(c in new_text for c in "āīūṣḍṭẓḥḫšġǧʿʾ")

    # Rebuild using runs
    if para.runs:
        run = para.runs[0]
    else:
        run = para.add_run()

    run.text = new_text
    run.italic = has_foreign
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)

    return True

print("Reading your chapter4 v1...")
doc = Document("Report Drafts/chapter4_experimental_design.docx")
print(f"  {len(doc.paragraphs)} paragraphs")

count = 0
for para in doc.paragraphs:
    if format_paragraph(para):
        count += 1

print(f"Reformatted {count} paragraphs with proper transliteration")

out_path = "Report Drafts/chapter4_design_v4.docx"
try:
    import os
    if os.path.exists(out_path):
        os.remove(out_path)
except: pass
doc.save(out_path)
