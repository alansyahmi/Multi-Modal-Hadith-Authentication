"""
Post-process Chapter 4 DOCX to italicize non-English terms.
Reads v2, applies formatting, saves as v3.
"""
import re
from docx import Document
from docx.shared import Pt

FOREIGN_TERMS = sorted([
    "Tahdhib al-Kamal", "Taqrib al-Tahdhib", "Kutub as-Sitta",
    "Jarh wa Ta'dil", "jarh wa ta'dil",
    "sanad", "isnad", "matn", "turuq",
    "inqita", "shudhudh", "illah",
    "tabaqa", "Tabaqat",
    "daif", "Daif", "sahih", "Sahih", "hasan", "Hasan",
    "matruk", "Matruk", "maqbul", "Maqbul", "saduq", "Saduq",
    "thiqah", "Thiqah", "kadhib", "tadlis", "ikhtilat", "jahala",
    "rijal", "abih", "muhaddithun",
    "Muwatta", "nisba",
    "Sanadset",
], key=len, reverse=True)

def italicize_paragraph(para):
    text = para.text
    if not text: return False

    # Check if any term exists in this paragraph
    pattern_parts = []
    for t in FOREIGN_TERMS:
        pattern_parts.append(re.escape(t))
    pattern = "|".join(pattern_parts)
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    if not matches: return False

    # Build segments: list of (text, is_foreign)
    segments = []
    last_end = 0
    for m in matches:
        if m.start() > last_end:
            segments.append((text[last_end:m.start()], False))
        segments.append((m.group(), True))
        last_end = m.end()
    if last_end < len(text):
        segments.append((text[last_end:], False))

    # Clear all existing runs
    for run in para.runs:
        run.text = ""

    # Rebuild with mixed formatting
    for i, (seg_text, is_foreign) in enumerate(segments):
        if not seg_text: continue
        if i == 0 and para.runs:
            run = para.runs[0]
        else:
            run = para.add_run()

        run.text = seg_text
        run.italic = is_foreign
        run.font.name = "Times New Roman"
        run.font.size = Pt(12)

    return True

doc = Document("Report Drafts/chapter4_experimental_design.docx")
print(f"Chapter: {len(doc.paragraphs)} paragraphs")

count = 0
for para in doc.paragraphs:
    if italicize_paragraph(para):
        count += 1

print(f"Italicized foreign terms in {count} paragraphs")

doc.save("Report Drafts/chapter4_experimental_design_v3.docx")
print("Saved: chapter4_experimental_design_v3.docx")
