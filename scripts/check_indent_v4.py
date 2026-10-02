import docx
from docx.shared import Inches, Pt, Cm

doc = docx.Document(r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v4.docx")

# Check what python-docx reads from existing indent
for p in doc.paragraphs:
    if p.text.strip().startswith("The digitisation of classical"):
        print(f"Existing para: first_line_indent = {p.paragraph_format.first_line_indent}")
        print(f"In inches: {p.paragraph_format.first_line_indent / 914400 if p.paragraph_format.first_line_indent else 'N/A'}")
        break

# Check new Ch5 para [458] first_line_indent
p = doc.paragraphs[458]
print(f"\nNew para [458]: first_line_indent = {p.paragraph_format.first_line_indent}")
print(f"Actual value stored: {p.paragraph_format.first_line_indent}")

# What should it be?
print(f"\n0.5 inch in EMU = {Inches(0.5)}")
print(f"720 twips in EMU = {720 * 914400 / 1440}")
