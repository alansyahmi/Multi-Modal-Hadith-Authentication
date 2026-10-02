"""
Fix remaining issues in v5:
1. Add missing Ch4 section rewrites (wrong heading key matching)
2. Add table captions for Tables 8-18
"""
import re, copy, docx
from lxml import etree

DST = r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v5.docx"
doc = docx.Document(DST)
NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

def find_heading(text):
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip() == text and p.style.name in ("Heading 1", "Heading 2", "Heading 3"):
            return i
    return None

ch4 = find_heading("CHAPTER 4")
ch5 = find_heading("CHAPTER 5")
ref = find_heading("REFERENCES")

# ── Find sections that still need rewriting ──
def get_sections(start, end):
    sections = []
    current = None
    body = []
    for i in range(start, end):
        p = doc.paragraphs[i]
        s = p.style.name
        t = p.text.strip()
        if s in ("Heading 2", "Heading 3") and t:
            if current:
                sections.append((current, body))
            current = (i, t, s)
            body = []
        elif current and s == "Normal":
            body.append(i)
    if current:
        sections.append((current, body))
    return sections

ch4_sections = get_sections(ch4, ch5)

# Check which sections still have old-style text (contain dashes or "I"/"we")
out = open(r"c:\Projects\FYP Project\Report Drafts\_v5_missing.txt", "w", encoding="utf-8")
out.write("=== Ch4 sections that may need rewriting ===\n")

for (idx, hdr, style), body in ch4_sections:
    # Read the first body paragraph
    if body:
        text = doc.paragraphs[body[0]].text[:120]
        # Check if it has old-style content
        if any(marker in text for marker in ["—", "–", "I ", " we ", "We "]):
            out.write(f"\n[{idx}] '{hdr}' (needs rewrite)\n  First line: {text}\n")
        else:
            out.write(f"\n[{idx}] '{hdr}' (OK)\n  First line: {text}\n")
    else:
        out.write(f"\n[{idx}] '{hdr}' (no body paragraphs)\n")

# Write full Ch4 section list
out.write("\n\n=== FULL SECTION LIST ===\n")
for (idx, hdr, style), body in ch4_sections:
    out.write(f"[{idx}] ({style}) {hdr}\n")

out.close()
print("Done")
