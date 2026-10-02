import docx
from lxml import etree

doc = docx.Document(r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v3.docx")
ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

with open(r"c:\Projects\FYP Project\Report Drafts\_v3_indent.txt", "w", encoding="utf-8") as f:
    # Check paragraphs in Ch1: after 1.1 Overview
    f.write("=== Ch1 paragraphs after 1.1 Overview ===\n")
    for i in range(138, 148):
        p = doc.paragraphs[i]
        ppr = p._element.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr")
        indent = ""
        if ppr is not None:
            ind = ppr.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ind")
            if ind is not None:
                indent = etree.tostring(ind, encoding="unicode")
        t = p.text.strip()[:80]
        f.write(f"[{i}] ({p.style.name}) indent={indent} | {t}\n")

    # Check Ch4 paragraphs
    f.write("\n=== Ch4 paragraphs after 4.1 Introduction ===\n")
    for i in range(385, 395):
        p = doc.paragraphs[i]
        ppr = p._element.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr")
        indent = ""
        if ppr is not None:
            ind = ppr.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ind")
            if ind is not None:
                indent = etree.tostring(ind, encoding="unicode")
            else:
                # Check for firstLine/firstLineChars
                fl = ppr.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}firstLine")
                if fl is not None:
                    indent = etree.tostring(fl, encoding="unicode")
        t = p.text.strip()[:80]
        f.write(f"[{i}] ({p.style.name}) indent={indent} | {t}\n")
