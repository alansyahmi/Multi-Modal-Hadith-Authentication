import docx
from lxml import etree

doc = docx.Document(r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v4.docx")
ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

with open(r"c:\Projects\FYP Project\Report Drafts\_v4_indent.txt", "w", encoding="utf-8") as f:
    # Find Ch5 start
    ch5_start = None
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip() == "CHAPTER 5":
            ch5_start = i
            break

    f.write(f"CHAPTER 5 starts at [{ch5_start}]\n\n")
    f.write("=== INDENTATION CHECK ===\n")

    # Check every paragraph in Ch5
    for i in range(ch5_start, min(ch5_start + 60, len(doc.paragraphs))):
        p = doc.paragraphs[i]
        t = p.text.strip()[:100] if p.text.strip() else "(empty)"
        ppr = p._element.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr")
        indent_val = "none"
        if ppr is not None:
            ind = ppr.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ind")
            if ind is not None:
                fl = ind.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}firstLine")
                if fl:
                    indent_val = f"firstLine={fl}"
        f.write(f"[{i}] ({p.style.name}) indent={indent_val} | {t}\n")
