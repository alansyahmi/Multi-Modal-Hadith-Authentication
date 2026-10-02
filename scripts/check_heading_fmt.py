"""Check heading formatting matches."""
from docx import Document
from docx.oxml.ns import qn

doc = Document(r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters.docx")
f = open(r"c:\Projects\FYP Project\Report Drafts\_heading_fmt.txt", "w", encoding="utf-8")

count = 0
for p in doc.paragraphs:
    txt = p.text.strip()
    if not txt:
        continue
    if 'Heading' in p.style.name:
        pPr = p._element.find(qn('w:pPr'))
        spacing_info = "no pPr"
        if pPr is not None:
            sp = pPr.find(qn('w:spacing'))
            if sp is not None:
                spacing_info = str(dict(sp.attrib))
            else:
                spacing_info = "no spacing"
        # Only log Chapter headings and 4.x headings
        if (txt.startswith('CHAPTER') or txt.startswith('4.') or txt.startswith('1.') or txt.startswith('3.')):
            f.write(f"[{p.style.name}] [{txt[:60]}] spacing={spacing_info}\n")
            count += 1
        if count > 30:
            break

f.close()
print("Written")
