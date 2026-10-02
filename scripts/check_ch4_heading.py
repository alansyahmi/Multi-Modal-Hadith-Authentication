"""Check Ch4 heading formatting only."""
from docx import Document
from docx.oxml.ns import qn

doc = Document(r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters.docx")
f = open(r"c:\Projects\FYP Project\Report Drafts__ch4h.txt", "w", encoding="utf-8")

for p in doc.paragraphs:
    txt = p.text.strip()
    if 'CHAPTER 4:' in txt or txt.startswith('4.'):
        if 'Heading' in p.style.name:
            pPr = p._element.find(qn('w:pPr'))
            info = "no pPr"
            if pPr is not None:
                sp = pPr.find(qn('w:spacing'))
                if sp is not None:
                    info = str(dict(sp.attrib))
                else:
                    info = "no spacing"
            f.write(f"[{p.style.name}] [{txt[:70]}] {info}\n")

f.close()
print("Done")
