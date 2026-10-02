"""Compare formatting between Ch1 and Ch4."""
from docx import Document
from docx.oxml.ns import qn

doc = Document(r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters.docx")
f = open(r"c:\Projects\FYP Project\Report Drafts\_fmt_compare.txt", "w", encoding="utf-8")

# Normal style
style = doc.styles['Normal']
f.write(f"Normal style alignment: {style.paragraph_format.alignment}\n")
f.write(f"Normal style line spacing: {style.paragraph_format.line_spacing}\n")
pPr = style.element.find(qn('w:pPr'))
if pPr is not None:
    jc = pPr.find(qn('w:jc'))
    if jc is not None:
        f.write(f"Normal style jc val: {jc.get(qn('w:val'))}\n")

# Ch1 paragraph
for p in doc.paragraphs:
    if p.text.strip().startswith('The authentication') and p.style.name == 'Normal':
        pPr = p._element.find(qn('w:pPr'))
        if pPr is not None:
            sp = pPr.find(qn('w:spacing'))
            if sp is not None:
                f.write(f"\nCh1 spacing attrs: {dict(sp.attrib)}\n")
            jc = pPr.find(qn('w:jc'))
            if jc is not None:
                f.write(f"Ch1 jc: {jc.get(qn('w:val'))}\n")
            else:
                f.write("Ch1: no jc element in pPr\n")
        break

# Ch4 paragraph
for p in doc.paragraphs:
    if p.text.strip().startswith('This chapter lays') and p.style.name == 'Normal':
        pPr = p._element.find(qn('w:pPr'))
        if pPr is not None:
            sp = pPr.find(qn('w:spacing'))
            if sp is not None:
                f.write(f"\nCh4 spacing attrs: {dict(sp.attrib)}\n")
            jc = pPr.find(qn('w:jc'))
            if jc is not None:
                f.write(f"Ch4 jc: {jc.get(qn('w:val'))}\n")
        break

# Also check original Ch1 heading for style
for p in doc.paragraphs:
    if p.text.strip().startswith('1.1 Ov') and p.style.name.startswith('Heading'):
        pPr = p._element.find(qn('w:pPr'))
        if pPr is not None:
            sp = pPr.find(qn('w:spacing'))
            if sp is not None:
                f.write(f"\nCh1 H2 spacing: {dict(sp.attrib)}\n")
        break

# And Ch4 heading
for p in doc.paragraphs:
    if p.text.strip().startswith('4.1 Intr') and p.style.name.startswith('Heading'):
        pPr = p._element.find(qn('w:pPr'))
        if pPr is not None:
            sp = pPr.find(qn('w:spacing'))
            if sp is not None:
                f.write(f"\nCh4 H2 spacing: {dict(sp.attrib)}\n")
        break

f.close()
print("Written")
