import docx
from lxml import etree

doc = docx.Document(r'c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v3.docx')

ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

# Full docDefaults
dd = doc.styles.element.find('.//w:docDefaults', ns)
print("=== FULL docDefaults ===")
if dd is not None:
    print(etree.tostring(dd, encoding='unicode', pretty_print=True))

# Check first normal body paragraph font/size/justification
print("\n=== First Ch1 body paragraph (after 1.1 Overview) ===")
for i, p in enumerate(doc.paragraphs):
    if p.text.strip().startswith("The authentication of prophetic traditions"):
        # Full XML of pPr and first rPr
        xml = etree.tostring(p._element, encoding='unicode', pretty_print=True)
        print(xml[:2000])
        break

# Check para [139] more carefully
print("\n=== Para 139 formatting ===")
p = doc.paragraphs[139]
pf = p.paragraph_format
print(f"alignment: {pf.alignment}")
print(f"first_line_indent: {pf.first_line_indent}")
print(f"line_spacing: {pf.line_spacing}")
print(f"space_before: {pf.space_before}")
print(f"space_after: {pf.space_after}")
if p.runs:
    r = p.runs[0]
    print(f"run font: {r.font.name}")
    print(f"run size: {r.font.size}")
    print(f"run bold: {r.font.bold}")

# Check Chapter 4 heading
print("\n=== Chapter 4 headings ===")
for i in range(379, 390):
    p = doc.paragraphs[i]
    print(f"[{i}] style={p.style.name} | text={p.text[:80]}")
