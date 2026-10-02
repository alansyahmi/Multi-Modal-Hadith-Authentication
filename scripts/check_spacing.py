import docx
from lxml import etree

doc = docx.Document(r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v3.docx")
ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

# Check Ch1 body paragraph [139] spacing
p139 = doc.paragraphs[139]
print("=== Ch1 para [139] pPr ===")
ppr = p139._element.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr")
if ppr is not None:
    print(etree.tostring(ppr, encoding="unicode", pretty_print=True))
else:
    print("No pPr")

# Check chapter heading
print("\n=== Ch1 heading [135] (INTRODUCTION) pPr ===")
p135 = doc.paragraphs[135]
ppr = p135._element.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr")
if ppr is not None:
    print(etree.tostring(ppr, encoding="unicode", pretty_print=True))

# Check CHAPTER 1 heading
print("\n=== Ch1 [132] pPr ===")
p132 = doc.paragraphs[132]
ppr = p132._element.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr")
if ppr is not None:
    print(etree.tostring(ppr, encoding="unicode", pretty_print=True))
else:
    print("No pPr")

# Check Ch4 body paragraph
print("\n=== Ch4 para [386] (4.1 Introduction body) pPr ===")
p386 = doc.paragraphs[386]
ppr = p386._element.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr")
if ppr is not None:
    print(etree.tostring(ppr, encoding="unicode", pretty_print=True))
