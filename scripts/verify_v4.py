import docx
from lxml import etree

doc = docx.Document(r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v4_ch5.docx")
ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

with open(r"c:\Projects\FYP Project\Report Drafts\_v4_verify.txt", "w", encoding="utf-8") as f:
    # Check spacing XML in new Ch5 para [457]
    p = doc.paragraphs[457]
    ppr = p._element.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr")
    f.write("=== Ch5 para [457] pPr ===\n")
    if ppr is not None:
        f.write(etree.tostring(ppr, encoding="unicode", pretty_print=True))

    # Check heading para [451]
    p = doc.paragraphs[451]
    ppr = p._element.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr")
    f.write("\n=== Ch5 heading [451] pPr ===\n")
    if ppr is not None:
        f.write(etree.tostring(ppr, encoding="unicode", pretty_print=True))

    # Check a Ch5 Heading 2
    p = doc.paragraphs[456]
    ppr = p._element.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr")
    f.write("\n=== Ch5 Heading2 [456] pPr ===\n")
    if ppr is not None:
        f.write(etree.tostring(ppr, encoding="unicode", pretty_print=True))

    # Compare with existing Ch4 body para
    f.write("\n=== Ch4 para [386] (existing) ===")
    p = doc.paragraphs[386]
    pf = p.paragraph_format
    f.write(f"\n  line_spacing={pf.line_spacing}\n")

    # Ch5 body para
    f.write("=== Ch5 para [457] (new) ===")
    p = doc.paragraphs[457]
    pf = p.paragraph_format
    f.write(f"\n  line_spacing={pf.line_spacing}\n")

print("Done")
