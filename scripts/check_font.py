import docx
from docx.shared import Pt, Emu
from lxml import etree

doc = docx.Document(r'c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v3.docx')

# Check document-level default font from XML
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
body = doc.element.body

# Check theme/fonts from styles.xml
styles_xml = doc.styles.element
fonts = styles_xml.findall('.//w:rFonts', ns)
print("=== rFonts in styles ===")
for f in fonts[:5]:
    print(etree.tostring(f, encoding='unicode'))

# Check actual XML of a normal paragraph in Ch1 to see font
print("\n=== Sample Normal paragraph XML (Ch1, first para after 1.1) ===")
for i, p in enumerate(doc.paragraphs):
    if p.style.name == 'Normal' and len(p.text) > 50:
        # Print XML of first run
        if p.runs:
            run_elem = p.runs[0]._element
            rpr = run_elem.find('.//w:rPr', ns)
            if rpr is not None:
                print(etree.tostring(rpr, encoding='unicode'))
            else:
                print("No rPr (no explicit formatting)")
        break

# Check document theme font
print("\n=== Document theme ===")
theme = doc.element.find('.//w:theme', ns)
if theme is not None:
    theme_fonts = theme.findall('.//a:majorFont', {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'})
    for tf in theme_fonts:
        print(etree.tostring(tf, encoding='unicode')[:500])

# Check paragraph justification for Ch1 paragraph
print("\n=== Paragraph justification ===")
for i, p in enumerate(doc.paragraphs):
    if p.style.name == 'Normal' and len(p.text) > 100:
        ppr = p._element.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr')
        if ppr is not None:
            jc = ppr.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}jc')
            if jc is not None:
                print(f"Para [{i}]: jc={jc.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')}")
            else:
                print(f"Para [{i}]: no jc element (inherits alignment)")
        else:
            print(f"Para [{i}]: no pPr")
        break

# Check if there are any default run properties in the document
print("\n=== Default run props ===")
def_rpr = body.find('.//w:body/w:pPr/w:rPr', ns)
if def_rpr is not None:
    print(etree.tostring(def_rpr, encoding='unicode'))

# Check document part for default styles
print("\n=== Document defaults from styles.xml ===")
doc_defaults = styles_xml.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}docDefaults')
if doc_defaults is not None:
    print(etree.tostring(doc_defaults, encoding='unicode')[:1000])
