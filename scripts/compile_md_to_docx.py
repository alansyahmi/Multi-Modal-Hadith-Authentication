"""Universal Markdown to DOCX compiler following thesis formatting specifications."""
import sys
import os
import re
import time
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree

sys.stdout.reconfigure(encoding="utf-8")

def set_run_font_xml(run, font_name, size_pt=11, bold=False, italic=False):
    rPr = run._element.get_or_add_rPr()
    
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.append(rFonts)
    for attr in list(rFonts.attrib.keys()):
        if 'Theme' in attr or 'theme' in attr:
            del rFonts.attrib[attr]
    rFonts.set(qn('w:ascii'), font_name)
    rFonts.set(qn('w:hAnsi'), font_name)
    rFonts.set(qn('w:cs'), font_name)
    rFonts.set(qn('w:eastAsia'), font_name)
    
    val_sz = str(int(size_pt * 2))
    sz = rPr.find(qn('w:sz'))
    if sz is None:
        sz = OxmlElement('w:sz')
        rPr.append(sz)
    sz.set(qn('w:val'), val_sz)
    
    szCs = rPr.find(qn('w:szCs'))
    if szCs is None:
        szCs = OxmlElement('w:szCs')
        rPr.append(szCs)
    szCs.set(qn('w:val'), val_sz)
    
    if bold:
        if rPr.find(qn('w:b')) is None:
            rPr.append(OxmlElement('w:b'))
    if italic:
        if rPr.find(qn('w:i')) is None:
            rPr.append(OxmlElement('w:i'))

def fix_document_defaults_and_styles(doc):
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    styles_el = doc.styles.element
    
    dd = styles_el.find('.//w:docDefaults', ns)
    if dd is not None:
        rPr = dd.find('.//w:rPr', ns)
        if rPr is not None:
            rFonts = rPr.find('.//w:rFonts', ns)
            if rFonts is not None:
                for attr in list(rFonts.attrib.keys()):
                    if 'Theme' in attr or 'theme' in attr:
                        del rFonts.attrib[attr]
                rFonts.set(qn('w:ascii'), 'Tahoma')
                rFonts.set(qn('w:hAnsi'), 'Tahoma')
                rFonts.set(qn('w:cs'), 'Tahoma')
                rFonts.set(qn('w:eastAsia'), 'Tahoma')
    
    for s in doc.styles:
        if hasattr(s, 'element') and s.element is not None:
            rPr = s.element.find('.//w:rPr', ns)
            if rPr is not None:
                rFonts = rPr.find('.//w:rFonts', ns)
                if rFonts is not None:
                    for attr in list(rFonts.attrib.keys()):
                        if 'Theme' in attr or 'theme' in attr:
                            del rFonts.attrib[attr]
                    rFonts.set(qn('w:ascii'), 'Tahoma')
                    rFonts.set(qn('w:hAnsi'), 'Tahoma')
                    rFonts.set(qn('w:cs'), 'Tahoma')
                    rFonts.set(qn('w:eastAsia'), 'Tahoma')

def compile_markdown_to_docx(md_path, docx_path):
    if not os.path.exists(md_path):
        print(f"File not found: {md_path}")
        return

    print(f"Compiling {md_path} -> {docx_path}")
    doc = Document()

    # Page setup
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

    # Styles
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Tahoma"
    font.size = Pt(11)
    font.color.rgb = RGBColor(0, 0, 0)
    style.paragraph_format.line_spacing = 1.5
    style.paragraph_format.space_after = Pt(6)
    style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    for level in range(1, 4):
        hs = doc.styles[f"Heading {level}"]
        hs.font.name = "Tahoma"
        hs.font.size = Pt(11)
        hs.font.bold = True
        hs.font.color.rgb = RGBColor(0, 0, 0)
        hs.paragraph_format.space_before = Pt(12)
        hs.paragraph_format.space_after = Pt(6)
        if level == 1:
            hs.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        else:
            hs.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    para_count = 0

    def parse_formatted_text(para, text, size=11, font_name="Tahoma", default_bold=False):
        parts = re.split(r'(`[^`]+`|\$[^\$]+\$|\*\*[^\*]+\*\*|\*[^\*]+\*|<br>)', text)
        for part in parts:
            if not part:
                continue
            if part == '<br>':
                para.add_run().add_break()
                continue
            
            run = para.add_run()
            
            if part.startswith('`') and part.endswith('`'):
                run.text = part[1:-1]
                set_run_font_xml(run, "Lucida Console", size_pt=10, bold=default_bold)
            elif part.startswith('$') and part.endswith('$'):
                run.text = part[1:-1]
                set_run_font_xml(run, font_name, size_pt=size, italic=True, bold=default_bold)
            elif part.startswith('**') and part.endswith('**'):
                run.text = part[2:-2]
                set_run_font_xml(run, font_name, size_pt=size, bold=True)
            elif part.startswith('*') and part.endswith('*'):
                run.text = part[1:-1]
                set_run_font_xml(run, font_name, size_pt=size, italic=True, bold=default_bold)
            else:
                run.text = part
                set_run_font_xml(run, font_name, size_pt=size, bold=default_bold)

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line == '---':
            i += 1
            continue

        if line.startswith('#'):
            para_count = 0
            level = line.count('#', 0, 4)
            text = line.lstrip('#').strip()
            if level == 1:
                text = text.upper()
            h = doc.add_paragraph()
            h.style = doc.styles[f"Heading {min(level, 3)}"]
            run = h.add_run(text)
            set_run_font_xml(run, "Tahoma", size_pt=11, bold=True)
            i += 1
            continue

        # Table title check
        if line.startswith('Table ') and ':' in line:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(line)
            set_run_font_xml(run, "Tahoma", size_pt=11, bold=True)
            i += 1
            continue

        # Markdown Table check
        if line.startswith('|'):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip())
                i += 1

            headers = [c.strip() for c in table_lines[0].split('|')[1:-1]]
            rows = []
            for tl in table_lines[2:]: # skip separator
                rows.append([c.strip() for c in tl.split('|')[1:-1]])

            col_count = len(headers)

            # Guard: reject blocks that are not real markdown tables.
            # 1) A real md table needs at least 3 lines (header + separator + data).
            # 2) Every data row must have the same column count as the header.
            # 3) Column count must be > 0.
            is_real_table = (
                len(table_lines) >= 3
                and col_count > 0
                and all(len(r) == col_count for r in rows)
            )

            if not is_real_table:
                for tl in table_lines:
                    p = doc.add_paragraph()
                    p.paragraph_format.left_indent = Inches(0.2)
                    set_run_font_xml(p.add_run(tl), "Consolas", size_pt=8.5)
                continue

            table = doc.add_table(rows=1 + len(rows), cols=col_count)
            table.style = 'Table Grid'
            table.alignment = WD_TABLE_ALIGNMENT.CENTER

            for ci, h in enumerate(headers):
                cell = table.rows[0].cells[ci]
                p = cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after = Pt(2)
                parse_formatted_text(p, h, size=11, default_bold=True)

            for ri, row in enumerate(rows):
                for ci, val in enumerate(row):
                    cell = table.rows[ri + 1].cells[ci]
                    p = cell.paragraphs[0]
                    if re.search(r'^\d+(\.\d+)?%?$', str(val)) or val in ['—', 'Yes', 'No']:
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    else:
                        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    p.paragraph_format.space_after = Pt(2)
                    parse_formatted_text(p, str(val), size=11)
            continue

        # Bullet list check
        if line.startswith('* ') or line.startswith('- '):
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.left_indent = Inches(0.5)
            p.paragraph_format.space_after = Pt(4)
            parse_formatted_text(p, line[2:].strip(), size=11)
            i += 1
            continue

        # Number list check
        if re.match(r'^\d+\.\s+', line):
            text = re.sub(r'^\d+\.\s+', '', line)
            p = doc.add_paragraph(style="List Number")
            p.paragraph_format.left_indent = Inches(0.5)
            p.paragraph_format.space_after = Pt(4)
            parse_formatted_text(p, text, size=11)
            i += 1
            continue

        # Normal paragraph
        para_count += 1
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.5
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

        if para_count >= 3:
            p.paragraph_format.first_line_indent = Inches(0.5)

        parse_formatted_text(p, line, size=11)
        i += 1

    fix_document_defaults_and_styles(doc)
    try:
        doc.save(docx_path)
        print(f"Successfully compiled: {docx_path}")
    except PermissionError:
        ts = int(time.time())
        alt_path = docx_path.replace('.docx', f'_{ts}.docx')
        doc.save(alt_path)
        print(f"Permission denied writing to {docx_path}. Saved to alternate path: {alt_path}")

if __name__ == "__main__":
    base_dir = r"c:\Projects\FYP Project\Report Drafts\new_report_draft_md"
    compile_markdown_to_docx(os.path.join(base_dir, "chapter4.md"), os.path.join(base_dir, "chapter4.docx"))
    compile_markdown_to_docx(os.path.join(base_dir, "chapter5.md"), os.path.join(base_dir, "chapter5.docx"))
