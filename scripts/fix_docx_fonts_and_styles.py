"""Utility script to fix font styles in existing Word documents."""
import sys
import os
import argparse
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

sys.stdout.reconfigure(encoding="utf-8")

def fix_document_fonts(file_path, output_path=None):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    if output_path is None:
        output_path = file_path
        
    print(f"Processing: {file_path}")
    doc = Document(file_path)
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    
    # 1. Fix docDefaults in styles.xml
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

    # 2. Fix all named styles in doc.styles
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

    # 3. Fix all paragraphs and runs in document body
    def process_paragraph(p):
        for r in p.runs:
            rPr = r._element.get_or_add_rPr()
            rFonts = rPr.find(qn('w:rFonts'))
            if rFonts is None:
                rFonts = OxmlElement('w:rFonts')
                rPr.append(rFonts)
                
            # Check if run font is code (e.g. Courier, Lucida, Consolas, or style is code)
            font_name = 'Tahoma'
            if r.font.name in ['Courier New', 'Lucida Console', 'Consolas', 'Courier']:
                font_name = 'Lucida Console'
                
            for attr in list(rFonts.attrib.keys()):
                if 'Theme' in attr or 'theme' in attr:
                    del rFonts.attrib[attr]
                    
            rFonts.set(qn('w:ascii'), font_name)
            rFonts.set(qn('w:hAnsi'), font_name)
            rFonts.set(qn('w:cs'), font_name)
            rFonts.set(qn('w:eastAsia'), font_name)

    for p in doc.paragraphs:
        process_paragraph(p)
        
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    process_paragraph(p)

    doc.save(output_path)
    print(f"Saved fixed document to: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        fix_document_fonts(sys.argv[1])
    else:
        # Fix all chapters in new_report_draft_md
        base_dir = r"c:\Projects\FYP Project\Report Drafts\new_report_draft_md"
        for fname in ["chapter3.docx", "chapter4.docx", "chapter5.docx"]:
            fpath = os.path.join(base_dir, fname)
            if os.path.exists(fpath):
                fix_document_fonts(fpath)
