import docx, sys
from docx.shared import Pt, Inches, Cm, Emu

doc = docx.Document(r'c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v3.docx')

def emu_to_cm(emu):
    return round(emu / 914400 * 2.54, 2)

with open(r'c:\Projects\FYP Project\Report Drafts\_v3_format_audit.txt', 'w', encoding='utf-8') as f:
    f.write('=== DEFAULT STYLE ===\n')
    def_style = doc.styles['Normal']
    f.write(f'Font: {def_style.font.name}, Size: {def_style.font.size}\n')
    f.write(f'Line spacing: {def_style.paragraph_format.line_spacing}\n')
    f.write(f'Alignment: {def_style.paragraph_format.alignment}\n')

    f.write('\n=== SECTIONS / MARGINS (in cm) ===\n')
    for i, sec in enumerate(doc.sections):
        f.write(f'Section {i}: T={emu_to_cm(sec.top_margin)}cm, B={emu_to_cm(sec.bottom_margin)}cm, '
                f'L={emu_to_cm(sec.left_margin)}cm, R={emu_to_cm(sec.right_margin)}cm\n')

    f.write('\n=== HEADING STYLES ===\n')
    for sname in ['Heading 1', 'Heading 2', 'Heading 3', 'Heading 4', 'Title', 'Subtitle', 'Normal']:
        try:
            s = doc.styles[sname]
            f.write(f'{sname}: font={s.font.name}, size={s.font.size}, bold={s.font.bold}, '
                    f'line_spacing={s.paragraph_format.line_spacing}, alignment={s.paragraph_format.alignment}\n')
        except:
            pass

    f.write('\n=== ALL PARAGRAPH STYLES (non-empty) ===\n')
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip():
            rfont = p.runs[0].font if p.runs else None
            rf_name = rfont.name if rfont else '?'
            rf_size = rfont.size if rfont else '?'
            f.write(f'[{i}] style={p.style.name} | font={rf_name} size={rf_size} | '
                    f'align={p.paragraph_format.alignment} | '
                    f'{p.text[:120].replace(chr(0x02bf), chr(0x27e8)).replace(chr(0x02be), chr(0x27e9))}\n')

    f.write(f'\n=== TABLE COUNT: {len(doc.tables)} ===\n')

print('Written to _v3_format_audit.txt')
