import pdfplumber

with pdfplumber.open(r"c:\Projects\FYP Project\FYP Guides\GAYAPENULISANTESIS-DISERTASIUMS2018.pdf") as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text and text.strip():
            clean = text.replace('', '*')
            lower = clean.lower()
            # Look for formatting-related keywords
            if any(kw in lower for kw in ['font', 'fon', 'tulisan', 'saiz', 'spacing', 'jarak', 'margin', 'ms word', 'format', 'heading', 'bab', 'kertas', 'arial', 'times', 'roman', '1.5', 'chapter']):
                print(f'\n--- PAGE {i+1} ---')
                print(clean[:2000])
