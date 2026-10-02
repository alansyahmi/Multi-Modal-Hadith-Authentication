import pdfplumber, sys

with pdfplumber.open(r"c:\Projects\FYP Project\FYP Guides\GAYAPENULISANTESIS-DISERTASIUMS2018.pdf") as pdf:
    print(f"Total pages: {len(pdf.pages)}")
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text and text.strip():
            # Replace problematic characters
            clean = text.replace('', '*')
            print(f"\n--- PAGE {i+1} ---")
            try:
                print(clean[:3000])
            except:
                print(clean.encode('utf-8', 'replace').decode('utf-8', 'replace')[:3000])
