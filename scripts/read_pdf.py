import fitz
doc = fitz.open(r"c:\Projects\FYP Project\FYP Guides\GAYAPENULISANTESIS-DISERTASIUMS2018.pdf")
print(f"Total pages: {doc.page_count}")
for i, page in enumerate(doc):
    text = page.get_text()
    if text.strip():
        print(f"\n--- PAGE {i+1} ---")
        print(text[:3000])
