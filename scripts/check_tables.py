from docx import Document
doc = Document(r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v3.docx")
f = open(r"c:\Projects\FYP Project\Report Drafts\_v3_tables.txt", "w", encoding="utf-8")
f.write(f"Total tables: {len(doc.tables)}\n\n")
for i, t in enumerate(doc.tables):
    rows_show = min(3, len(t.rows))
    fc = ""
    if t.rows and t.rows[0].cells:
        fc = t.rows[0].cells[0].text[:60]
    f.write(f"Table {i}: {len(t.rows)} rows x {len(t.columns)} cols | \"{fc}\"\n")
f.close()
print("Done")
