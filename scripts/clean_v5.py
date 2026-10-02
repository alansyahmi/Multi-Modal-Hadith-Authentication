"""Remove orphaned Tables 8-18 and finalise v5."""
import docx
from lxml import etree

DST = r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v5.docx"
doc = docx.Document(DST)
NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

body = doc.element.body

# Find and remove orphaned tables (ones without a caption before them)
tbl_elements = []
for elem in body:
    if elem.tag == f'{{{NS}}}tbl':
        # Check if there's a caption paragraph before it
        prev = elem.getprevious()
        has_caption = False
        while prev is not None:
            if prev.tag == f'{{{NS}}}p':
                for t in prev.iter(f'{{{NS}}}t'):
                    if t.text and t.text.strip().startswith("Table "):
                        has_caption = True
                        break
            if has_caption:
                break
            prev = prev.getprevious()
        tbl_elements.append((elem, has_caption))

# Remove orphaned tables (no caption)
removed = 0
for elem, has_caption in tbl_elements:
    if not has_caption:
        body.remove(elem)
        removed += 1

print(f"Removed {removed} orphaned tables")

# Also remove any remaining orphaned blank paragraphs between Ch4 and Ch5
doc.save(DST)
print(f"Saved to {DST}")

# Final stats
doc2 = docx.Document(DST)
print(f"Final paragraphs: {len(doc2.paragraphs)}")
print(f"Final tables: {len(doc2.tables)}")
