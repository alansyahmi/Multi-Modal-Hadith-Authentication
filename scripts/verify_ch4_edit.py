import docx, re
doc = docx.Document(r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v4.docx")

out = open(r"c:\Projects\FYP Project\Report Drafts\_v4_ch4_verify.txt", "w", encoding="utf-8")

ch4_start = None
for i, p in enumerate(doc.paragraphs):
    if p.text.strip() == "CHAPTER 4":
        ch4_start = i
        break

out.write(f"Chapter 4 at [{ch4_start}]\n\n")
out.write("=== Updated paragraphs ===\n")
for i in range(ch4_start, ch4_start + 70):
    t = doc.paragraphs[i].text
    if "geographic" in t.lower() or "geospatial" in t.lower() or "nisba" in t.lower() or "8 structural" in t.lower():
        out.write(f"\n[{i}] style={doc.paragraphs[i].style.name}\n")
        # Show run-level detail
        for j, r in enumerate(doc.paragraphs[i].runs):
            italic = "ITALIC" if r.italic else "normal"
            out.write(f"  run[{j}] ({italic}): {r.text[:120]}\n")
        out.write(f"\n  [FULL TEXT]:\n  {t[:500]}\n\n")

# Also check formatting preserved
out.write("=== Formatting check ===\n")
for idx in [418, 425]:
    p = doc.paragraphs[idx]
    pf = p.paragraph_format
    out.write(f"\nPara [{idx}]: align={pf.alignment}, line_spacing={pf.line_spacing}, indent={pf.first_line_indent}\n")

out.close()
print("Done")
