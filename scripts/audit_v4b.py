import docx
doc = docx.Document(r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v4_ch5.docx")

with open(r"c:\Projects\FYP Project\Report Drafts\_v4_transition.txt", "w", encoding="utf-8") as f:
    # Check Chapter 4 ending and Chapter 5 starting
    f.write("=== TRANSITION: Chapter 4 → Chapter 5 ===\n")
    for i in range(447, 458):
        p = doc.paragraphs[i]
        t = p.text.strip()[:150] if p.text.strip() else "(empty)"
        f.write(f"[{i}] ({p.style.name}) {t}\n")

    # Check that REFERENCES is still intact
    f.write("\n=== REFERENCES ===\n")
    for i in range(506, 510):
        p = doc.paragraphs[i]
        t = p.text.strip()[:150] if p.text.strip() else "(empty)"
        f.write(f"[{i}] ({p.style.name}) {t}\n")

    # Verify Heading 1 size (14pt = 28 half-pts)
    f.write("\n=== HEADING 1 SIZE CHECK ===\n")
    h1 = doc.styles["Heading 1"]
    f.write(f"Heading 1: size={h1.font.size}, bold={h1.font.bold}\n")
    if h1.font.size:
        f.write(f"  In pts: {h1.font.size.pt}\n")

    # Check a Ch5 paragraph inherits Tahoma 11
    f.write("\n=== CHAPTER 5 PARAGRAPH FORMAT CHECK ===\n")
    # Check the first body paragraph of Ch5 [457]
    for idx in [457, 458, 461]:
        p = doc.paragraphs[idx]
        pf = p.paragraph_format
        f.write(f"\nPara [{idx}]: {p.text[:80]}\n")
        f.write(f"  style={p.style.name}, align={pf.alignment}, line_spacing={pf.line_spacing}\n")
        if p.runs:
            r = p.runs[0]
            f.write(f"  run: font={r.font.name}, size={r.font.size}, bold={r.font.bold}\n")

print("Written")
