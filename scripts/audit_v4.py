import docx, sys
doc = docx.Document(r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v4_ch5.docx")

with open(r"c:\Projects\FYP Project\Report Drafts\_v4_audit.txt", "w", encoding="utf-8") as f:
    f.write(f"Total paragraphs: {len(doc.paragraphs)}\n")
    f.write(f"Total tables: {len(doc.tables)}\n\n")

    # Find Chapter 4 and Chapter 5 boundaries
    ch4_start = ch4_end = ch5_start = ch5_end = ref_idx = None
    for i, p in enumerate(doc.paragraphs):
        t = p.text.strip()
        if t == "CHAPTER 4":
            ch4_start = i
        if t == "CHAPTER 5":
            ch5_start = i
        if t == "REFERENCES":
            ref_idx = i
        if ch5_start is not None and ch5_end is None and i > ch5_start and t == "REFERENCES":
            ch5_end = i - 1
        if ch4_start is not None and ch4_end is None and i > ch4_start and t == "CHAPTER 5":
            ch4_end = i - 1

    f.write(f"CHAPTER 4: paragraphs [{ch4_start}–{ch4_end}]\n")
    f.write(f"CHAPTER 5: paragraphs [{ch5_start}–{ch5_end}]\n")
    f.write(f"REFERENCES: paragraph [{ref_idx}]\n\n")

    # List Chapter 5 paragraphs
    f.write("=== CHAPTER 5 PARAGRAPHS ===\n")
    for i in range(ch5_start, min(ch5_end + 1, len(doc.paragraphs))):
        p = doc.paragraphs[i]
        t = p.text.strip() if p.text else "(empty)"
        f.write(f"[{i}] ({p.style.name}) {t[:120]}\n")

print("Done")
