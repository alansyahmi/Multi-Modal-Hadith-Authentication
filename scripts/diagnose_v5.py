"""Diagnose what happened to v5 body content."""
import docx

doc = docx.Document(r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v5.docx")
out = open(r"c:\Projects\FYP Project\Report Drafts\_v5_diagnose.txt", "w", encoding="utf-8")

# Find chapter boundaries
ch4 = ch5 = ref = None
for i, p in enumerate(doc.paragraphs):
    t = p.text.strip()
    if t == "CHAPTER 4": ch4 = i
    if t == "CHAPTER 5": ch5 = i
    if t == "REFERENCES" and p.style.name == "Heading 1": ref = i

out.write(f"Ch4=[{ch4}], Ch5=[{ch5}], REF=[{ref}]\n\n")

# Show ALL paragraphs from Ch4 to REF (with run count)
out.write("=== ALL PARAGRAPHS Ch4 to REF ===\n")
for i in range(ch4, ref + 5):
    p = doc.paragraphs[i]
    t = p.text.strip()
    s = p.style.name
    num_runs = len(p.runs)
    if t:
        out.write(f"[{i}] ({s}) runs={num_runs} | {t[:120]}\n")
    else:
        out.write(f"[{i}] ({s}) runs={num_runs} | (empty)\n")

out.close()
print("Done")
