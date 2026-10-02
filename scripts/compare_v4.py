"""Compare v4_ch5 (user's) vs v4 (my scripted) for differences."""
import docx

a = docx.Document(r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v4_ch5.docx")
b = docx.Document(r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v4.docx")

out = open(r"c:\Projects\FYP Project\Report Drafts\_v4_compare.txt", "w", encoding="utf-8")

out.write(f"v4_ch5: {len(a.paragraphs)} paragraphs, {len(a.tables)} tables\n")
out.write(f"v4: {len(b.paragraphs)} paragraphs, {len(b.tables)} tables\n\n")

# Find chapter boundaries
def find_ch(p, name):
    for i, para in enumerate(p.paragraphs):
        if para.text.strip() == name:
            return i
    return None

for ver_name, doc in [("v4_ch5", a), ("v4", b)]:
    ch1 = find_ch(doc, "CHAPTER 1")
    ch4 = find_ch(doc, "CHAPTER 4")
    ch5 = find_ch(doc, "CHAPTER 5")
    ref = find_ch(doc, "REFERENCES")
    out.write(f"{ver_name}: Ch1=[{ch1}], Ch4=[{ch4}], Ch5=[{ch5}], REF=[{ref}]\n")

# Compare Chapter 1 formatting structure
out.write("\n=== Ch1 body para formatting (first para after 1.1) ===\n")
for ver_name, doc in [("v4_ch5", a), ("v4", b)]:
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip().startswith("The authentication of prophetic traditions"):
            pf = p.paragraph_format
            r = p.runs[0] if p.runs else None
            out.write(f"{ver_name} [{i}]:\n")
            out.write(f"  style={p.style.name}\n")
            out.write(f"  alignment={pf.alignment}\n")
            out.write(f"  line_spacing={pf.line_spacing}\n")
            out.write(f"  first_line_indent={pf.first_line_indent}\n")
            out.write(f"  space_before={pf.space_before}\n")
            out.write(f"  space_after={pf.space_after}\n")
            if r:
                out.write(f"  run_font={r.font.name}, run_size={r.font.size}, run_bold={r.font.bold}\n")
            out.write(f"  text={p.text[:120]}\n\n")
            break

# Check if v4_ch5 has italic runs in Ch5 (user may have applied formatting manually)
out.write("=== Ch5 italic check (first 5 paragraphs with Arabic terms) ===\n")
for ver_name, doc in [("v4_ch5", a), ("v4", b)]:
    ch5 = find_ch(doc, "CHAPTER 5")
    out.write(f"\n--- {ver_name} ---\n")
    count = 0
    for i in range(ch5, min(ch5 + 60, len(doc.paragraphs))):
        t = doc.paragraphs[i].text
        if any(kw in t for kw in ["tarājim", "Jarḥ", "Kutub", "sanad", "matn", "tadlīs"]):
            for j, r in enumerate(doc.paragraphs[i].runs):
                if r.italic:
                    out.write(f"  [{i}] italic run: {r.text[:60]}\n")
                    count += 1
            if count >= 15:
                break
    if count == 0:
        out.write("  (no italic runs found)\n")

out.close()
print("Done")
