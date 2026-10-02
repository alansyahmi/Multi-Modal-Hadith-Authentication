"""Compare Ch5 text content between v4_ch5 (user's) and v4 (mine)."""
import docx

a = docx.Document(r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v4_ch5.docx")
b = docx.Document(r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v4.docx")

out = open(r"c:\Projects\FYP Project\Report Drafts\_v4_ch5_diff.txt", "w", encoding="utf-8")

def find_ch(doc, name):
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip() == name:
            return i
    return None

ch5_a = find_ch(a, "CHAPTER 5")
ch5_b = find_ch(b, "CHAPTER 5")

# Find end of Ch5 in both
ref_a = find_ch(a, "REFERENCES")
ref_b = find_ch(b, "REFERENCES")

out.write(f"v4_ch5 Ch5: [{ch5_a}] to [{ref_a - 1}]\n")
out.write(f"v4 Ch5: [{ch5_b}] to [{ref_b - 1}]\n\n")

# Compare paragraph by paragraph
out.write("=== PARAGRAPH-BY-PARAGRAPH COMPARISON ===\n\n")
min_len = min(ref_a - ch5_a, ref_b - ch5_b)

for offset in range(min_len):
    i = ch5_a + offset
    j = ch5_b + offset
    pa = a.paragraphs[i]
    pb = b.paragraphs[j]
    ta = pa.text.strip()
    tb = pb.text.strip()

    if ta != tb:
        out.write(f"\n[offset {offset}] v4_ch5[{i}] vs v4[{j}]:\n")
        out.write(f"  v4_ch5 style={pa.style.name}\n")
        out.write(f"  v4      style={pb.style.name}\n")
        if not ta:
            out.write("  v4_ch5: (empty)\n")
        else:
            out.write(f"  v4_ch5: {ta[:150]}\n")
        if not tb:
            out.write("  v4:     (empty)\n")
        else:
            out.write(f"  v4:     {tb[:150]}\n")

out.write(f"\n\n=== FULL v4_ch5 Ch5 TEXT ===\n")
for i in range(ch5_a, ref_a):
    t = a.paragraphs[i].text.strip()
    style = a.paragraphs[i].style.name
    if t:
        out.write(f"[{i}] ({style}) {t[:200]}\n")
    else:
        out.write(f"[{i}] ({style}) (empty)\n")

out.close()
print("Done")
