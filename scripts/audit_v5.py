"""Quick audit of v5."""
import docx
doc = docx.Document(r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v5.docx")
out = open(r"c:\Projects\FYP Project\Report Drafts\_v5_audit.txt", "w", encoding="utf-8")

out.write(f"Paragraphs: {len(doc.paragraphs)}\n")
out.write(f"Tables: {len(doc.tables)}\n\n")

def find_ch(name):
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip() == name and p.style.name == "Heading 1":
            return i
    return None

ch1 = find_ch("CHAPTER 1")
ch4 = find_ch("CHAPTER 4")
ch5 = find_ch("CHAPTER 5")
ref = find_ch("REFERENCES")

out.write(f"Ch1=[{ch1}], Ch4=[{ch4}], Ch5=[{ch5}], REF=[{ref}]\n\n")

# List heading structure from Ch4 onwards
out.write("=== STRUCTURE (Ch4 onwards) ===\n")
for i in range(ch4, min(ref + 5, len(doc.paragraphs))):
    p = doc.paragraphs[i]
    t = p.text.strip()
    s = p.style.name
    if s in ("Heading 1", "Heading 2", "Heading 3") or (s == "Normal" and t and len(t) > 50):
        out.write(f"[{i}] ({s}) {t[:120]}\n")
    elif s == "Normal" and t.startswith("•"):
        out.write(f"[{i}] ({s}) {t[:120]}\n")

# Check for dashes in Ch5
out.write("\n=== DASH CHECK (Ch5) ===\n")
dash_count = 0
for i in range(ch5, ref):
    t = doc.paragraphs[i].text
    if "—" in t or "–" in t:
        out.write(f"  [{i}] {t[:100]}\n")
        dash_count += 1
out.write(f"  Total dashes found: {dash_count}\n")

# Check for "we"/"I" in Ch5
out.write("\n=== PRONOUN CHECK (Ch5) ===\n")
pronoun_count = 0
for i in range(ch5, ref):
    t = doc.paragraphs[i].text.lower()
    for pronoun in [" we ", " we,", " we.", " i ", " i,", " i."]:
        if pronoun in t:
            out.write(f"  [{i}] {t[:100]}\n")
            pronoun_count += 1
out.write(f"  Total pronouns found: {pronoun_count}\n")

out.close()
print("Done")
