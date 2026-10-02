"""Quality check v5 against style guide."""
import docx
doc = docx.Document(r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v5.docx")
out = open(r"c:\Projects\FYP Project\Report Drafts\_v5_qc.txt", "w", encoding="utf-8")

def find_ch(name):
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip() == name and p.style.name == "Heading 1":
            return i
    return None

ch4 = find_ch("CHAPTER 4")
ch5 = find_ch("CHAPTER 5")
ref = find_ch("REFERENCES")

# 1. No dashes
dashes = 0
for i in range(ch4, ref):
    for ch in "—–":
        if ch in doc.paragraphs[i].text:
            dashes += 1
out.write(f"1. DASHES: {dashes} (0 = pass)\n")

# 2. No I/we
pronouns = 0
for i in range(ch4, ref):
    t = doc.paragraphs[i].text.lower()
    for p in [" we ", " we,", " i ", " i,", " i'm", " i've"]:
        pronouns += t.count(p)
out.write(f"2. PRONOUNS: {pronouns} (0 = pass)\n")

# 3. All body paras have runs
empty = 0
for i in range(ch4, ref):
    p = doc.paragraphs[i]
    if p.style.name == "Normal" and p.text.strip() and len(p.runs) == 0:
        empty += 1
out.write(f"3. EMPTY BODY PARAS: {empty} (0 = pass)\n")

# 4. Ch5 starts with correct intro
out.write("\n4. CH5 OPENING:\n")
for i in range(ch5, min(ch5 + 5, len(doc.paragraphs))):
    p = doc.paragraphs[i]
    if p.text.strip() and p.style.name == "Normal":
        out.write(f"   {p.text[:120]}...\n")
        break

# 5. Sample body content from mid-Ch4
out.write("\n5. CH4 SAMPLE (Exp 6):\n")
for i in range(ch4, ch5):
    if "4.6.1 Experiment 6" in doc.paragraphs[i].text:
        j = i + 1
        while j < min(i + 5, len(doc.paragraphs)):
            p = doc.paragraphs[j]
            if p.text.strip() and p.style.name == "Normal":
                out.write(f"   {p.text[:120]}...\n")
            j += 1
        break

# 6. Ch5 sample (5.2.2)
out.write("\n6. CH5 SAMPLE (5.2.2):\n")
for i in range(ch5, ref):
    if "5.2.2 Two-Tier" in doc.paragraphs[i].text:
        j = i + 1
        count = 0
        while j < min(i + 8, len(doc.paragraphs)) and count < 4:
            p = doc.paragraphs[j]
            if p.text.strip() and p.style.name == "Normal":
                out.write(f"   {p.text[:120]}...\n")
                count += 1
            j += 1
        break

out.close()
print("Done")
