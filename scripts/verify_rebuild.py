import docx, re
doc = docx.Document(r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v4.docx")
out = open(r"c:\Projects\FYP Project\Report Drafts\_v4_rebuild_verify.txt", "w", encoding="utf-8")

def find_ch(name):
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip() == name and p.style.name == "Heading 1":
            return i
    return None

ch4 = find_ch("CHAPTER 4")
ch5 = find_ch("CHAPTER 5")
ref = find_ch("REFERENCES")

# ── 1. UK spelling check ──
out.write("=== UK SPELLING CHECK ===\n")
us_found = 0
for i in range(ch5, ref):
    t = doc.paragraphs[i].text
    # Flag US patterns
    for pat in ["normalize", "normalization", "vocalize", "characterize"]:
        if pat in t:
            out.write(f"  WARNING [{i}]: US '{pat}' in: {t[:100]}\n")
            us_found += 1
if us_found == 0:
    out.write("  ✓ No US spellings found\n\n")

# ── 2. Geospatial check in Ch5 ──
out.write("=== GEOSPATIAL MENTIONS (Ch5) ===\n")
for i in range(ch5, ref):
    t = doc.paragraphs[i].text
    if "geospatial" in t.lower():
        out.write(f"  [{i}] {t[:300]}\n\n")

# ── 3. Geospatial in FYP2 list check ──
out.write("=== 5.4.2 FYP2 LIST CHECK ===\n")
for i in range(ch5, ref):
    if "will be addressed in FYP2 by:" in doc.paragraphs[i].text:
        t = doc.paragraphs[i].text
        out.write(f"  [{i}] {t[:300]}\n\n")
        if "geospatial consistency verification" in t:
            out.write("  ✗ Still has geospatial in FYP2 list!\n")
        else:
            out.write("  ✓ Geospatial removed from FYP2 list\n")

# ── 4. Italic run check ──
out.write("=== ITALIC RUN SAMPLES ===\n")
checked = 0
for i in range(ch5, ref):
    p = doc.paragraphs[i]
    if p.style.name != "Normal" or not p.text.strip():
        continue
    for r in p.runs:
        if r.italic and len(r.text.strip()) > 2:
            out.write(f"  [{i}] italic: '{r.text.strip()[:50]}'\n")
            checked += 1
            if checked >= 20:
                break
    if checked >= 20:
        break
out.write(f"\n  ({checked} italic terms shown)\n\n")

# ── 5. Ch4 geospatial check ──
out.write("=== GEOSPATIAL MENTIONS (Ch4) ===\n")
for i in range(ch4, ch5):
    t = doc.paragraphs[i].text
    if any(kw in t.lower() for kw in ["geospatial", "geographic region", "geographic flag"]):
        out.write(f"  [{i}] {t[:300]}\n\n")

out.close()
print("Done")
