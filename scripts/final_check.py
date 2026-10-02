"""Final comprehensive check of v4 document."""
import docx, re
doc = docx.Document(r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v4.docx")
out = open(r"c:\Projects\FYP Project\Report Drafts\_v4_final_check.txt", "w", encoding="utf-8")

def find_ch(name):
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip() == name and p.style.name == "Heading 1":
            return i
    return None

ch4, ch5, ref = find_ch("CHAPTER 4"), find_ch("CHAPTER 5"), find_ch("REFERENCES")

# 1. No US spellings in Ch5
out.write("1. UK SPELLING\n")
us = ["normalize", "normalization", "vocalize", "unvocalize", "characterize", "organization"]
found = False
for i in range(ch5, ref):
    for pat in us:
        if pat in doc.paragraphs[i].text.lower():
            out.write(f"   ✗ [{i}] US: '{pat}' in {doc.paragraphs[i].text[:80]}\n")
            found = True
if not found:
    out.write("   ✓ All UK\n")

# 2. No geospatial deferred to FYP2
out.write("\n2. GEOSPATIAL IN Ch5\n")
for i in range(ch5, ref):
    t = doc.paragraphs[i].text
    if "geospatial" in t.lower():
        if "location-based" in t and "not yet implemented" in t:
            out.write(f"   ✗ STILL HAS deferred geospatial! [{i}]\n")
        else:
            out.write(f"   ✓ [{i}]: {t[:150]}...\n")

# 3. Multi-word italic terms are single runs
out.write("\n3. MULTI-WORD ITALIC RUNS\n")
mw_terms = ["Kutub as-Sitta", "Jarḥ wa Taʿdīl", "Tahdhīb al-Kamāl"]
for i in range(ch5, ref):
    t = doc.paragraphs[i].text
    for mw in mw_terms:
        if mw in t:
            # Check if it appears as a single italic run
            for r in doc.paragraphs[i].runs:
                if r.italic and mw in r.text:
                    out.write(f"   ✓ [{i}] '{mw}' is one italic run\n")
                    break
            else:
                out.write(f"   ? [{i}] '{mw}' exists in text but not as single italic run\n")

# 4. Ch4 geospatial mentions
out.write("\n4. CH4 GEOSPATIAL\n")
for i in range(ch4, ch5):
    t = doc.paragraphs[i].text
    if any(kw in t for kw in ["geospatial", "geographic region", "geographic flag"]):
        out.write(f"   ✓ [{i}]: {t[:120]}...\n")

# 5. Overall stats
out.write(f"\n5. STATS\n")
out.write(f"   Paragraphs: {len(doc.paragraphs)}\n")
out.write(f"   Tables: {len(doc.tables)}\n")
out.write(f"   Ch5: [{ch5}] to [{ref-1}]\n")

out.close()
print("Done")
