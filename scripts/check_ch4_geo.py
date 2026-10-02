import docx
doc = docx.Document(r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v4.docx")
out = open(r"c:\Projects\FYP Project\Report Drafts\_ch4_geo.txt", "w", encoding="utf-8")

# Find Chapter 4
ch4_start = None
for i, p in enumerate(doc.paragraphs):
    if p.text.strip() == "CHAPTER 4":
        ch4_start = i
        break

out.write(f"Chapter 4 starts at [{ch4_start}]\n\n")
out.write("=== All geo/location mentions in Chapter 4 ===\n")
for i in range(ch4_start, ch4_start + 70):
    t = doc.paragraphs[i].text
    if any(kw in t.lower() for kw in ["geo", "location", "gazetteer", "nisba"]):
        out.write(f"[{i}] {t[:400]}\n\n")

out.write("\n=== Experiment 6 / GCN Ablation node features ===\n")
for i in range(ch4_start, ch4_start + 70):
    t = doc.paragraphs[i].text
    if "feature" in t.lower() and ("node" in t.lower() or "graph" in t.lower() or "dimension" in t.lower()):
        out.write(f"[{i}] {t[:300]}\n\n")

# Also check what scripts exist for geospatial
import os, glob
scripts_dir = r"c:\Projects\FYP Project\scripts"
out.write("\n=== Geospatial-related scripts ===\n")
for f in sorted(glob.glob(os.path.join(scripts_dir, "*geo*")) + glob.glob(os.path.join(scripts_dir, "*location*")) + glob.glob(os.path.join(scripts_dir, "*spatial*"))):
    out.write(f"{os.path.basename(f)}\n")

# Check output directories
out.write("\n=== Geospatial output files ===\n")
for root, dirs, files in os.walk(r"c:\Projects\FYP Project\output"):
    for f in files:
        if any(kw in f.lower() for kw in ["geo", "location", "spatial", "coord", "gazetteer"]):
            out.write(f"{root}\\{f}\n")

out.close()
print("Done")
