"""Final review of the edited report."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from docx import Document

doc = Document(
    r"c:\Projects\FYP Project\Report Drafts\new_report_draft_md\fyp1_full_report_v2_edited.docx"
)

# 1. Section numbering conflicts
print("=== SECTION NUMBERING ===")
seen_sections = set()
for i, p in enumerate(doc.paragraphs):
    t = p.text.strip()
    if p.style.name.startswith("Heading"):
        # Extract section number like "3.2", "4.1.1"
        import re
        m = re.match(r"^(\d+(?:\.\d+)*)\s", t)
        if m:
            sec = m.group(1)
            if sec in seen_sections:
                print(f"  DUPLICATE section {sec}: [{i}] {t[:80]}")
            seen_sections.add(sec)

# 2. Abstract check
print("\n=== ABSTRACT (first 300 chars) ===")
in_ab = False
for p in doc.paragraphs:
    t = p.text.strip()
    if "ABSTRACT" in t and p.style.name.startswith("Heading"):
        in_ab = True
        continue
    if "ABSTRAK" in t and p.style.name.startswith("Heading"):
        break
    if in_ab and t:
        print(f"  {t[:300]}")

# 3. LOT check
print("\n=== LOT ===")
in_lot = False
for p in doc.paragraphs:
    t = p.text.strip()
    if "LIST OF TABLES" in t:
        in_lot = True
        continue
    if "LIST OF FIGURES" in t:
        break
    if in_lot and t:
        print(f"  {t[:100]}")

# 4. LOF check
print("\n=== LOF ===")
in_lof = False
for p in doc.paragraphs:
    t = p.text.strip()
    if "LIST OF FIGURES" in t:
        in_lof = True
        continue
    if in_lof and t:
        if t.startswith("CHAPTER"):
            break
        print(f"  {t[:100]}")

# 5. Ch5 title
print("\n=== Ch5 TITLE ===")
for i, p in enumerate(doc.paragraphs):
    t = p.text.strip()
    if "CHAPTER 5" in t:
        print(f"  [{i}] {t}")
        # Next heading
        for j in range(i + 1, min(i + 5, len(doc.paragraphs))):
            nt = doc.paragraphs[j].text.strip()
            if nt:
                print(f"  [{j}] {nt}")
                break
        break

# 6. FYP1/FYP2 refs
print("\n=== FYP1/FYP2 references ===")
started = False
for i, p in enumerate(doc.paragraphs):
    t = p.text
    if "CHAPTER 1" in t:
        started = True
    if "REFERENCES" in t and p.style.name.startswith("Heading"):
        break
    if not started:
        continue
    if "FYP1" in t or "FYP2" in t:
        print(f"  [{i}] {t[:120].strip()}")

# 7. Unnumbered headings
print("\n=== UNNUMBERED HEADINGS ===")
for i, p in enumerate(doc.paragraphs):
    t = p.text.strip()
    if p.style.name.startswith("Heading"):
        import re
        if not re.match(r"^\d", t) and t not in [
            "DECLARATION", "ACKNOWLEDGEMENT", "ABSTRACT", "ABSTRAK",
            "LIST OF TABLES", "LIST OF FIGURES",
            "CHAPTER 1", "CHAPTER 2", "CHAPTER 3", "CHAPTER 4",
            "CHAPTER 5", "CHAPTER 6", "INTRODUCTION", "LITERATURE REVIEW",
            "PROJECT METHODOLOGY", "EXPERIMENT DESIGN",
            "PRELIMINARY IMPLEMENTATION", "CONCLUSION",
            "REFERENCES", "APPENDICES",
        ] and "RANGKA" not in t:
            print(f"  [{i}] ({p.style.name}) {t[:80]}")
