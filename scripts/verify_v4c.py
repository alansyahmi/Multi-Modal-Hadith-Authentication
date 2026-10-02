import docx
doc = docx.Document(r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v4.docx")

with open(r"c:\Projects\FYP Project\Report Drafts\_v4_italics.txt", "w", encoding="utf-8") as f:
    # Find Ch5 start
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip() == "CHAPTER 5":
            ch5_start = i
            break

    f.write(f"CHAPTER 5 starts at [{ch5_start}]\n\n")
    f.write("=== RUN-LEVEL FORMATTING CHECK ===\n\n")

    # Check specific paragraphs with italic terms
    targets = [
        "biographical entries",
        "Jarḥ wa Taʿdīl keywords",
        "three primary failure modes",
        "CAMeL-BERT",
        "edge risk modifiers",
        "evaluation against"
    ]

    seen = set()
    for i in range(ch5_start, min(ch5_start + 60, len(doc.paragraphs))):
        p = doc.paragraphs[i]
        t = p.text.strip()
        if not t:
            continue
        # Show run-level detail for paragraphs containing Arabic terms
        if any(kw in t for kw in ["tarājim", "Jarḥ", "Kutub", "sanad", "matn", "tadlīs"]):
            f.write(f"[{i}] {t[:60]}...\n")
            for j, run in enumerate(p.runs):
                italic = "ITALIC" if run.italic else "normal"
                f.write(f"  run[{j}]: {italic} | {run.text[:80]}\n")
            f.write("\n")

    # Check for US -> UK consistency
    f.write("\n=== UK SPELLING CHECK ===\n")
    us_patterns = ["normalize", "normalization", "vocalize", "vocalization", "characterize", "organization"]
    for i in range(ch5_start, min(ch5_start + 60, len(doc.paragraphs))):
        t = doc.paragraphs[i].text
        for pat in us_patterns:
            if pat in t:
                f.write(f"WARNING [{i}]: US spelling '{pat}' found\n")

    f.write("UK spelling check complete.\n")
