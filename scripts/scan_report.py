"""Scan DOCX for outdated references."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from docx import Document

doc = Document(
    r"c:\Projects\FYP Project\Report Drafts\new_report_draft_md\fyp1_full_report_v2.docx"
)

checks = [
    "AR-Sanad 280K",
    "AR‐Sanad 280K",
    "1500 narrator",
    "1,500 narrator",
    "~1,500",
    "PRELIMINARY IMPLEMENTATION",
    "Named Entity Recognition",
    "Jumh",
    "FYP1",
    "FYP2",
    "rule-based",
    "soft aggregation",
    "Graph Autoencoder",
    "GNN",
]

# Only scan main body (after LOF, before REFERENCES)
started = False
for i, p in enumerate(doc.paragraphs):
    t = p.text
    if "CHAPTER 1" in t:
        started = True
    if "REFERENCES" in t and p.style.name.startswith("Heading"):
        break
    if not started:
        continue
    for c in checks:
        if c.lower() in t.lower():
            print(f'  Found "{c}" at para {i}: {t[:150].strip()}')
            break
