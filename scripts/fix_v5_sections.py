"""
Fix remaining Ch4 issues in v5:
1. Broken Experiment 8/9 headings (list items got Heading 2 style)
2. Unrewritten sections (4.4, 4.4.2, 4.5, 4.6, 4.6.1, 4.7, 4.7.1, 4.7.2, 4.7.3)
"""
import docx
from lxml import etree

DST = r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_v5.docx"
doc = docx.Document(DST)
NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# ── 1. Fix broken Heading 2 paragraphs that should be Normal ──
# These are bullet points and list items from Experiment 8/9 that got Heading 2 style
fix_headings = {
    "• 0.70 to 0.84: Marginally Accepted.": True,
    "• 0.50 to 0.69: Uncertain.": True,
    "• Below 0.50: Rejected (Ḍaʿīf).": True,
    "• Minimum chains per matn: 3.": True,
    "• Convergence metric: Standard deviation and coefficient of variation.": True,
}

for i, p in enumerate(doc.paragraphs):
    t = p.text.strip()
    if t in fix_headings and p.style.name.startswith("Heading"):
        p.style = doc.styles["Normal"]
        print(f"  Fixed [{i}] style: Heading -> Normal")

# Also fix any experimental-question-like paragraphs that got Heading style
for i, p in enumerate(doc.paragraphs):
    t = p.text.strip()
    if t.startswith("Experimental question:") and p.style.name.startswith("Heading"):
        p.style = doc.styles["Normal"]
        print(f"  Fixed [{i}] experimental question style: Heading -> Normal")
    if t.startswith("Expected output:") and p.style.name.startswith("Heading"):
        p.style = doc.styles["Normal"]
        print(f"  Fixed [{i}] expected output style: Heading -> Normal")

# ── 2. Rewrite remaining Ch4 sections that have old content ──
rewrites = {
    "4.2.1 Hardware Configuration": (
        "The experiments ran on a local workstation with an AMD Ryzen 5 5600X processor (6 cores, 12 threads), "
        "32 GB of DDR4 RAM, and an NVIDIA GeForce RTX 3060 GPU with 12 GB of VRAM."
    ),
    "4.2.2 Software Configuration": (
        "The software environment uses Python 3.13 with PyTorch 2.x, PyTorch Geometric, and the CAMeL Tools library."
    ),
    "4.4 Entity Extraction and Disambiguation Experiments": (
        "These experiments bridge Stages 1 and 2. Once the data is clean and aligned, the system needs to "
        "identify narrators within the chains and link them to the biographical encyclopedia entries."
    ),
    "4.4.2 Experiment 4: Narrator Name Matching and Encyclopaedia Coverage": (
        "Experimental question: How well does the system match Sanadset narrator names to the biographical "
        "encyclopedia entries?",
        "The system normalises each narrator name by stripping diacritics and collapsing alif variants, then "
        "checks for exact matches. Unmatched names fall back to fuzzy matching using the rapidfuzz "
        "partial_ratio function with an 80% threshold.",
        "Key parameters for this experiment were set as follows:",
        "• Fuzzy threshold: 80%.",
        "• Normalisation: Diacritic stripping, alif variant collapsing, yā and hā collapsing.",
        "• Matching library: rapidfuzz partial_ratio.",
        "Expected output: A JSON file (narrator_matches.json) recording the match status for each narrator. "
        "The biographical encyclopedia serves as the reference database, and matched narrators are the only "
        "ones that receive grade scores in the Knowledge Graph."
    ),
    "4.5.1 Experiment 5: Graph Topology and Feature Engineering": (
        "Experimental question: How does the scale of the transmission network affect graph properties and "
        "downstream GNN performance?",
        "The system builds the graph by walking through Sanadset rows, pulling narrator names from each chain, "
        "and creating directed edges to represent teacher-student relationships. Node features include "
        "log-transformed appearance count, in-degree, out-degree, average chain position, the Ibn Hajar grade "
        "score, and a binary geographic region flag for geospatial consistency verification.",
        "Key parameters for this experiment were set as follows:",
        "• Source rows: 100,000 Sanadset records.",
        "• Grade system: Ibn Hajar's 12-tier system mapped to continuous scores in [0.0, 1.0].",
        "• Score assignment: GRADE_SCORE_MAP from src/stage2/config.py.",
        "Expected output: A PyTorch Geometric graph file (pyg_graph_large.npz) and a graded variant "
        "(pyg_graph_large_graded.npz) for semi-supervised GNN training."
    ),
    "4.6 GNN Architecture and Training Experiments": (
        "These experiments test the GNN components described in Stage 3 of the methodology."
    ),
    "4.6.1 Experiment 6: Graph Convolutional Network Ablation Study": (
        "Experimental question: How much does adding semantic features improve narrator reliability prediction "
        "over graph-structural features alone?",
        "The system uses a two-layer GCN built in PyTorch Geometric. It reads node features and outputs a "
        "continuous reliability score in [0.0, 1.0]. Three variants are compared: graph-only with 8 structural "
        "features (degree centrality, appearance count, chain position, grade score, and a binary geographic "
        "region flag); graph plus BERT with structural features plus 128-dimensional PCA-reduced embeddings; "
        "and graph plus lexicon with structural features plus 7 lexicon sentiment features.",
        "Hyperparameters for this experiment were set as follows:",
        "• Hidden channels: 64.",
        "• Layers: 2.",
        "• Dropout: 0.3.",
        "• Learning rate: 0.001.",
        "• Epochs: 200 with early stopping at patience 30.",
        "Expected output: A comparison of MAE, R², Spearman and Pearson correlations, and binary accuracy "
        "(threshold at 0.65) across all three variants."
    ),
    "4.7 Trust Scoring and Evaluation Experiments": (
        "These experiments close the loop on PO1 and PO3. They take scores from the GNN and compare them "
        "against established Hadith classifications from the Kutub as-Sitta dataset."
    ),
    "4.7.1 Experiment 7: Semantic Sentiment Quantisation of Jarḥ wa Taʿdīl Terms": (
        "Experimental question: Can the qualitative Jarḥ wa Taʿdīl terms from the biographical encyclopedia "
        "be mapped to a continuous numerical scale that matches established Hadith grades?",
        "This experiment compares two tracks. Track A uses a fine-tuned AraBERT v2 model trained as a "
        "regression model on biographical sentences. Track B uses the GRADE_SCORE_MAP keyword lookup table "
        "defined in src/stage2/config.py.",
        "Key parameters for this experiment were set as follows:",
        "• Track A model: AraBERT v2 with regression head.",
        "• Track B mapping: GRADE_SCORE_MAP from config.py.",
        "• Evaluation metric: Cohen's Kappa agreement with established grades.",
        "• Validation set: 200 held-out Hadiths.",
        "Expected output: A comparison of Kappa scores for both tracks and a recommendation for which "
        "track to integrate into the GNN pipeline."
    ),
    "4.7.2 Experiment 8: Chain-Level Reliability Scoring": (
        "Experimental question: Can the full pipeline from sanad text to final Hadith grade produce scores "
        "that agree with established scholar classifications?",
        "The scoring pipeline (hadith_authenticator.py) processes each chain through four steps. It parses "
        "the sanad string into individual narrator names. It looks up each narrator in the Knowledge Graph. "
        "It aggregates individual reliability scores using the multiplicative model from Section 3.4.3. "
        "It produces a chain-level verdict with confidence metrics.",
        "Grade thresholds for classification are set as follows:",
        "• 0.85 and above: Accepted (Ṣaḥīḥ or Ḥasan).",
        "• 0.70 to 0.84: Marginally Accepted.",
        "• 0.50 to 0.69: Uncertain.",
        "• Below 0.50: Rejected (Ḍaʿīf).",
        "Expected output: For all 1,200 evaluation Hadiths, a predicted grade plus confidence score. "
        "Overall evaluation uses accuracy, precision, recall, and F1-score against ground-truth Kutub "
        "as-Sitta classifications."
    ),
    "4.7.3 Experiment 9: Multi-Sanad Matn Analysis": (
        "Experimental question: When the same matn appears with different narrator chains, how well do "
        "their scores converge?",
        "The system normalises matn texts and groups them by SHA-256 hash. Groups with three or more "
        "distinct chains are selected for analysis. For each group, the system computes the standard "
        "deviation and coefficient of variation of the chain-level scores.",
        "Key parameters for this experiment were set as follows:",
        "• Minimum chains per matn: 3.",
        "• Convergence metric: Standard deviation and coefficient of variation.",
        "Expected output: A statistical summary showing score variance within each matn group. Low "
        "variance across independent chains suggests internal consistency in the scoring model."
    ),
}

def set_para_text(para, text):
    for r in list(para.runs):
        r._element.getparent().remove(r._element)
    para.add_run(text)

# Apply rewrites
for i, p in enumerate(doc.paragraphs):
    t = p.text.strip()
    if t in rewrites and p.style.name in ("Heading 2", "Heading 3"):
        new_lines = rewrites[t]
        if isinstance(new_lines, str):
            new_lines = [new_lines]
        print(f"  Rewriting '{t[:50]}...' with {len(new_lines)} lines")
        # Find the body paragraphs after this heading
        body_paras = []
        j = i + 1
        while j < len(doc.paragraphs):
            np = doc.paragraphs[j]
            if np.style.name in ("Heading 2", "Heading 3") and np.text.strip():
                break
            if np.style.name == "Normal":
                body_paras.append(j)
            j += 1

        if body_paras:
            # Replace first body para
            first = body_paras.pop(0)
            set_para_text(doc.paragraphs[first], new_lines[0])

            # Insert remaining lines after it
            ref_elem = doc.paragraphs[first]._element
            body = doc.element.body
            for line in new_lines[1:]:
                new_p = docx.oxml.OxmlElement("w:p")
                new_r = docx.oxml.OxmlElement("w:r")
                new_t = docx.oxml.OxmlElement("w:t")
                new_t.text = line
                new_t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
                new_r.append(new_t)
                new_p.append(new_r)
                # Add Normal style
                pPr = docx.oxml.OxmlElement("w:pPr")
                pStyle = docx.oxml.OxmlElement("w:pStyle")
                pStyle.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', 'Normal')
                pPr.append(pStyle)
                new_p.insert(0, pPr)
                ref_elem.addnext(new_p)
                ref_elem = new_p

            # Remove old body paragraphs
            for idx in reversed(body_paras):
                doc.element.body.remove(doc.paragraphs[idx]._element)

            print(f"    Replaced {len(body_paras) + 1} old paragraphs")

doc.save(DST)
print("\nSaved")
