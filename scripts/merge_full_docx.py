"""
Replace old Ch4 with corrected version using direct XML insertion.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from lxml import etree

doc = Document(r"c:\Projects\FYP Project\Report Drafts\FYP Project 1 Revised Edition_UPDATED.docx")
body = doc.element.body
print(f"Original: {len(doc.paragraphs)} paragraphs, {len(doc.tables)} tables")

# ── Find old CHAPTER 4 and REFERENCES ──
children = list(body)
ch4_start = ref_idx = None

for i, child in enumerate(children):
    if child.tag == qn('w:p'):
        texts = child.findall('.//' + qn('w:t'))
        txt = ''.join(t.text or '' for t in texts).strip()
        if txt == 'CHAPTER 4' and ch4_start is None:
            ch4_start = i
        if txt == 'REFERENCES' and ref_idx is None:
            ref_idx = i

print(f"Old Ch4: element {ch4_start}, REFERENCES: element {ref_idx}")

# ── Collect elements to REPLACE ──
old_elements = children[ch4_start:ref_idx]  # Everything from CHAPTER 4 heading to just before REFERENCES
print(f"Elements to remove: {len(old_elements)}")

# ── Create replacement elements ──
new_elements = []

def mk_heading(text, level):
    """Create a heading element and append to new_elements list."""
    from lxml import etree
    p = doc.add_heading(text, level=level)
    # Let style inheritance handle formatting — no direct font overrides
    new_elements.append(p._element)

def mk_para(text, bold=False):
    """Create a paragraph element and append to new_elements list."""
    p = doc.add_paragraph()
    # Match original paragraph formatting: 1.5 line spacing, 12pt space before
    pf = p.paragraph_format
    pf.line_spacing = 1.5
    pf.space_before = Pt(12)
    # Don't set alignment — inherit from document style like Ch1
    run = p.add_run(text)
    # Let Normal style handle font/size — only override bold when needed
    if bold:
        run.bold = True
    new_elements.append(p._element)

def mk_table(headers, rows):
    """Create a table element and append to new_elements list."""
    table = doc.add_table(rows=1+len(rows), cols=len(headers))
    table.style = 'Table Grid'
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for cp in cell.paragraphs:
            for r in cp.runs:
                r.bold = True  # Keep header bold
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.rows[ri+1].cells[ci]
            cell.text = str(val)
            # No explicit font — inherit from Normal style
    new_elements.append(table._element)

# ═══════════════════════════════════════════════
# BUILD REPLACEMENT CHAPTER 4
# ═══════════════════════════════════════════════

mk_heading("CHAPTER 4: EXPERIMENTAL DESIGN", 1)
mk_heading("4.1 Introduction", 2)
mk_para("This chapter lays out the experiments we designed to test each of the three research objectives — developing a quantitative reliability index from scholarly judgments (PO1), building a Knowledge Graph with GNN-based structural analysis (PO2), and evaluating the framework against established Hadith gradings (PO3). Every experiment is organised to mirror the three-stage pipeline from Chapter 3, so the link between method and objective stays clear throughout.")
mk_para("I cover the full experimental environment — hardware, software, datasets — before walking through each experiment in turn: what we set out to learn, how we designed the test, what parameters matter, and what we expected to find. Where the actual implementation ended up diverging from the idealised methodology, I flag it and explain why. The aim is to give anyone reading this a complete, repeatable account of every experiment.")

mk_heading("4.2 Experimental Environment", 2)
mk_para("All experiments ran on a local development workstation with the specs below. When GPU training was needed for larger models, we supplemented with Google Colab’s T4 runtime.")

mk_heading("4.2.1 Hardware Configuration", 3)
mk_table(["Component", "Specification"], [
    ["CPU", "Intel Core i7 (12th Gen), 8 cores / 16 threads"],
    ["RAM", "32 GB DDR4"],
    ["GPU (local)", "NVIDIA GeForce RTX 3060 (12 GB VRAM)"],
    ["GPU (cloud)", "Google Colab T4"],
    ["Storage", "1 TB NVMe SSD"],
    ["OS", "Windows 10 Home (22H2)"],
])

mk_heading("4.2.2 Software Configuration", 3)
mk_table(["Layer", "Technology"], [
    ["Python", "3.11.7"],
    ["Deep Learning", "PyTorch 2.1.2, torch-geometric 2.4.0"],
    ["Arabic NLP", "CAMeL Tools 1.5.3, transformers 4.37.2"],
    ["Graph Processing", "NetworkX 3.2.1, PyTorch Geometric"],
    ["Fuzzy Matching", "rapidfuzz 3.6.0, scikit-learn 1.4.0"],
    ["Data Processing", "pandas 2.1.5, numpy 1.26.2"],
    ["Development", "VS Code, Jupyter Notebook, Git"],
])

mk_heading("4.3 Data Preparation Experiments", 2)
mk_para("Before any entity extraction or graph building could happen, the raw data had to be cleaned and cross-referenced. These first two experiments tackle exactly that — parsing biographical volumes from classical sources, then aligning Hadith records across two independently collected datasets. Both feed directly into PO1 and PO2.")

mk_heading("4.3.1 Experiment 1: Biographical Text Parsing", 3)
mk_para("What we wanted to know:", bold=True)
mk_para("Could a Regex-based parser reliably isolate individual biographical entries (tarajim) from the raw OpenITI volumes of Tahdhib al-Kamal (35 volumes) and Taqrib al-Tahdhib (1 volume)?")
mk_para("How we tested it", bold=True)
mk_para("The parser (src.shamela.narrator_parser.segment_volume) runs a multi-stage Regex pipeline over each raw volume. First it strips architectural metadata — YAML headers, page-number selectors, editorial comments. Then it identifies entry boundaries via historical identifiers and narrator-name patterns. Each extracted entry receives a confidence label: high (name + teachers + grades), medium (name + partial metadata), or low (name only). We ran this against all 35 volumes of Tahdhib al-Kamal and the single Taqrib al-Tahdhib volume, merging everything into a unified narrator encyclopedia.")
mk_para("Key parameters", bold=True)
mk_table(["Parameter", "Value", "Rationale"], [
    ["Volume range", "1–35 (Tahdhib) + 1 (Taqrib)", "Complete coverage of both biographical sources"],
    ["Confidence criteria", "High: name+teachers+grades; Med: name+partial; Low: name only", "Provides data quality signal for downstream tasks"],
    ["Text cap per entry", "3,000 characters", "Balances storage cost against context needs"],
])
mk_para("What we expected", bold=True)
mk_para("A structured JSON export of every extracted entry — entry ID, volume, name, full text, teachers, students, grades, death info, and confidence. Target: high-confidence extraction for at least 70% of entries.")

mk_heading("4.3.2 Experiment 2: Two-Tier Alignment Framework", 3)
mk_para("What we wanted to know:", bold=True)
mk_para("Could we reliably cross-reference Hadith records between the Sanadset 650K corpus and the HuggingFace Kutub as-Sitta dataset, despite them having no shared index?")
mk_para("How we tested it", bold=True)
mk_para("Tier 1 uses a sparse TF-IDF vectorizer with character n-grams (3-5) to reduce the search space from 650K candidates to a top-10 shortlist per query. Tier 2 applies rapidfuzz partial_ratio with a 95% threshold to confirm matches. Records below threshold are quarantined. We tested on a stratified sample of 1,200 Hadiths — 200 from each of the six canonical books — with a fixed seed of 42.")
mk_para("Key parameters", bold=True)
mk_table(["Parameter", "Value", "Rationale"], [
    ["TF-IDF n-gram range", "(3, 5) characters", "Catches sub-word variations and Arabic root patterns"],
    ["TF-IDF max features", "50,000", "Wide vocabulary within memory limits"],
    ["Top-K candidates", "10", "Search space reduction from 650K to 10 per query"],
    ["Fuzzy threshold", ">= 95% partial_ratio", "High precision; tuned empirically"],
    ["Stratified sample", "200 per book x 6 = 1,200", "Balanced across all Kutub as-Sitta"],
    ["Random seed", "42", "Deterministic subset every run"],
    ["Record ID", "SHA-256 of consonantal text skeleton", "Immutable primary key per Section 3.2.3"],
])
mk_para("What we expected", bold=True)
mk_para("A JSON file of alignment results for all 1,200 queries. Target: >= 75% match rate, < 5% quarantined.")

mk_heading("4.4 Entity Extraction and Disambiguation Experiments", 2)
mk_para("These experiments bridge Stages 1 and 2. Once the data is clean and aligned, the next question is how to identify narrators and link them to a canonical identity — the core of PO2.")

mk_heading("4.4.1 Experiment 3: BERT Embedding Extraction for Biographical Jarḥ wa Taʿdīl Features", 3)
mk_para("What we wanted to know:", bold=True)
mk_para("Could deep contextual embeddings of the biographical text passages that contain Jarḥ wa Taʿdīl assessments give us useful semantic features for GNN node representations?")
mk_para("How we tested it", bold=True)
mk_para("The classical kutub ar-rijal contain qualitative judgements about each narrator’s moral probity (ʿadāla) and mnemonic precision (ḍabṭ). These are expressed through a technical vocabulary — Thiqa, Ṣadūq, Kadhdhāb, Layyin — and through longer contextual phrases a simple keyword lookup might miss. The biographical entries parsed in Experiment 1 serve as input. Text segments carrying Jarḥ wa Taʿdīl content are extracted and fed through CAMeL-BERT to produce [CLS] token embeddings. The 768-dimensional output is PCA-reduced to 128 dimensions and attached as semantic node features in the Knowledge Graph.")
mk_para("Key parameters", bold=True)
mk_table(["Parameter", "Value", "Rationale"], [
    ["Model", "CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment", "Arabic-specific BERT; 110M params"],
    ["Input source", "Biographical entries (tahdhib + taqrib)", "Only biographical texts — NOT matn (out of scope)"],
    ["Text segments", "Jarḥ wa Taʿdīl phrases + context (up to 5/narrator)", "Captures both term and qualifying context"],
    ["PCA target", "128 (from 768)", "Balances sematic richness against graph feature dominance"],
])
mk_para("What we expected", bold=True)
mk_para("An augmented graph file (pyg_graph_bert.npz) with 8 structural + 128 BERT features per node (136 total). A lexicon-only variant using GRADE_SCORE_MAP for ablation comparison.")

mk_heading("4.4.2 Experiment 4: Narrator Name Matching and Encyclopedia Coverage", 3)
mk_para("What we wanted to know:", bold=True)
mk_para("How well can we match Sanadset narrator names to the biographical encyclopedia extracted from classical kutub ar-rijal?")
mk_para("How we tested it", bold=True)
mk_para("Narrator names from aligned Hadith chains were normalised (diacritic strip, alif variants, ya/ha collapse) and looked up in the encyclopedia. Exact matches accepted immediately. Fuzzy fallback via rapidfuzz partial_ratio at >= 80%. We tracked both match proportion and availability of grades, death dates, and teacher-student lists.")
mk_para("Key parameters", bold=True)
mk_table(["Parameter", "Value", "Rationale"], [
    ["Fuzzy threshold", ">= 80% partial_ratio", "Balances recall against false positives"],
    ["Normalisation pipeline", "Diacritic -> alif -> ya/ha", "Same consonantal-skeleton approach as Stage 1"],
    ["Evaluation set", "Unique narrators from aligned 1,200 Hadiths", "Approximately 1,200–1,500 narrators"],
])
mk_para("What we expected", bold=True)
mk_para("narrator_matches.json with match status per narrator. Target: >= 80% overall match rate, >= 60% with grade annotations.")

mk_heading("4.5 Knowledge Graph Construction Experiments", 2)
mk_heading("4.5.1 Experiment 5: Graph Topology and Feature Engineering", 3)
mk_para("What we wanted to know:", bold=True)
mk_para("How does the scale of the transmission network affect graph properties and downstream GNN performance?")
mk_para("How we tested it", bold=True)
mk_para("We built the graph from Sanadset chains as directed edges (teacher to student). Node features: log appearance count, in-degree, out-degree, average chain position, and grade score from Ibn Hajar’s 12-tier mapping. Edges carry positional data and frequency counts. We varied Sanadset rows from 10,000 to 100,000 to observe effects on graph properties.")
mk_para("Grade scoring", bold=True)
mk_para("Ibn Hajar’s 12-tier system maps to [0.0, 1.0] via src/stage2/config.py. Companions (Sahaba): 1.0; Thiqa Thiqa: 0.95; Thiqa: 0.85; Saduq: 0.70; Daʿif: 0.15; Kadhdhab/Waddaʿ: 0.0. Unknown grades default to 0.30.")
mk_para("Key parameters", bold=True)
mk_table(["Parameter", "Value", "Rationale"], [
    ["Sanadset rows", "100,000", "Dense topology within memory limits"],
    ["Max chain length", "30 narrators", "Trims rare outliers"],
    ["Name filter", ">= 3 chars, excludes kinship terms", "Filters non-narrator entities"],
])
mk_para("What we expected", bold=True)
mk_para("pyg_graph_large.npz with ~50K-60K nodes and ~200K-300K edges. The graded variant adds ground-truth labels for semi-supervised GNN training.")

mk_heading("4.6 GNN Architecture and Training Experiments", 2)
mk_heading("4.6.1 Experiment 6: Graph Convolutional Network Ablation Study", 3)
mk_para("What we wanted to know:", bold=True)
mk_para("How much do BERT embeddings of Jarḥ wa Taʿdīl passages and GRADE_SCORE_MAP keyword scores improve narrator reliability prediction over graph-structural features alone?")
mk_para("How we tested it", bold=True)
mk_para("A two-layer GCN in PyTorch Geometric predicts continuous reliability scores [0, 1] for each narrator. Three variants: (a) graph-only — 8 structural features; (b) graph + BERT — structural + 128-dim PCA embeddings; (c) graph + lexicon — structural + GRADE_SCORE_MAP keyword score. Training is semi-supervised: 70% of grades masked and predicted via neighbourhood propagation from 30% labelled nodes.")
mk_para("Hyperparameters", bold=True)
mk_table(["Parameter", "Value", "Rationale"], [
    ["Hidden channels", "64", "Fits ~140-dim input"],
    ["Layers", "2", "2-hop neighbourhood; deeper risks over-smoothing"],
    ["Dropout", "0.3", "Standard GCN regularisation"],
    ["Learning rate", "0.001", "Adam default; stable convergence"],
    ["Weight decay", "5 x 10^-4", "L2 regularisation"],
    ["Max epochs", "200 (early stop at 30 patience)", "Converges within this window"],
    ["Train/val/test", "70/15/15%", "Standard semi-supervised split"],
    ["Runs per variant", "3 (different seeds)", "Metrics averaged for robustness"],
])
mk_para("What we expected", bold=True)
mk_para("Comparison of MAE, R-squared, Spearman and Pearson correlations, and binary accuracy (threshold >= 0.65). The ablation quantifies the marginal contribution of semantic features over pure graph-structural signals.")

mk_heading("4.7 Trust Scoring and Evaluation Experiments", 2)
mk_para("These experiments close the loop on PO1 and PO3 — converting GNN scores into Sahih/Hasan/Daʿif/Mawduʿ categories.")

mk_heading("4.7.1 Experiment 7: Semantic Sentiment Quantisation of Jarḥ wa Taʿdīl Terms", 3)
mk_para("What we wanted to know:", bold=True)
mk_para("Can the qualitative Jarḥ wa Taʿdīl terms extracted from the biographical encyclopedia be reliably mapped to continuous reliability scores reflecting the classical scholarly hierarchy?")
mk_para("How we tested it", bold=True)
mk_para("This experiment operationalises the core idea behind PO1: translating the classical critics’ vocabulary into a numerical index. The GRADE_SCORE_MAP (src/stage2/config.py) assigns [0.0, 1.0] to each term in Ibn Hajar’s 12-tier system. Track B (currently implemented): the static keyword lookup assigns scores when the biographical parser extracts grade terms from narrator entries. Track A (AraBERT v2 regression on full biographical sentences — planned for FYP2) will capture context, negations, and compound assessments that keyword lookup misses.")
mk_para("Key parameters", bold=True)
mk_table(["Parameter", "Value", "Rationale"], [
    ["Grade lexicon", "Ibn Hajar 12-tier map (53 terms)", "All major Jarḥ wa Taʿdīl categories"],
    ["Score range", "0.0–1.0", "Continuous index for fine-grained differentiation"],
    ["Default (unknown grade)", "0.30", "Conservative midpoint between acceptable and weak"],
    ["Track A (planned)", "AraBERT v2 regression on biographical sentences", "Context-aware; captures negations and compound assessments"],
])

mk_heading("4.7.2 Experiment 8: Chain-Level Reliability Scoring", 3)
mk_para("What we wanted to know:", bold=True)
mk_para("Can the full pipeline — from sanad text to final Hadith grade — produce scores that agree with canonical Kutub as-Sitta grades?")
mk_para("How we tested it", bold=True)
mk_para("The pipeline (hadith_authenticator.py) parses the sanad, looks up each narrator in the KG for GNN-predicted scores, computes chain statistics (mean, min, variance), and maps to a grade using thresholds calibrated against the 1,200-sample dataset. We compare predicted grades against the ground-truth Grade column from HuggingFace data.")
mk_para("Grade thresholds", bold=True)
mk_table(["Grade", "Score Range", "Description"], [
    ["Sahih (Authentic)", "Mean >= 0.80 AND min >= 0.70", "High average; no weak link"],
    ["Hasan (Good)", "Mean >= 0.65", "Solid average with acceptable weakest narrator"],
    ["Daʿif (Weak)", "Mean >= 0.35", "Low average or weak narrator present"],
    ["Mawduʿ (Fabricated)", "Mean < 0.35", "Critical weakness — fabricator likely"],
])
mk_para("What we expected", bold=True)
mk_para("Predicted grades for all 1,200 evaluation Hadiths. Metrics: accuracy, macro F1, RMSE, Cohen’s Kappa. Baseline comparison against majority-class classifier.")

mk_heading("4.7.3 Experiment 9: Multi-Sanad Matn Analysis", 3)
mk_para("What we wanted to know:", bold=True)
mk_para("When the same matn appears with different narrator chains, how well do their scores agree — and does the aggregate give us something more reliable than any single chain?")
mk_para("How we tested it", bold=True)
mk_para("Matn texts from Sanadset were normalised and grouped by SHA-256 hash. Groups with >= 3 distinct narrator chains were analysed. Each chain scored independently via the GNN pipeline. Aggregate scores computed four ways: mean of means, best-path (strongest chain), consensus (penalising cross-chain variance), and worst-chain (weakest-link pessimism). Results categorised into unanimous strong, unanimous weak, and conflicted.")
mk_para("Key parameters", bold=True)
mk_table(["Parameter", "Value", "Rationale"], [
    ["Min chains per group", "3", "Sufficient for meaningful variance statistics"],
    ["Matn length filter", ">= 80 characters", "Filters truncated or incomplete matns"],
    ["Sanadset rows scanned", "150,000", "Wide enough net for multi-chain matns"],
])

mk_heading("4.8 Summary", 2)
mk_para("Nine experiments were described across the three pipeline stages: data preparation (Experiments 1–2), entity extraction and graph construction (Experiments 3–5), and GNN training plus evaluation (Experiments 6–9). Each maps to a specific objective with parameters, rationale, and expected outcomes explicitly stated.")
mk_para("The next chapter moves from design to execution. It presents the results of implementing Stages 1 and 2 — alignment statistics, biographical corpus parser performance, graph properties, and initial GNN training curves.")

print(f"Created {len(new_elements)} replacement elements")

# ═══════════════════════════════════════════════
# REPLACE old elements with new elements
# ═══════════════════════════════════════════════

# First, remove all old Ch4 elements from the body
for elem in old_elements:
    body.remove(elem)

# Now insert the new elements at the position where old Ch4 used to be
# The first element after old Ch4 was REFERENCES. It should still be there.
# Find the current REFERENCES element
ref_el = None
for child in body:
    if child.tag == qn('w:p'):
        texts = child.findall('.//' + qn('w:t'))
        txt = ''.join(t.text or '' for t in texts).strip()
        if txt == 'REFERENCES':
            ref_el = child
            break

print(f"REFERENCES found: {ref_el is not None}")

if ref_el is not None:
    # Use addprevious() to insert before REFERENCES
    # Insert in document order: each new element goes right before ref,
    # and the next one goes before ref again, after the prior insertion.
    for elem in new_elements:
        ref_el.addprevious(elem)
    print(f"Inserted {len(new_elements)} elements before REFERENCES")
else:
    print("ERROR: Could not find REFERENCES")
    sys.exit(1)

# ── Save ──
out_path = r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters.docx"
doc.save(out_path)
print(f"Saved: {out_path}")
print(f"Final: {len(doc.paragraphs)} paragraphs, {len(doc.tables)} tables")
