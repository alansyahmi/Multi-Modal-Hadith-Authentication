"""
Build Chapter 4: Experiment Design as a .docx file — Humanized prose.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
import os

doc = Document()

# ── Styles ──
style = doc.styles['Normal']
font = style.font
font.name = 'Tahoma'
font.size = Pt(12)
style.paragraph_format.line_spacing = 1.5
style.paragraph_format.space_after = Pt(6)

def add_heading_txt(text, level):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = 'Tahoma'
    return h

def add_para(text, bold=False, italic=False, size=12):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = 'Tahoma'
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    return p

def add_table(headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.bold = True
                r.font.name = 'Tahoma'
                r.font.size = Pt(10)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = str(val)
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.name = 'Tahoma'
                    r.font.size = Pt(10)
    return table

# ════════════════════════════════════════════════════════════════
# CHAPTER 4
# ════════════════════════════════════════════════════════════════

add_heading_txt('CHAPTER 4: EXPERIMENTAL DESIGN', 1)

# ── 4.1 ──
add_heading_txt('4.1 Introduction', 2)

add_para(
    'This chapter lays out the experiments we designed to test each of the three research objectives — '
    'developing a quantitative reliability index from scholarly judgments (PO1), building a Knowledge Graph '
    'with GNN-based structural analysis (PO2), and evaluating the framework against established Hadith '
    'gradings (PO3). Every experiment is organised to mirror the three-stage pipeline from Chapter 3, so '
    'the link between method and objective stays clear throughout.'
)

add_para(
    'I cover the full experimental environment — hardware, software, datasets — before walking through '
    'each experiment in turn: what we set out to learn, how we designed the test, what parameters matter, '
    'and what we expected to find. Where the actual implementation ended up diverging from the idealised '
    'methodology, I flag it and explain why. The aim is to give anyone reading this a complete, repeatable '
    'account of every experiment.'
)

# ── 4.2 ──
add_heading_txt('4.2 Experimental Environment', 2)

add_para(
    'All experiments ran on a local development workstation with the specs below. When GPU training was '
    'needed for larger models, we supplemented with Google Colab\'s T4 runtime.'
)

add_heading_txt('4.2.1 Hardware Configuration', 3)

add_table(
    ['Component', 'Specification'],
    [
        ['CPU', 'Intel Core i7 (12th Gen), 8 cores / 16 threads'],
        ['RAM', '32 GB DDR4'],
        ['GPU (local)', 'NVIDIA GeForce RTX 3060 (12 GB VRAM) — used for small-scale testing'],
        ['GPU (cloud)', 'Google Colab T4 — used for large-scale GNN training and BERT inference'],
        ['Storage', '1 TB NVMe SSD'],
        ['OS', 'Windows 10 Home (22H2)'],
    ]
)

add_heading_txt('4.2.2 Software Configuration', 3)

add_table(
    ['Layer', 'Technology'],
    [
        ['Python', '3.11.7'],
        ['Deep Learning', 'PyTorch 2.1.2, torch-geometric 2.4.0'],
        ['Arabic NLP', 'CAMeL Tools 1.5.3, transformers 4.37.2'],
        ['Graph Processing', 'NetworkX 3.2.1, PyTorch Geometric'],
        ['Fuzzy Matching', 'rapidfuzz 3.6.0, scikit-learn 1.4.0'],
        ['Data Processing', 'pandas 2.1.5, numpy 1.26.2'],
        ['Visualisation', 'matplotlib 3.8.2, seaborn 0.13.1'],
        ['Geocoding (future)', 'Geopy 2.4.1'],
        ['Development', 'VS Code, Jupyter Notebook, Git'],
    ]
)

# ── 4.3 ──
add_heading_txt('4.3 Data Preparation Experiments', 2)

add_para(
    'Before any entity extraction or graph building could happen, the raw data had to be cleaned and cross-referenced. '
    'These first two experiments tackle exactly that — parsing biographical volumes from classical sources, then '
    'aligning Hadith records across two independently collected datasets. Both feed directly into PO1 and PO2.'
)

add_heading_txt('4.3.1 Experiment 1: Biographical Text Parsing', 3)

add_para(
    'What we wanted to know: Could a Regex-based parser reliably isolate individual biographical entries '
    '(tarājim) from the raw OpenITI volumes of Tahdhīb al-Kamāl and Taqrīb al-Tahdhīb?'
)

add_para('How we tested it', bold=True)

add_para(
    'The parser (src.shamela.narrator_parser.segment_volume) runs a multi-stage Regex pipeline over each raw volume. '
    'First it strips architectural metadata — YAML headers, page-number selectors, editorial comments. '
    'Then it hunts for entry boundaries using historical identifiers and narrator-name patterns. Every entry '
    'it finds gets a confidence label: high if the entry has a name plus teachers and grades, medium if the '
    'name comes with partial metadata, low if we only caught the name. We ran this against all 35 volumes '
    'of Tahdhīb al-Kamāl and the single Taqrīb al-Tahdhīb volume, then merged everything into one unified '
    'narrator encyclopedia.'
)

add_para('Key parameters', bold=True)
add_table(
    ['Parameter', 'Value', 'Why'],
    [
        ['Volume range', '1–35 (Tahdhīb) + 1 (Taqrīb)', 'Covers both biographical sources in full'],
        ['Entry IDs', 'Sequential, starting from 1', 'Lets us cross-reference across volumes'],
        ['Confidence criteria', 'High: name + teachers + grades; Medium: name + partial data; Low: name only', 'Gives downstream tasks a handle on data quality'],
        ['Text cap per entry', '3,000 characters', 'Balances storage cost against context needs'],
    ]
)

add_para('What we expected', bold=True)
add_para(
    'A structured JSON export of every extracted entry — entry ID, volume, name, full text, teachers, '
    'students, grades, death info, and confidence. Our target was high-confidence extraction for at least '
    '70% of entries. The rest would inevitably land in medium or low due to orthographic ambiguity and '
    'all the structural quirks you get in centuries-old manuscript transcriptions.'
)

add_heading_txt('4.3.2 Experiment 2: Two-Tier Alignment Framework', 3)

add_para(
    'What we wanted to know: could we reliably cross-reference Hadith records between the Sanadset 650K corpus '
    'and the HuggingFace Kutub as-Sitta dataset, despite them having no shared index?'
)

add_para('How we tested it', bold=True)

add_para(
    'The pipeline (src.alignment.hybrid_aligner) works exactly as described in Section 3.2.3. Tier 1 uses '
    'a sparse TF-IDF vectorizer with character n-grams to cut the search space from 650K candidates down '
    'to a top-10 shortlist per query. Tier 2 then runs rapidfuzz partial_ratio with a 95% threshold to '
    'confirm the match. Anything that falls short of that bar gets quarantined for manual review. '
    'We tested this on a stratified sample of 1,200 Hadiths — 200 from each of the six canonical books — '
    'with a fixed seed of 42 so the exact same subset appears every time.'
)

add_para('Key parameters', bold=True)
add_table(
    ['Parameter', 'Value', 'Why'],
    [
        ['TF-IDF n-gram range', '(3, 5) characters', 'Catches sub-word variations and common Arabic root patterns'],
        ['TF-IDF max features', '50,000', 'Wide vocabulary without blowing out memory'],
        ['Top-K candidates', '10', 'Brings the search space from 650K to single digits per query'],
        ['Fuzzy threshold', '≥ 95% partial_ratio', 'High precision; we tuned this empirically'],
        ['Stratified sample', '200/ book × 6 books = 1,200', 'Balanced representation across all Kutub as-Sitta'],
        ['Random seed', '42', 'Deterministic — rerun and get the same subset'],
        ['Record ID', 'SHA-256 of consonantal text skeleton', 'Immutable key per Section 3.2.3'],
    ]
)

add_para('What we expected', bold=True)
add_para(
    'A JSON file of alignment results for all 1,200 queries — similarity score, TF-IDF score, match status, '
    'and quarantine flag. We aimed for at least 75% success rate on matches, with under 5% of records '
    'sent to quarantine.'
)

# ── 4.4 ──
add_heading_txt('4.4 Entity Extraction and Disambiguation Experiments', 2)

add_para(
    'These experiments bridge Stages 1 and 2. Once the data is clean and aligned, the next question is how '
    'to identify narrators and link them to a canonical identity. That is the core of PO2.'
)

add_heading_txt('4.4.1 Experiment 3: BERT Embedding Extraction for Biographical Jarḥ wa Taʿdīl Features', 3)

add_para(
    'What we wanted to know: could deep contextual embeddings of the biographical text passages that contain '
    'Jarḥ wa Taʿdīl assessments give us useful semantic features for the GNN node representations?'
)

add_para('How we tested it', bold=True)

add_para(
    'The core methodological premise here is straightforward: the classical kutub ar-rijal contain qualitative '
    'judgements about each narrator\'s moral probity (ʿadāla) and mnemonic precision (ḍabṭ). These judgements '
    'are expressed through a highly technical vocabulary — Thiqa, Ṣadūq, Kaḏḏāb, Layyin — but also through '
    'longer contextual phrases that a simple keyword lookup might miss. The idea is to encode the full '
    'biographical text segment that contains the evaluative statement, rather than just the term itself.'
)

add_para(
    'The biographical entries parsed in Experiment 1 serve as the input corpus. For each narrator entry, '
    'the text segments that carry Jarḥ wa Taʿdīl content (typically preceded by phrases like "kāna..." or '
    'following the narrator\'s name) are extracted. These segments are fed through a CAMeL-BERT model '
    '(bert-base-arabic-camelbert-msa-sentiment) to produce [CLS] token embeddings that capture the semantic '
    'nuance of the scholarly assessment. The 768-dimensional output is PCA-reduced to 128 dimensions and '
    'attached as semantic node features in the Knowledge Graph, complementing the structural features derived '
    'from transmission network topology.'
)

add_para('Key parameters', bold=True)
add_table(
    ['Parameter', 'Value', 'Why'],
    [
        ['Model', 'CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment', 'Arabic-specific BERT; 110M params; pre-trained on classical + MSA'],
        ['Batch size', '128', 'Maximises CPU throughput'],
        ['Max length', '256 tokens', 'Sufficient for a typical Jarḥ wa Taʿdīl passage'],
        ['Input source', 'Biographical entries (tahdhib_al_kamal + taqrib_al_tahdhib)', 'Only the biographical texts, not Hadith matn (out of scope)'],
        ['Text segments', 'Phrases containing Jarḥ wa Taʿdīl terms + surrounding context (up to 5 per narrator)', 'Captures both the term and its qualifying context'],
        ['PCA target', '128 (from 768)', 'Balances semantic richness against graph feature dominance'],
    ]
)

add_para('What we expected', bold=True)
add_para(
    'An augmented graph file (pyg_graph_bert.npz) where each narrator node carries 8 graph-structural features '
    'plus 128-dimensional BERT embeddings of their biographical assessments — totalling 136 dimensions. '
    'Alongside this, a lexicon-only variant using the GRADE_SCORE_MAP (Experiment 7) for comparison in the '
    'GNN ablation study (Experiment 6).'
)

add_heading_txt('4.4.2 Experiment 4: Narrator Name Matching and Encyclopedia Coverage', 3)

add_para(
    'What we wanted to know: how well can we match Sanadset narrator names to the biographical encyclopedia?'
)

add_para('How we tested it', bold=True)

add_para(
    'We took narrators from the aligned Hadith chains, normalised them (strip diacritics, normalise alif variants, '
    'collapse yā and hā), and tried to find them in the narrator encyclopedia. Exact matches were accepted '
    'straight away. When a name did not match exactly, we fell back to fuzzy matching with rapidfuzz '
    'partial_ratio at a threshold of 80%. We tracked both the proportion of successful matches and, for the '
    'matched entries, whether the encyclopedia carried grades, death dates, and teacher-student lists.'
)

add_para('Key parameters', bold=True)
add_table(
    ['Parameter', 'Value', 'Why'],
    [
        ['Fuzzy threshold', '≥ 80% partial_ratio', 'Empirically balances recall against false positives'],
        ['Normalisation', 'Diacritic strip → alif norm → yā/hā norm', 'Same consonantal-skeleton approach as Stage 1'],
        ['Evaluation set', 'Unique narrators from aligned eval set', 'Roughly 1,200–1,500 narrators from Kutub as-Sitta'],
    ]
)

add_para('What we expected', bold=True)
add_para(
    'A JSON file (narrator_matches.json) with match status per narrator. Target: ≥ 80% overall match rate '
    '(exact + fuzzy combined) and at least 60% of those matched carrying grade annotations from Ibn Hajar\'s corpus.'
)

# ── 4.5 ──
add_heading_txt('4.5 Knowledge Graph Construction Experiments', 2)

add_heading_txt('4.5.1 Experiment 5: Graph Topology and Feature Engineering', 3)

add_para(
    'What we wanted to know: how does the scale of the transmission network affect graph properties and '
    'downstream GNN performance?'
)

add_para('How we tested it', bold=True)

add_para(
    'We built the graph by walking through Sanadset rows, pulling narrator names from each chain, and '
    'creating directed edges to represent teacher-student relationships (teacher → student). Node features '
    'include log-transformed appearance count, in-degree, out-degree, average chain position, and the '
    'grade score from Ibn Hajar\'s 12-tier mapping. Edges carry positional data and frequency counts. '
    'We varied the number of Sanadset rows from 10,000 to 100,000 to see how node count, edge density, '
    'and degree distribution change with scale.'
)

add_para('Grade scoring', bold=True)
add_para(
    'Ibn Hajar\'s 12-tier system maps to continuous scores in [0.0, 1.0] via src/stage2/config.py. '
    'Companions (Ṣaḥāba) get 1.0; Thiqa Thiqa (supremely reliable) gets 0.95; Thiqa (trustworthy) gets '
    '0.85; Ṣadūq (truthful) gets 0.70; and so on down to Kaḏḏāb (liar) and Waḍḍāʿ (fabricator) at 0.0. '
    'Narrators with unknown or missing grades default to 0.30.'
)

add_para('Key parameters', bold=True)
add_table(
    ['Parameter', 'Value', 'Why'],
    [
        ['Sanadset rows', '100,000', 'Dense enough topology without exceeding memory limits'],
        ['Max chain length', '30 narrators', 'Trims rare outliers while preserving typical chain lengths'],
        ['Name filter', '≥ 3 characters, excludes kinship terms', 'Filters out non-narrators like "his father", "his brother"'],
        ['Edge aggregation', 'Position list per edge pair', 'Enables richer edge-level features'],
    ]
)

add_para('What we expected', bold=True)
add_para(
    'A PyTorch Geometric graph file (pyg_graph_large.npz) with ~50K–60K nodes and ~200K–300K edges. '
    'The graded variant (pyg_graph_large_graded.npz) would put ground-truth labels in the first feature '
    'column for semi-supervised GNN training.'
)

# ── 4.6 ──
add_heading_txt('4.6 GNN Architecture and Training Experiments', 2)

add_heading_txt('4.6.1 Experiment 6: Graph Convolutional Network Ablation Study', 3)

add_para(
    'What we wanted to know: how much does adding semantic features from the biographical texts (BERT embeddings '
    'of Jarḥ wa Taʿdīl passages, GRADE_SCORE_MAP keyword scores) improve narrator reliability prediction over '
    'graph-structural features alone?'
)

add_para('How we tested it', bold=True)

add_para(
    'We built a two-layer GCN in PyTorch Geometric. The model reads node features and outputs a continuous '
    'reliability score in [0, 1] for each narrator. Three variants were compared: '
    '(a) graph-only — just the 8 structural features; '
    '(b) graph + BERT — structural features plus the 128-dimensional PCA-reduced embeddings; '
    '(c) graph + lexicon — structural features plus the GRADE_SCORE_MAP keyword score (1 feature from '
    'biographical Jarḥ wa Taʿdīl term extraction). '
    'Training is semi-supervised: 70% of nodes have their grade masked, and the model predicts them by '
    'propagating information from the 30% labelled nodes through the graph.'
)

add_para('Hyperparameters', bold=True)
add_table(
    ['Parameter', 'Value', 'Why'],
    [
        ['Hidden channels', '64', 'Fits the ~140-dim input without over-parameterising'],
        ['Layers', '2', 'Captures 2-hop neighbourhood; deeper would risk over-smoothing'],
        ['Dropout', '0.3', 'Standard GCN regularisation'],
        ['Learning rate', '0.001', 'Adam default; gave stable convergence'],
        ['Weight decay', '5 × 10⁻⁴', 'L2 regularisation to keep generalisation in check'],
        ['Max epochs', '200', 'Converges well within this; early stop at patience=30'],
        ['Loss', 'MSE', 'Continuous target in [0, 1]'],
        ['Train / val / test', '70% / 15% / 15%', 'Standard split for semi-supervised node regression'],
        ['Runs per variant', '3 (different seeds)', 'Average across runs for robustness'],
    ]
)

add_para('What we expected', bold=True)
add_para(
    'A clean comparison of MAE, R², Spearman and Pearson correlations, and binary accuracy (threshold at '
    '≥ 0.65 for "accepted"). The ablation would show us whether the extra semantic modalities add anything '
    'over the pure graph-structural signal.'
)

# ── 4.7 ──
add_heading_txt('4.7 Trust Scoring and Evaluation Experiments', 2)

add_para(
    'These experiments close the loop on PO1 and PO3 — taking scores from the GNN and turning them into '
    'something that maps onto the traditional Ṣaḥīḥ/Ḥasan/Ḍaʿīf/Mawḍūʿ categories.'
)

add_heading_txt('4.7.1 Experiment 7: Semantic Sentiment Quantisation of Jarḥ wa Taʿdīl Terms', 3)

add_para(
    'What we wanted to know: can the qualitative Jarḥ wa Taʿdīl terms extracted from the biographical '
    'encyclopedia be reliably mapped to continuous reliability scores that reflect the classical scholarly hierarchy?'
)

add_para('How we tested it', bold=True)

add_para(
    'This experiment operationalises the core idea behind PO1: translating the classical critics\' vocabulary '
    'into a numerical index. The GRADE_SCORE_MAP (defined in src/stage2/config.py) assigns a continuous score '
    'in [0.0, 1.0] to each term in Ibn Hajar al-ʿAsqalānī\'s 12-tier grading system. The mapping follows the '
    'classical hierarchy: terms of supreme praise (Thiqa Thiqa, Thiqa Ḥāfiẓ) sit at the top; terms of '
    'trustworthiness (Thiqa, Ma\'mūn) occupy the 0.85 band; Ṣadūq and its variants fall around 0.65–0.70; '
    'criticism terms like Ḍaʿīf sit at 0.15; and severe condemnations (Kaḏḏāb, Waḍḍāʿ, Dajāl) land at 0.0. '
    'Narrators with no explicit grade default to 0.30.'
)

add_para(
    'Two tracks are planned per the methodology in Section 3.4.1. Track B — the static keyword lookup using '
    'this mapping — is the one currently implemented. When the biographical parser extracts a grade term from '
    'a narrator\'s entry (e.g., "Thiqa" from Taqrīb al-Tahdhīb), the system looks it up in the map and '
    'assigns the corresponding score to that narrator node in the Knowledge Graph. If multiple assessments '
    'exist for the same narrator, the scores can be aggregated via weighted mean or Bayesian posterior '
    'estimation. Track A — a transformer-based (AraBERT v2) regression model that reads the full contextual '
    'biographical sentence rather than just the keyword — is planned for FYP2. The current experiment validates '
    'the Track B output distribution and establishes the baseline for comparison.'
)

add_para('Key parameters', bold=True)
add_table(
    ['Parameter', 'Value', 'Why'],
    [
        ['Grade lexicon', 'Ibn Hajar 12-tier map (GRADE_SCORE_MAP in src/stage2/config.py)', '53 distinct Arabic terms mapped to scores; covers all major Jarḥ wa Taʿdīl categories'],
        ['Score range', '0.0–1.0', 'Continuous index; enables fine-grained differentiation between narrators'],
        ['Default (unknown grade)', '0.30', 'Conservative midpoint between acceptable and weak'],
        ['Aggregation (future)', 'Weighted mean / Bayesian Beta posterior', 'Accounts for multiple assessments and scholar prominence'],
        ['Track A (future)', 'AraBERT v2 regression on full biographical sentence', 'Captures context, negations, and compound assessments that keyword lookup misses'],
    ]
)

add_heading_txt('4.7.2 Experiment 8: Chain-Level Reliability Scoring', 3)

add_para(
    'What we wanted to know: can the full pipeline — from sanad text to final Hadith grade — produce scores '
    'that agree with the canonical grades in Kutub as-Sitta?'
)

add_para('How we tested it', bold=True)

add_para(
    'The scoring pipeline (hadith_authenticator.py) works in four steps. First, it parses the sanad string '
    'into a list of narrator names. Second, it looks each name up in the Knowledge Graph to get the GNN-predicted '
    'reliability score. Third, it computes chain-level statistics — mean, minimum, variance. Fourth, it maps '
    'the aggregate to a grade using thresholds we calibrated against the 1,200-sample dataset from Experiment 2. '
    'We then compare predicted grades against the ground-truth Grade column in the HuggingFace data.'
)

add_para('Grade thresholds', bold=True)
add_table(
    ['Grade', 'Score Range', 'What It Means'],
    [
        ['Ṣaḥīḥ (Authentic)', 'Mean ≥ 0.80 AND min ≥ 0.70', 'High average reliability; no weak link in chain'],
        ['Ḥasan (Good)', 'Mean ≥ 0.65', 'Solid average with an acceptable weakest narrator'],
        ['Ḍaʿīf (Weak)', 'Mean ≥ 0.35', 'Low average or a weak narrator drags it down'],
        ['Mawḍūʿ (Fabricated)', 'Mean < 0.35', 'Critical weakness — a fabricator is present'],
    ]
)

add_para('What we expected', bold=True)
add_para(
    'For all 1,200 evaluation Hadiths, a predicted grade plus confidence score. The overall evaluation uses '
    'accuracy, macro-averaged F1, RMSE (against normalised ordinal grades), and Cohen\'s Kappa. A naive '
    'majority-class baseline would tell us whether the full pipeline is actually adding value.'
)

add_heading_txt('4.7.3 Experiment 9: Multi-Sanad Matn Analysis', 3)

add_para(
    'What we wanted to know: when the same matn appears with different narrator chains, how well do their '
    'scores agree — and does the aggregate give us something more reliable than any single chain?'
)

add_para('How we tested it', bold=True)

add_para(
    'We normalised matn texts from the Sanadset corpus and grouped them by SHA-256 hash. Groups that had '
    'three or more distinct narrator chains were pulled out for deeper analysis. Each chain was scored '
    'independently through the GNN pipeline. Then we computed aggregate scores four ways: simple mean '
    'of chain means, best-path (the strongest single chain), consensus score (which penalises high variance '
    'across chains), and worst-chain (weakest-link pessimism). Results got categorised into unanimous strong, '
    'unanimous weak, and conflicted groups.'
)

add_para('Key parameters', bold=True)
add_table(
    ['Parameter', 'Value', 'Why'],
    [
        ['Minimum chains per group', '3', 'Enough for meaningful variance statistics'],
        ['Matn normalisation', 'Full diacritic strip + character norm', 'Consistent with Stage 1 alignment'],
        ['Matn length filter', '≥ 80 characters', 'Filters out truncated or incomplete matns'],
        ['Sanadset rows scanned', '150,000', 'Wide enough to find multi-chain matns'],
    ]
)

# ── 4.8 ──
add_heading_txt('4.8 Summary', 2)

add_para(
    'What this chapter set out to do was lay a clean, repeatable foundation for every experiment in this project. '
    'Nine experiments were described in all, organised across the three pipeline stages: data preparation '
    '(Experiments 1–2), entity extraction and graph construction (Experiments 3–5), and GNN training plus '
    'evaluation (Experiments 6–9). Each one maps back to a specific objective, with the parameters, rationale, '
    'and expected outcomes spelled out explicitly.'
)

add_para(
    'The next chapter moves from design to execution. It presents what came out of Stages 1 and 2 — the data '
    'ingestion pipeline in action, alignment statistics, how the biographical corpus parser performed, the '
    'graph that emerged from the Sanadset chains, and the first training curves from the GNN.'
)

# ── Save ──
os.makedirs(r"c:\Projects\FYP Project\Report Drafts", exist_ok=True)
out_path = r"c:\Projects\FYP Project\Report Drafts\fyp_project_1_full_chapters_NEW.docx"
doc.save(out_path)
print(f"Saved: {out_path}")
